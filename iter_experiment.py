from sklearn.datasets import fetch_openml
from bayesian_algorithms import skopt, discretize, gpyopt
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import get_score
import config
import multiprocessing as mp
import approaches

def experiment_bayesian_iter_performance():
    """ Runs bayesian optimization to compare the performance depending on the iteration steps for all datasets/estimators/metrics.
    Attention:  - all approaches run with fixed number of to be selected features (most practice relevant)
                - Training score is the actual final function value used for optimization (might be 1 for "round" or "probabilistic_round" because many features can be selected)
                - Testing score is the score on the test data based on the n highest features (of the final vector)
    """

    for task, dataset in config.data_ids.items():
        for dataset_id, flag in dataset.items():
            if flag == True:
                # import dataset
                data, target = fetch_openml(
                    data_id=dataset_id, return_X_y=True, as_frame=True)

                # create multiprocess pool
                pool = mp.Pool(processes=config.n_processes)
                mp_results = []

                # use kfold cross validation
                kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)

                if task == "classification":
                    estimators = approaches.classification_estimators
                else:
                    estimators = approaches.regression_estimators

                # add tasks to multiprocessing pipeline
                for train_index, test_index in kf:
                    for estimator, metrics in estimators.items():
                        for metric, _ in metrics.items():
                            for learning_method, _ in approaches.learning_methods.items():
                                for discretization_method, _ in approaches.discretization_methods.items():
                                    for acq, _ in approaches.acquisition_functions.items():
                                        # TODO: currently final feature set is always limited, maybe also include case where bay opt selects an undefined number of features
                                        for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                            if learning_method == "GP":
                                                for kernel, _ in approaches.kernels.items():
                                                    if kernel == "HAMMING":
                                                        # hamming kernel only for categorical and binary search-spaces
                                                        if discretization_method == "categorical" or discretization_method == "binary":
                                                            mp_results.append((dataset_id, estimator, metric, learning_method, kernel, discretization_method, acq, n_features, test_index, pool.apply_async(
                                                            skopt, args=(data.loc[train_index], target.loc[train_index], n_features, kernel, learning_method, discretization_method, estimator, metric, acq, config.max_calls, True))))
                                                    else:
                                                        mp_results.append((dataset_id, estimator, metric, learning_method, kernel, discretization_method, acq, n_features, test_index, pool.apply_async(
                                                        gpyopt, args=(data.loc[train_index], target.loc[train_index], n_features, kernel, learning_method, discretization_method, estimator, metric, acq, config.max_calls, True))))
                                            else:
                                                # random forest only for categorical or binary search-spaces
                                                if learning_method == "categorical" or learning_method == "binary":
                                                    mp_results.append((dataset_id, estimator, metric, learning_method, "-", discretization_method, acq, n_features, test_index, pool.apply_async(
                                                        skopt, args=(data.loc[train_index], target.loc[train_index], n_features, None, learning_method, discretization_method, estimator, metric, acq, config.max_calls, True))))

                # get finished tasks (display tqdm progressbar)
                results = [tuple(r[0:9]) + tuple([r[9].get()]) for r in tqdm(mp_results)]

                # store in pandas dataframe
                list_columns = ["Dataset ID", "Estimator", "Metric", "Learning Method", "Kernel", "Discretization Method", "Acquisition Function", "Number Features", "Iteration Steps"]

                # extract data and create dataframe
                df = []
                for dataset_id, estimator, metric, learning_method, kernel, discretization_method, acq, n_features, test_index, (_, vector_list, fun_list) in results:
                    vector_list = vector_list
                    current_max_training_score = 0
                    current_max_test_score = 0
                    max_vector = []
                    for n_calls in range(0,len(vector_list),1):
                        # if new maximum is found at iteration step, add to result dataframe
                        training_score = fun_list[n_calls]
                        if current_max_training_score < training_score:
                            current_max_training_score = training_score
                            # calculate current max feature vector
                            if n_features is None:
                                max_vector = vector_list[n_calls]
                            else:
                                max_vector = discretize(vector_list[n_calls], "n_highest", n_features)
                            # calculate test score based on vector that has max training score
                            current_max_test_score = get_score(data.loc[test_index], target.loc[test_index], max_vector, estimator, metric)
                        df.append([dataset_id, estimator, metric, learning_method, kernel, discretization_method, acq, n_features, n_calls, current_max_training_score, current_max_test_score])

                df_result = pd.DataFrame(df, columns=list_columns+["Training Score", "Testing Score"])
                df_result_grouped = df_result.groupby(list_columns, as_index=False).agg(
                    {"Training Score": ["mean"], "Testing Score": ["mean"]})
                df_result_grouped_ordered = df_result_grouped.sort_values(by=["Dataset ID", "Estimator", "Metric", "Learning Method", "Kernel", "Discretization Method", "Acquisition Function", "Number Features"], ascending=True) 
                df_result_grouped_ordered.to_csv(
                    "results/iteration_number_experiment/bay_opt_iterations_" + str(dataset_id) + ".csv", index=False)

experiment_bayesian_iter_performance()