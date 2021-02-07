from bayesian_algorithms import skopt
from comparison_algorithms import rfe, sfs, sfm, vt, skb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import get_score, add_testing_score
import config
import multiprocessing as mp
import pandas as pd


# Estimator and metric properties (choosing estimator cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
classification_estimators = {
    "svc_linear": {
        "accuracy": "Support Vector Classification - Accuracy Score"
    }
    # "k_neighbours_classifier": { # results in error message
    #    "accuracy": "k Neighbours Classification - Accuracy Score"
    # }
}
regression_estimators = {
    "linear_regression": {
        "r2": "Linear Regression - Coefficient of Determination",
        "explained_variance": "Linear Regression - Explained variance regression score function"
    },
    "svr_linear": {
        "r2": "Support Vector Regression - Coefficient of Determination",
        "explained_variance": "Support Vector Regression - Explained variance regression score function"
    }
}

# Bayesian Optimization Properties
bay_opt_parameters = ["Approach", "Learning Method",
                      "Kernel", "Discretization Method", "n_features"]
bayesian_approaches = {
    skopt: "Scikit Optimize"
}
learning_methods = {
    "GP": "Gaussian Process",
    "RF": "Random Forest",
    "ET": "Extra Trees",
    "GBRT": "Gradient Boosting Quantile"
}
kernels = {
    # kernels for Gaussian Processes only
    "MATERN": "Matern",
    "HAMMING": "Hamming",
    "RBF": "Radial Basis Functions"  # squared-exponential kernel
}
discretization_methods = {
    "round": "round",
    "n_highest": "n highest",
    "binary": "binary"
}

# Comparison Approaches Properties
comparison_parameters = ["Approach", "Algorithm", "n_features"]
comparison_approaches = {
    "filter": {
        vt: "Variance Threshold",
        skb: "SelectKBest"
    },
    "wrapper": {
        # sfs: "Sequential Feature Selection" # bad performance when many features
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm: "Select From Model"
    }
}


def init_progress_bar():
    """ Initialize progress bar (one step for each execution of a comparison or bayesian approach).

    """
    # calculate number of datasets
    number_datasets_classification = 0
    number_datasets_regression = 0
    for type, iter in config.data_ids.items():
        if type == "classification":
            for _, flag in iter.items():
                if flag == True:
                    number_datasets_classification += 1
        else:
            for _, flag in iter.items():
                if flag == True:
                    number_datasets_regression += 1

    # calculate number of estimators/metrics
    number_regression_estimators = 0
    for _, iter in regression_estimators.items():
        for _, _ in iter.items():
            number_regression_estimators += 1
    number_classification_estimators = 0
    for _, iter in classification_estimators.items():
        for _, _ in iter.items():
            number_classification_estimators += 1

    # calculate numbe rof dataset/estimator combinations
    number_datasets_and_estimators = ((number_datasets_classification * number_classification_estimators
                                       ) + (number_datasets_regression * number_regression_estimators)) * config.n_splits

    # calculate number of bayesian approaches
    number_of_bayesian = (len(bayesian_approaches) * len(discretization_methods)) * (
        (len(learning_methods)-1) + len(kernels))  # only gaussian processes use kernels

    # calculate number of comparison approaches
    number_of_comparison = 0
    for _, approach in comparison_approaches.items():
        for _, _ in approach.items():
            number_of_comparison += 1

    # calculate total progress bar steps
    progress_total = number_datasets_and_estimators * \
        (number_of_bayesian + number_of_comparison)
    pbar = tqdm(total=progress_total)
    pbar.set_description("Processed")

    return pbar


def __run_all_bayesian(data_training, data_test, target_training, target_test, estimator, metric, n_calls, queue):
    """ Run all bayesian optimization approaches with all possible parameters.

    Keyword arguments:
    data_training -- feature matrix of training data
    data_test -- feature matrix of test data
    target_training -- target vector of training data
    target_test -- target vector of test data
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score
    queue -- queue to synchronize progress bar

    """
    nr_of_features = len(data_training.columns)
    # Define result dataframes
    df_results = pd.DataFrame(
        columns=bay_opt_parameters+["Vector", "Training Score"])
    for algo, algo_descr in bayesian_approaches.items():
        for learn, learn_descr in learning_methods.items():
            for discr, discr_descr in discretization_methods.items():
                if learn == "GP":
                    for kernel, kernel_descr in kernels.items():
                        if discr == "n_highest":
                            # TODO: with more than one n_features
                            n_features = round(nr_of_features/2)
                            vector = algo(data=data_training, target=target_training, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, metric=metric, n_calls=n_calls)
                        else:
                            vector = algo(data=data_training, target=target_training, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, estimator=estimator, metric=metric, n_calls=n_calls)
                            n_features = "-"
                        score = get_score(
                            data_training, target_training, vector, estimator, metric)
                        df_results.loc[len(df_results)] = [
                            algo_descr, learn_descr, kernel_descr, discr_descr, n_features, vector, score]
                        queue.put(1)  # increase progress bar
                else:
                    if discr == "n_highest":
                        # TODO: with more than one n_features
                        n_features = round(nr_of_features/2)
                        vector = algo(data=data_training, target=target_training, learning_method=learn,
                                      discretization_method=discr, n_features=n_features, n_calls=n_calls)
                    else:
                        vector = algo(
                            data=data_training, target=target_training, learning_method=learn, discretization_method=discr, estimator=estimator, metric=metric, n_calls=n_calls)
                        n_features = "-"
                    score = get_score(data_training, target_training, vector,
                                      estimator, metric)
                    df_results.loc[len(df_results)] = [
                        algo_descr, learn_descr, "-", discr_descr, n_features, vector, score]
                    queue.put(1)  # increase progress bar
    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_test, target_test, df_results, estimator, metric)

    return df_results_with_test_scores


def __run_all_comparison(data_training, data_test, target_training, target_test, estimator, metric, queue):
    """ Run all comparison approaches with all possible algorithms and parameters.

    Keyword arguments:
    data_training -- feature matrix of training data
    data_test -- feature matrix of test data
    target_training -- target vector of training data
    target_test -- target vector of test data
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score
    queue -- queue to synchronize progress bar

    """
    nr_of_features = len(data_training.columns)
    # Define result dataframe
    df_results = pd.DataFrame(
        columns=comparison_parameters+["Vector", "Training Score"])
    for approach, approach_descr in comparison_approaches.items():
        for algo, algo_descr in approach_descr.items():
            if algo == vt:
                vector = algo(data=data_training, target=target_training)
                score = get_score(data_training, target_training, vector,
                                  estimator, metric)
                df_results.loc[len(df_results)] = [
                    approach, algo_descr, "-", vector, score]
            else:
                # TODO: choose only specific n_features
                for n_features in range(5, nr_of_features+1, 100):
                    vector = algo(data=data_training, target=target_training,
                                  n_features=n_features, estimator=estimator)
                    score = get_score(data_training, target_training, vector,
                                      estimator, metric)
                    df_results.loc[len(df_results)] = [
                        approach, algo_descr, n_features, vector, score]
            queue.put(1)  # increase progress bar

    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_test, target_test, df_results, estimator, metric)

    return df_results_with_test_scores


def __progressbar_listener(q):
    """ Queue listener to increase progressbar from threads

    Keyword arguments:
    q -- queue

    """
    pbar = init_progress_bar()
    for amount in iter(q.get, None):
        pbar.update(amount)


def __run_all_bayesian_comparison(openml_data_id, estimator, metric, n_calls, queue):
    """ Run both bayesian and comparison approaches with all possible parameter combinations (only dataset, estimator and metric are fixed)

    Keyword arguments:
    openml_data_id -- dataset id from openml.org
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score
    n_calls -- number of iterations in bayesian optimization
    queue -- queue to synchronize progress bar

    """
    # Import dataset
    data, target = fetch_openml(
        data_id=openml_data_id, return_X_y=True, as_frame=True)

    # Initialize result dataframe
    df_bay_opt = pd.DataFrame(columns=bay_opt_parameters +
                              ["Vector", "Training Score", "Testing Score"])
    df_comparison = pd.DataFrame(
        columns=comparison_parameters+["Vector", "Training Score", "Testing Score"])

    # Split dataset into testing and training data then run all approaches in parallel
    kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)
    pool = mp.Pool(processes=config.n_processes)

    result_comparison = [pool.apply_async(__run_all_comparison, args=(data.loc[train_index], data.loc[test_index],
                                                                      target[train_index], target[test_index], estimator, metric, queue)) for train_index, test_index in kf]
    result_bayesian = [pool.apply_async(__run_all_bayesian, args=(data.loc[train_index], data.loc[test_index],
                                                                  target[train_index], target[test_index], estimator, metric, n_calls, queue)) for train_index, test_index in kf]
    df_comparison = [r.get() for r in result_comparison]
    df_bayesian = [r.get() for r in result_bayesian]

    # Concat bayesian and comparison result-arrays to combined dataframes
    df_comparison = pd.concat(df_comparison, ignore_index=True)
    df_bayesian = pd.concat(df_bayesian, ignore_index=True)

    # add column with number of selected features
    df_bay_opt["actual_features"] = df_bay_opt.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_comparison["actual_features"] = df_comparison.apply(
        lambda row: sum(row["Vector"]), axis=1)

    # Group results of cross-validation runs and determine min, max and mean of score-vaules and selected features
    df_bay_opt_grouped = df_bay_opt.groupby(bay_opt_parameters, as_index=False).agg(
        {"actual_features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_comparison_grouped = df_comparison.groupby(comparison_parameters, as_index=False).agg(
        {"actual_features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})

    return df_bay_opt_grouped, df_comparison_grouped


def experiment_all_datasets_and_estimators():
    """ Runs an experiment involving all bayesian/comparison approaches with all possible parameters, datasets, estimators and metrics.

    """
    # Initialize queue to syncronize progress bar
    queue = mp.Manager().Queue()
    proc = mp.Process(target=__progressbar_listener, args=(queue,))
    proc.start()

    # run all datasets
    for task, dataset in config.data_ids.items():
        for dataset_id, flag in dataset.items():
            if flag == True:
                if task == "classification":
                    for estimator, metrics in classification_estimators.items():
                        for metric, _ in metrics.items():
                            bayesian, comparison = __run_all_bayesian_comparison(
                                dataset_id, estimator, metric, config.n_calls, queue)
                            # Write grouped results to csv-file
                            bayesian.to_csv("results/classification/bay_opt_" +
                                            str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                            comparison.to_csv(
                                "results/classification/comparison_"+str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                elif task == "regression":
                    for estimator, metrics in regression_estimators.items():
                        for metric, _ in metrics.items():
                            bayesian, comparison = __run_all_bayesian_comparison(
                                dataset_id, estimator, metric, config.n_calls, queue)
                            # Write grouped results to csv-file
                            bayesian.to_csv("results/regression/bay_opt_"+str(dataset_id) +
                                            "_"+estimator+"_"+metric+".csv", index=False)
                            comparison.to_csv("results/regression/comparison_" +
                                              str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
    queue.put(None)
    proc.join()




def experiment_bayesian_iter_performance():
    """ Runs bayesian optimization to compare the performance depending on the iteration steps.

    """
    # Settings
    dataset_id = 1485
    learning_method = "GP"
    discretization_method = "round"
    estimator = "linear_regression"
    metric = "r2"
    kernel = "MATERN"

    # Import dataset
    data, target = fetch_openml(data_id=dataset_id, return_X_y=True, as_frame=True)

    # use kfold cross validation
    kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)

    # create multiprocess pool and apply task in parallel
    pool = mp.Pool(processes=config.n_processes)
    mp_results = []
    
    for train_index, test_index in kf:
        for n_calls in range(5, 106, 10):
            mp_results.append((n_calls, train_index, test_index, pool.apply_async(skopt, args=(data.loc[train_index], target.loc[train_index], None, kernel, learning_method, discretization_method, estimator, metric, "PI", n_calls))))
    
    results = [(r[0], r[1], r[2], r[3].get()) for r in tqdm(mp_results)]
    pool.close()
    pool.join()

    # store in pandas dataframe
    df_result = pd.DataFrame(([n_calls, get_score(data.loc[train_index], target.loc[train_index], vector, estimator, metric), get_score(data.loc[test_index], target.loc[test_index], vector, estimator, metric)] for n_calls, train_index, test_index, vector in results), columns=["Iteration Steps", "Training Score", "Testing Score"])
    df_result_grouped = df_result.groupby(["Iteration Steps"], as_index=False).agg(
        {"Training Score": ["mean"], "Testing Score": ["mean"]})

    #print(df_result)
    print(df_result_grouped)
    df_result_grouped.to_csv("results/bay_opt_iterations_" +
                                              str(dataset_id)+"_"+estimator+"_"+metric+ "_" + learning_method + "_" + kernel + "_" + discretization_method + ".csv", index=False)

experiment_bayesian_iter_performance()
# run_all_datasets_and_estimators()

"""def debug():
    data, target = fetch_openml(
        data_id=1485, return_X_y=True, as_frame=True)
    vector = skopt(data, target, n_calls=100)
    print(vector)
    print(sum(vector))
    print(get_score(data, target, vector))"""


#debug()
# TODO Question: Why does RBF kernel work with binary search space?
