import multiprocessing as mp
from sklearn.datasets import fetch_openml
from comparison_algorithms import sfs, rfe, sfm
from bayesian_algorithms import skopt
from sklearn.model_selection import KFold
import pandas as pd
from utils import get_score, add_testing_score
from tqdm import tqdm

##################
# Settings
##################
# number of processes for parallelization
n_processes = 32
# number of splits for cross-validation
n_splits = 5
# number of iterations in bayesian optimization
n_calls = 50
# openml.org dataset id (30 features: 1510, 10000 features: 1458, 500 features: 1485); # IMPORTANT: classification datasets must have numeric target classes only
data_ids = {
    "classification": {
        1510: False, # 30 features
        1485: False, # 500 features
        1458: False, # 10000 features
        1079: False # 22278 features
    },
    "regression": {
        1510: True, # 30 features
        1485: False, # 500 features
        1458: False, # 10000 features
        1079: False # 22278 features
    }
}


# Estimator and metric properties (choosing estimator cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
classification_estimators = {
    # very bad performance for many features!
    "svc_linear": {
        "accuracy": "SVC - Accuracy Score"
    }
}
regression_estimators = {
    "linear_regression": {
        "r2": "Linear Regression - Coefficient of Determination"
    },
    "svr_linear": {
        "r2": "Linear Regression - Coefficient of Determination"
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
    "wrapper": {
        sfs: "Sequential Feature Selection",
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm: "Select From Model"
    }
}


def init_progress_bar():
    """ Initialize progress bar (one step for each execution of a comparison or bayesian approach).

    """
    # calculate number of active dataset-estimator combinations
    number_datasets_classification = 0
    number_datasets_regression = 0
    for type, iter in data_ids.items():
        if type == "classification":
            for _, flag in iter.items():
                if flag == True:
                    number_datasets_classification += 1
        else:
            for _, flag in iter.items():
                if flag == True:
                    number_datasets_regression += 1
    number_datasets_and_estimators = ((number_datasets_classification * len(classification_estimators)
                                       ) + (number_datasets_regression * len(regression_estimators))) * n_splits

    # calculate number of bayesian approaches
    number_of_bayesian = (len(bayesian_approaches) * len(discretization_methods)) * (
        (len(learning_methods)-1) + len(kernels))  # only gaussian processes use kernels

    # calculate number of comparison approaches
    number_of_comparison = 0
    for _, approach in comparison_approaches.items():
        for _, _ in approach.items():
            number_of_comparison += 1
    print(number_of_comparison)

    # calculate total progress bar steps
    progress_total = number_datasets_and_estimators * \
        (number_of_bayesian + number_of_comparison)
    pbar = tqdm(total=progress_total)
    pbar.set_description("Processed")

    return pbar


def __run_all_bayesian(data, target, estimator, metric, queue):
    """ Run all bayesian optimization approaches with all possible parameters.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets

    """
    nr_of_features = len(data.columns)
    # Define result dataframes
    dataframe = pd.DataFrame(
        columns=bay_opt_parameters+["Vector", "Training Score"])
    for algo, algo_descr in bayesian_approaches.items():
        for learn, learn_descr in learning_methods.items():
            for discr, discr_descr in discretization_methods.items():
                if learn == "GP":
                    for kernel, kernel_descr in kernels.items():
                        if discr == "n_highest":
                            # TODO: with more than one n_features
                            n_features = round(nr_of_features/2)
                            vector = algo(data=data, target=target, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, metric=metric)
                        else:
                            vector = algo(data=data, target=target, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, estimator=estimator, metric=metric)
                            n_features = "-"
                        score = get_score(
                            data, target, vector, estimator, metric)
                        dataframe.loc[len(dataframe)] = [
                            algo_descr, learn_descr, kernel_descr, discr_descr, n_features, vector, score]
                        queue.put(1)  # increase progress bar
                else:
                    if discr == "n_highest":
                        # TODO: with more than one n_features
                        n_features = round(nr_of_features/2)
                        vector = algo(data=data, target=target, learning_method=learn,
                                      discretization_method=discr, n_features=n_features)
                    else:
                        vector = algo(
                            data=data, target=target, learning_method=learn, discretization_method=discr, estimator=estimator, metric=metric)
                        n_features = "-"
                    score = get_score(data, target, vector,
                                      estimator, metric)
                    dataframe.loc[len(dataframe)] = [
                        algo_descr, learn_descr, "-", discr_descr, n_features, vector, score]
                    queue.put(1)  # increase progress bar
    return dataframe


def __run_all_comparison(data, target, estimator, metric, queue):
    """ Run all comparison approaches with all possible algorithms and parameters.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets

    """
    nr_of_features = len(data.columns)
    # Define result dataframe
    dataframe = pd.DataFrame(
        columns=comparison_parameters+["Vector", "Training Score"])
    for approach, approach_descr in comparison_approaches.items():
        for algo, algo_descr in approach_descr.items():
            # TODO: choose only specific n_features
            for n_features in range(5, nr_of_features+1, 100):
                vector = algo(data=data, target=target,
                              n_features=n_features, estimator=estimator)
                score = get_score(data, target, vector,
                                  estimator, metric)
                dataframe.loc[len(dataframe)] = [
                    approach, algo_descr, n_features, vector, score]
                queue.put(1)  # increase progress bar
    return dataframe


def __run_training_testing(data, target, train_index, test_index, estimator, metric, queue):
    """ Run bayesian and comparison approaches on training set, then add score of test set. Return results as dataframe.

    Keyword arguments:
    train_index -- indices of training set
    test_index -- indices of testing set

    """
    X_train, X_test = data.loc[train_index], data.loc[test_index]
    y_train, y_test = target[train_index], target[test_index]
    #
    # run all bayesian approaches
    #
    df_current_bayesian = __run_all_bayesian(
        X_train, y_train, estimator, metric, queue)
    df_current_bayesian = add_testing_score(
        X_test, y_test, df_current_bayesian, estimator, metric)
    #
    # run all comparison approaches
    #
    df_current_comparison = __run_all_comparison(
        X_train, y_train, estimator, metric, queue)
    df_current_comparison = add_testing_score(
        X_test, y_test, df_current_comparison, estimator, metric)

    return df_current_bayesian, df_current_comparison


def progressbar_listener(q):
    """ Queue listener to increase progressbar from threads

    Keyword arguments:
    q -- queue

    """
    pbar = init_progress_bar()
    for amount in iter(q.get, None):
        pbar.update(amount)


def __run_experiment(openml_data_id, estimator, metric, queue):
    # Import dataset
    data, target = fetch_openml(
        data_id=openml_data_id, return_X_y=True, as_frame=True)

    # Initialize result dataframe
    df_bay_opt = pd.DataFrame(columns=bay_opt_parameters +
                              ["Vector", "Training Score", "Testing Score"])
    df_comparison = pd.DataFrame(
        columns=comparison_parameters+["Vector", "Training Score", "Testing Score"])

    # Split dataset into testing and training data then run all approaches in parallel
    kf = KFold(n_splits=n_splits, shuffle=True)
    pool = mp.Pool(processes=n_processes)
    mp_result = [pool.apply_async(__run_training_testing, args=(data, target,
                                                                train_index, test_index, estimator, metric, queue)) for train_index, test_index in kf.split(data)]
    df_result = [p.get() for p in mp_result]

    # Concat bayesian and comparison results to separate dataframes
    df_bay_opt = pd.concat([x[0] for x in df_result], ignore_index=True)
    df_comparison = pd.concat([x[1] for x in df_result], ignore_index=True)

    # add column with number of selected features
    df_bay_opt["actual_features"] = df_bay_opt.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_comparison["actual_features"] = df_comparison.apply(
        lambda row: sum(row["Vector"]), axis=1)

    # Group results of cross-validation runs and determine min, max and mean of score-vaules and selected features
    df_bay_opt_grouped = df_bay_opt.groupby(bay_opt_parameters, as_index=False).agg(
        {"actual_features": ["mean", "min", "max"], "Training Score": ["mean", "min", "max"], "Testing Score": ["mean", "min", "max"]})
    df_comparison_grouped = df_comparison.groupby(comparison_parameters, as_index=False).agg(
        {"actual_features": ["mean", "min", "max"], "Training Score": ["mean", "min", "max"], "Testing Score": ["mean", "min", "max"]})

    return df_bay_opt_grouped, df_comparison_grouped


def main():
    # Initialize queue to syncronize progress bar
    queue = mp.Manager().Queue()
    proc = mp.Process(target=progressbar_listener, args=(queue,))
    proc.start()

    # run all datasets
    for task, dataset in data_ids.items():
        for dataset_id, flag in dataset.items():
            if flag == True:
                if task == "classification":
                    for estimator, metrics in classification_estimators.items():
                        for metric, _ in metrics.items():
                            bayesian, comparison = __run_experiment(
                                dataset_id, estimator, metric, queue)
                            # Write grouped results to csv-file
                            bayesian.to_csv("results/bay_opt_" +
                                            str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                            comparison.to_csv(
                                "results/comparison_"+str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                elif task == "regression":
                    for estimator, metrics in regression_estimators.items():
                        for metric, _ in metrics.items():
                            bayesian, comparison = __run_experiment(
                                dataset_id, estimator, metric, queue)
                            # Write grouped results to csv-file
                            bayesian.to_csv("results/bay_opt_"+str(dataset_id) +
                                            "_"+estimator+"_"+metric+".csv", index=False)
                            comparison.to_csv("results/comparison_" +
                                              str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
    queue.put(None)
    proc.join()


main()

# TODO Question: Why does RBF kernel work with binary search space?
