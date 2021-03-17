from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import pandas as pd
import time

from comparison_algorithms import rfe, sfs, sfm
from utils import get_score, add_testing_score
import approaches
import config

def init_progress_bar():
    """ Initialize progress bar (calculate number of steps).

    Return: progressbar object

    """
    # calculate number of datasets
    number_datasets_classification = 0
    for type, iter in config.data_ids.items():
        if type == "classification":
            for _, flag in iter.items():
                if flag == True:
                    number_datasets_classification += 1
        else:
            raise ValueError(
                "Only classification datasets are supported currently!")

    # calculate number of estimators/metrics
    number_classification_estimators = 0
    for _, iter in approaches.classification_estimators.items():
        for _, _ in iter.items():
            number_classification_estimators += 1

    # calculate number of dataset/estimator combinations
    number_datasets_and_estimators = (number_datasets_classification * number_classification_estimators) * config.n_splits

    # calculate number of bayesian approaches
    #number_of_bayesian = (len(approaches.discretization_methods) * len(approaches.acquisition_functions)) * (
    #    (len(approaches.learning_methods)-1) + len(approaches.kernels))  # only gaussian processes use kernels
    # 1. integer, n_highest, probabilistic_round and round together with GP matern and RBF kernels
    # 2. categorical with GP Hamming kernel and random forest
    number_of_bayesian = len(approaches.acquisition_functions) * ((4 * 2) + (2))

    # calculate number of comparison approaches
    number_of_comparison = 0
    for _, approach in approaches.comparison_approaches.items():
        for _, _ in approach.items():
            number_of_comparison += 1

    # calculate number of runs with different number of features
    feature_runs = (config.max_nr_features-config.min_nr_features) / \
        config.iter_step_nr_features + 1

    # calculate total progress bar steps
    progress_total = feature_runs * number_datasets_and_estimators * \
        (number_of_bayesian + number_of_comparison + 1) # also consider training without feature selection (+1)
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
    n_calls -- number of iterations in bayesian optimization
    queue -- queue to synchronize progress bar

    Return: Dataframe of all possible bayesian results (including testing scores)

    """
    # Define result dataframes
    df_results = pd.DataFrame(
        columns=approaches.bay_opt_parameters+["Duration Black Box", "Duration Overhead", "Number of Iterations", "Vector", "Training Score"])
    for algo, algo_descr in approaches.bayesian_approaches.items():
        for learn, learn_descr in approaches.learning_methods.items():
            for discr, discr_descr in approaches.discretization_methods.items():
                if discr == "binary" or discr == "categorical":
                    # always sample points with excact n_features selected features
                    acq_optimizer = "n_sampling"
                else:
                    acq_optimizer = "sampling"
                for acq, _ in approaches.acquisition_functions.items():
                    if learn == "GP":
                        for kernel, kernel_descr in approaches.kernels.items():
                            # kernels only apply for gaussian processes
                            if kernel == "HAMMING":
                                # hamming kernel only for categorical and binary search-spaces
                                if discr == "categorical":
                                    for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                        start_time = time.time()
                                        vector, score, nr_iters, black_box_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                     kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                        duration = time.time() - start_time
                                        duration_overhead = duration - black_box_duration
                                        df_results.loc[len(df_results)] = [
                                            algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, black_box_duration, duration_overhead, nr_iters, vector, score]
                                        queue.put(1)  # increase progress bar
                            else:
                                # Matern and RBF kernels for all discretization methods except categorical
                                if discr != "categorical":
                                    for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                        start_time = time.time()
                                        vector, score, nr_iters, black_box_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                     kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                        duration = time.time() - start_time
                                        duration_overhead = duration - black_box_duration
                                        df_results.loc[len(df_results)] = [
                                            algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, black_box_duration, duration_overhead, nr_iters, vector, score]
                                        queue.put(1)  # increase progress bar
                    else:
                        # random forests only in categorical search space
                        if discr == "categorical":
                            for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                start_time = time.time()
                                vector, score, nr_iters, black_box_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                    discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_features=n_features, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                duration = time.time() - start_time
                                duration_overhead = duration - black_box_duration
                                df_results.loc[len(df_results)] = [
                                    algo_descr, learn_descr, "-", discr_descr, acq, n_features, black_box_duration, duration_overhead, nr_iters, vector, score]
                                queue.put(1)  # increase progress bar
            
    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_training, data_test, target_training, target_test, df_results, estimator, metric)

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

    Return: Dataframe of all possible comparison results (including testing scores)

    """
    # Define result dataframe
    df_results = pd.DataFrame(
        columns=approaches.comparison_parameters+["Duration", "Vector", "Training Score"])
    for approach, approach_descr in approaches.comparison_approaches.items():
        for algo, algo_descr in approach_descr.items():
            if (algo is rfe or algo is sfm or algo is sfs) and estimator == "k_neighbours_classifier":
                # k nearest neighbors does not support weights (needed for some wrapper and embedded approaches)
                queue.put(1)  # increase progress bar
            else:
                for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                    start_time = time.time()
                    vector = algo(data=data_training, target=target_training,
                                  n_features=n_features, estimator=estimator)
                    duration = time.time() - start_time
                    score = get_score(data_training, data_training, target_training, target_training, vector,
                                      estimator, metric)
                    df_results.loc[len(df_results)] = [
                        approach, algo_descr, n_features, duration, vector, score]
                    queue.put(1)  # increase progress bar

    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_training, data_test, target_training, target_test, df_results, estimator, metric)

    return df_results_with_test_scores

def __run_without_fs(data_training, data_test, target_training, target_test, estimator, metric, queue):
    """ Train model withoud feature selection.

    Keyword arguments:
    data_training -- feature matrix of training data
    data_test -- feature matrix of test data
    target_training -- target vector of training data
    target_test -- target vector of test data
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score
    queue -- queue to synchronize progress bar

    Return: Dataframe of the result scores without any feature selection

    """
    df_results = pd.DataFrame(columns=["Training Score", "Testing Score"])
    vector = [1 for _ in range(0,len(data_training.columns))]

    training_score = get_score(data_training, data_training, target_training, target_training, vector,
                                      estimator, metric)
    testing_score = get_score(data_training, data_test, target_training, target_test, vector,
                                      estimator, metric)
    df_results.loc[len(df_results)] = [training_score, testing_score]

    queue.put(1)  # increase progress bar

    return df_results


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

    Return: tuple of bayesian results, comparison results and results without feature selection

    """
    # Import dataset
    data, target = fetch_openml(
        data_id=openml_data_id, return_X_y=True, as_frame=True)

    # Initialize result dataframe
    df_bay_opt = pd.DataFrame(columns=approaches.bay_opt_parameters +
                              ["Duration Black Box", "Duration Overhead" "Number of Iterations", "Vector", "Training Score", "Testing Score"])
    df_comparison = pd.DataFrame(
        columns=approaches.comparison_parameters+["Duration", "Vector", "Training Score", "Testing Score"])
    df_without_fs = pd.DataFrame(columns=["Training Score", "Testing Score"])

    # Split dataset into testing and training data then run all approaches in parallel
    kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)
    pool = mp.Pool(processes=config.n_processes)
    mp_results = []
    for train_index, test_index in kf:
        mp_results.append(("comparison", pool.apply_async(__run_all_comparison, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index], "target_training":target[train_index], "target_test":target[test_index], "estimator":estimator, "metric":metric, "queue":queue})))
        mp_results.append(("bayesian", pool.apply_async(__run_all_bayesian, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index],
                                                                                  "target_training":target[train_index], "target_test":target[test_index], "estimator":estimator, "metric":metric, "n_calls":n_calls, "queue":queue})))
        mp_results.append(("without", pool.apply_async(__run_without_fs, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index],
                                                                                      "target_training":target[train_index], "target_test":target[test_index], "estimator":estimator, "metric":metric, "queue":queue})))

    # receive results from multiprocessing
    results_comparison = []
    results_bay_opt = []
    results_without_fs = []
    for approach, r in mp_results:
        if approach == "comparison":
            results_comparison.append(r.get())
        elif approach == "bayesian":
            results_bay_opt.append(r.get())
        else:
            results_without_fs.append(r.get())
    pool.close()
    pool.join()

    # Concat bayesian and comparison result-arrays to combined dataframes
    df_comparison = pd.concat(results_comparison, ignore_index=True)
    df_bay_opt = pd.concat(results_bay_opt, ignore_index=True)
    df_without_fs = pd.concat(results_without_fs, ignore_index=True)

    # add column with number of selected features
    df_bay_opt["Actual Features"] = df_bay_opt.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_comparison["Actual Features"] = df_comparison.apply(
        lambda row: sum(row["Vector"]), axis=1)

    # convert int to np.int64 to be able to aggrogate
    df_bay_opt["Number of Iterations"] = df_bay_opt.apply(
        lambda row: np.int64(row["Number of Iterations"]), axis=1)

    # Group results of cross-validation runs and determine min, max and mean of score-vaules and selected features
    df_bay_opt_grouped = df_bay_opt.groupby(approaches.bay_opt_parameters, as_index=False).agg(
        {"Duration Black Box": ["mean"], "Duration Overhead": ["mean"], "Number of Iterations": ["mean"], "Actual Features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_comparison_grouped = df_comparison.groupby(approaches.comparison_parameters, as_index=False).agg(
        {"Duration": ["mean"], "Actual Features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_without_fs = df_without_fs.agg(
        {"Training Score": ["mean"], "Testing Score": ["mean"]})

    return df_bay_opt_grouped, df_comparison_grouped, df_without_fs


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
                    estimators = approaches.classification_estimators
                else:
                    raise ValueError("Only classification estimators are supported currently!")

                for estimator, metrics in estimators.items():
                    for metric, _ in metrics.items():
                        bayesian, comparison, without_fs = __run_all_bayesian_comparison(
                            dataset_id, estimator, metric, config.n_calls, queue)
                        # Write grouped results to csv-file
                        bayesian.to_csv("results/comparison_bayesian_experiment/" + task + "/bay_opt_" +
                                        str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                        comparison.to_csv(
                            "results/comparison_bayesian_experiment/" + task + "/comparison_"+str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                        without_fs.to_csv("results/comparison_bayesian_experiment/" + task + "/withoutfs_"+str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)

    queue.put(None)
    proc.join()

if __name__ == "__main__":
    experiment_all_datasets_and_estimators()
