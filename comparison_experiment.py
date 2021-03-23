from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import pandas as pd
import time

from comparison_algorithms import rfe, sfs, sfm_svc, sfm_logistic_regression, sfm_random_forest
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
                                        vector, score, nr_iters, black_box_duration, overhead_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                     kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                        df_results.loc[len(df_results)] = [
                                            algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, black_box_duration, overhead_duration, nr_iters, vector, score]
                                        queue.put(1)  # increase progress bar
                            else:
                                # Matern and RBF kernels for all discretization methods except categorical
                                if discr != "categorical":
                                    for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                        vector, score, nr_iters, black_box_duration, overhead_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                     kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                        df_results.loc[len(df_results)] = [
                                            algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, black_box_duration, overhead_duration, nr_iters, vector, score]
                                        queue.put(1)  # increase progress bar
                    else:
                        # random forests only in categorical search space
                        if discr == "categorical":
                            for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                vector, score, nr_iters, black_box_duration, overhead_duration = algo(data=data_training, target=target_training, learning_method=learn,
                                                    discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_features=n_features, n_calls=n_calls, cross_validation=config.n_splits_bay_opt, acq_optimizer=acq_optimizer, n_convergence=config.n_convergence, n_acq_points=config.n_acq_points)
                                df_results.loc[len(df_results)] = [
                                    algo_descr, learn_descr, "-", discr_descr, acq, n_features, black_box_duration, overhead_duration, nr_iters, vector, score]
                                queue.put(1)  # increase progress bar
            
    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_training, data_test, target_training, target_test, df_results, estimator, metric)

    # add column with number of selected features
    df_results_with_test_scores["Actual Features"] = df_results_with_test_scores.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_results_with_test_scores["Actual Features"] = df_results_with_test_scores.apply(
        lambda row: sum(row["Vector"]), axis=1)

    # add estimator and metric
    df_results_with_test_scores["Estimator"] = estimator
    df_results_with_test_scores["Metric"] = metric

    # convert int to np.int64 to be able to aggrogate
    df_results_with_test_scores["Number of Iterations"] = df_results_with_test_scores.apply(lambda row: np.int64(row["Number of Iterations"]), axis=1)

    return df_results_with_test_scores


def __run_all_comparison(data_training, data_test, target_training, target_test, estimator, metric):
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
            for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                start_time = time.time()
                vector = algo(data=data_training, target=target_training,
                              n_features=n_features, estimator=estimator)
                duration = time.time() - start_time
                score = get_score(data_training, data_training, target_training, target_training, vector,
                                  estimator, metric)
                df_results.loc[len(df_results)] = [
                    approach, algo_descr, n_features, duration, vector, score]

    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_training, data_test, target_training, target_test, df_results, estimator, metric)

    # add estimator and metric
    df_results_with_test_scores["Estimator"] = estimator
    df_results_with_test_scores["Metric"] = metric

    return df_results_with_test_scores

def __run_without_fs(data_training, data_test, target_training, target_test, estimator, metric):
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

    df_results["Estimator"] = estimator
    df_results["Metric"] = metric

    return df_results


def __run_bay_and_add_test_score(algo, algo_descr, learn_descr, kernel_descr, discr_descr, acq, data_training, data_test, target_training, target_test, learning_method, kernel, discretization_method, n_features, estimator, acq_func, metric, n_calls, cross_validation, acq_optimizer, n_convergence, n_acq_points):
    """ Actual run of bayesian optimization algorithm. Adds test scores and combines all into a dataframe.

    Keyword arguments:
    algo -- bayesian optimization algorithm
    algo_descr, learn_descr, kernel_descr, discr_descr, acq -- descriptions of algorithm
    data_training -- feature matrix of training data
    data_test -- feature matrix of test data
    target_training -- target vector of training data
    target_test -- target vector of test data
    learning_method -- model used for Bayesian optimization (GP or RF)
    kernel -- kernel type for Gaussian processes
    discretization_method -- define method on how to work with search space
    n_features -- number of features to be selected; if 'None' then best result is selected
    estimator -- estimator used to determine score
    acq_func -- aquisition function to be used
    metric -- calculate score based on metric
    n_calls -- number of iterations
    cross_validation - number of folds to perform cross validation inside bayesian optimization on
    acq_optimizer -- strategy to sample points of aqcuisition function  
    n_convergence -- stop optimization if for n_convergence iterations the optimum did not change
    n_acq_points -- number of points of the acquisition function to evaluate in each iteration

    Return: dataframe with results
    """
    # run algorithm
    vector, score, nr_iters, black_box_duration, overhead_duration = algo(data=data_training, target=target_training, learning_method=learning_method, kernel=kernel, discretization_method=discretization_method, n_features=n_features, estimator=estimator, acq_func=acq_func, metric=metric, n_calls=n_calls, cross_validation=cross_validation, acq_optimizer=acq_optimizer, n_convergence=n_convergence, n_acq_points=n_acq_points)
    # create dataframe
    df_results = pd.DataFrame(
        columns=approaches.bay_opt_parameters+["Duration Black Box", "Duration Overhead", "Number of Iterations", "Vector", "Training Score"])
    df_results.loc[len(df_results)] = [algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, black_box_duration, overhead_duration, nr_iters, vector, score]

    df_results_with_test_scores = add_testing_score(data_training, data_test, target_training, target_test, df_results, estimator, metric)

    # add estimator and metric
    df_results_with_test_scores["Estimator"] = estimator
    df_results_with_test_scores["Metric"] = metric

    # convert int to np.int64 to be able to aggrogate
    df_results_with_test_scores["Number of Iterations"] = df_results_with_test_scores.apply(lambda row: np.int64(row["Number of Iterations"]), axis=1)

    return df_results_with_test_scores
    

def __run_all_bayesian_mp(data_training, data_test, target_training, target_test, estimator, metric, n_calls, mp_results, pool, dataset_id):
    """ Run all Bayesian optimization approaches and add tasks to multiprocessing pool.

    Keyword arguments:
    data_training -- feature matrix of training data
    data_test -- feature matrix of test data
    target_training -- target vector of training data
    target_test -- target vector of test data
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score
    n_calls -- maximum number of iterations of bayesian optimization
    mp_results -- result array
    pool -- multiprocessing pool
    dataset_id -- id of dataset (to identify results)

    """
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
                                        mp_results.append(("bayopt", dataset_id, (pool.apply_async(__run_bay_and_add_test_score, [], {"algo": algo, "algo_descr":algo_descr, "learn_descr":learn_descr, "kernel_descr":kernel_descr, "discr_descr":discr_descr, "acq":acq, "data_training":data_training, "data_test": data_test, "target_training":target_training, "target_test":target_test, "learning_method":learn, "kernel":kernel, "discretization_method":discr, "n_features":n_features, "estimator":estimator, "acq_func":acq, "metric":metric, "n_calls":n_calls, "cross_validation":config.n_splits_bay_opt, "acq_optimizer":acq_optimizer, "n_convergence":config.n_convergence, "n_acq_points":config.n_acq_points}))))
                            else:
                                # Matern and RBF kernels for all discretization methods except categorical
                                if discr != "categorical":
                                    for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                         mp_results.append(("bayopt", dataset_id, (pool.apply_async(__run_bay_and_add_test_score, [], {"algo": algo, "algo_descr":algo_descr, "learn_descr":learn_descr, "kernel_descr":kernel_descr, "discr_descr":discr_descr, "acq":acq, "data_training":data_training, "data_test": data_test, "target_training":target_training, "target_test":target_test, "learning_method":learn, "kernel":kernel, "discretization_method":discr, "n_features":n_features, "estimator":estimator, "acq_func":acq, "metric":metric, "n_calls":n_calls, "cross_validation":config.n_splits_bay_opt, "acq_optimizer":acq_optimizer, "n_convergence":config.n_convergence, "n_acq_points":config.n_acq_points}))))
                    else:
                        # random forests only in categorical search space
                        if discr == "categorical":
                            for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                 mp_results.append(("bayopt", dataset_id, (pool.apply_async(__run_bay_and_add_test_score, [], {"algo": algo, "algo_descr":algo_descr, "learn_descr":learn_descr, "kernel_descr":"-", "discr_descr":discr_descr, "acq":acq, "data_training":data_training, "data_test": data_test, "target_training":target_training, "target_test":target_test, "learning_method":learn, "kernel":None, "discretization_method":discr, "n_features":n_features, "estimator":estimator, "acq_func":acq, "metric":metric, "n_calls":n_calls, "cross_validation":config.n_splits_bay_opt, "acq_optimizer":acq_optimizer, "n_convergence":config.n_convergence, "n_acq_points":config.n_acq_points}))))


def experiment_all_datasets_and_estimators():
    """ Runs an experiment involving all bayesian/comparison approaches with all possible parameters, datasets, estimators and metrics.

    """
    pool = mp.Pool(processes=config.n_processes)
    mp_results = []

    # run all datasets
    for task, dataset in config.data_ids.items():
        for dataset_id, flag in dataset.items():
            if flag == True:
                # Import dataset
                data, target = fetch_openml(
                data_id=dataset_id, return_X_y=True, as_frame=True)

                # run cross-validation
                kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)

                for train_index, test_index in kf:
                    for estimator, metrics in approaches.classification_estimators.items():
                        for metric, _ in metrics.items():
                            # run approaches
                            __run_all_bayesian_mp(data_training=data.loc[train_index], data_test=data.loc[test_index], target_training=target.loc[train_index], target_test=target.loc[test_index], estimator=estimator, metric=metric, n_calls=config.n_calls, mp_results=mp_results, pool=pool, dataset_id=dataset_id)
                            #mp_results.append(("bayopt", dataset_id, pool.apply_async(__run_all_bayesian, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index], "target_training":target.loc[train_index], "target_test":target.loc[test_index], "estimator":estimator, "metric":metric, "n_calls":config.n_calls, "queue":queue})))
                            mp_results.append(("comparison", dataset_id, pool.apply_async(__run_all_comparison, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index], "target_training":target.loc[train_index], "target_test":target.loc[test_index], "estimator":estimator, "metric":metric})))
                            mp_results.append(("withoutfs", dataset_id, pool.apply_async(__run_without_fs, [], {"data_training":data.loc[train_index], "data_test":data.loc[test_index], "target_training":target.loc[train_index], "target_test":target.loc[test_index], "estimator":estimator, "metric":metric})))

    # get results
    results = [(r[0],r[1]) + (r[2].get(),) for r in tqdm(mp_results)]

    # separate bay opt, comparison and without fs results
    res_bay_opt = []
    res_comparison = []
    res_without_fs = []
    for approach, dataset_id, res in results:
        res["did"] = dataset_id # add dataset id to dataframe
        if approach == "bayopt":
            res_bay_opt.append(res)
        elif approach == "comparison":
            res_comparison.append(res)
        elif approach == "withoutfs":
            res_without_fs.append(res)

    # concat to single dataframes
    df_bay_opt = pd.concat(res_bay_opt)
    df_comparison = pd.concat(res_comparison)
    df_without_fs = pd.concat(res_without_fs)

    # add column with number of selected features
    df_bay_opt["Actual Features"] = df_bay_opt.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_comparison["Actual Features"] = df_comparison.apply(
        lambda row: sum(row["Vector"]), axis=1)
        
    df_bay_opt_grouped = df_bay_opt.groupby(["did", "Estimator", "Metric"] + approaches.bay_opt_parameters, as_index=False).agg(
        {"Duration Black Box": ["mean"], "Duration Overhead": ["mean"], "Number of Iterations": ["mean"], "Actual Features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_comparison_grouped = df_comparison.groupby(["did", "Estimator", "Metric"] + approaches.comparison_parameters, as_index=False).agg(
        {"Duration": ["mean"], "Actual Features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_without_fs_grouped = df_without_fs.groupby(["did", "Estimator", "Metric"], as_index=False).agg(
        {"Training Score": ["mean"], "Testing Score": ["mean"]})

    for name, group in df_bay_opt_grouped.groupby(["did", "Estimator", "Metric"], as_index=False):
        group.iloc[:, 3:].to_csv("results/comparison_bayesian_experiment/classification" + "/bayopt_" +
                    str(name[0])+"_"+str(name[1])+"_"+str(name[2])+".csv", index=False)
    for name, group in df_comparison_grouped.groupby(["did", "Estimator", "Metric"], as_index=False):
        group.iloc[:, 3:].to_csv("results/comparison_bayesian_experiment/classification" + "/comparison_" +
                    str(name[0])+"_"+str(name[1])+"_"+str(name[2])+".csv", index=False)
    for name, group in df_without_fs_grouped.groupby(["did", "Estimator", "Metric"], as_index=False):
        group.iloc[:, 3:].to_csv("results/comparison_bayesian_experiment/classification" + "/withoutfs_" +
                    str(name[0])+"_"+str(name[1])+"_"+str(name[2])+".csv", index=False)


    pool.close()
    pool.join()

if __name__ == "__main__":
    experiment_all_datasets_and_estimators()
