from bayesian_algorithms import skopt, discretize, gpyopt
from comparison_algorithms import rfe, sfs, sfm, n_best_anova_f, n_best_mutual, n_best_pearsonr
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import get_score, add_testing_score
import config
import multiprocessing as mp
import pandas as pd
import approaches


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
    for _, iter in approaches.regression_estimators.items():
        for _, _ in iter.items():
            number_regression_estimators += 1
    number_classification_estimators = 0
    for _, iter in approaches.classification_estimators.items():
        for _, _ in iter.items():
            number_classification_estimators += 1

    # calculate numbe rof dataset/estimator combinations
    number_datasets_and_estimators = ((number_datasets_classification * number_classification_estimators
                                       ) + (number_datasets_regression * number_regression_estimators)) * config.n_splits

    # calculate number of bayesian approaches
    number_of_bayesian = (len(approaches.bayesian_approaches) * len(approaches.discretization_methods) * len(approaches.acquisition_functions)) * (
        (len(approaches.learning_methods)-1) + len(approaches.kernels))  # only gaussian processes use kernels

    # calculate number of comparison approaches
    number_of_comparison = 0
    for _, approach in approaches.comparison_approaches.items():
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
    # Define result dataframes
    df_results = pd.DataFrame(
        columns=approaches.bay_opt_parameters+["Vector", "Training Score"])

    for learn, learn_descr in approaches.learning_methods.items():
        for discr, discr_descr in approaches.discretization_methods.items():
            for acq, _ in approaches.acquisition_functions.items():
                if learn == "GP":
                    for kernel, kernel_descr in approaches.kernels.items():
                        # use mainly gpyopt, only with hamming kernel gpyopt
                        if kernel == "HAMMING":
                            algo = skopt
                            algo_descr = "scikit-optimize"
                        else:
                            algo = gpyopt
                            algo_descr = "gpyopt"

                        vector = []
                        if discr == "n_highest" or discr == "round":
                            for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                                vector, score = algo(data=data_training, target=target_training, learning_method=learn,
                                              kernel=kernel, discretization_method=discr, n_features=n_features, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt)
                                df_results.loc[len(df_results)] = [
                                       algo_descr, learn_descr, kernel_descr, discr_descr, acq, n_features, vector, score]
                                """if discr == "round":
                                    # run "round" without a predifined number of features
                                    vector, score = algo(data=data_training, target=target_training, learning_method=learn,
                                                  kernel=kernel, discretization_method=discr, n_features=None, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls)
                                    df_results.loc[len(df_results)] = [
                                        algo_descr, learn_descr, kernel_descr, discr_descr, acq, "-", vector, score]"""
                        else:
                            vector, score = algo(data=data_training, target=target_training, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt)
                            df_results.loc[len(df_results)] = [
                                algo_descr, learn_descr, kernel_descr, discr_descr, acq, "-", vector, score]
                        queue.put(1)  # increase progress bar
                else:
                    # use only skopt for learning methods other than GP
                    algo = skopt
                    algo_descr = "scikit-optimize"

                    vector = []
                    if discr == "n_highest" or "round":
                        for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                            vector, score = algo(data=data_training, target=target_training, learning_method=learn,
                                          discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_features=n_features, n_calls=n_calls, cross_validation=config.n_splits_bay_opt)
                            df_results.loc[len(df_results)] = [
                                algo_descr, learn_descr, "-", discr_descr, acq, n_features, vector, score]
                            """if discr == "round":
                                    # run "round" without a predifined number of features
                                    vector, score = algo(data=data_training, target=target_training, learning_method=learn,
                                                  discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_features=None, n_calls=n_calls)
                                    df_results.loc[len(df_results)] = [
                                        algo_descr, learn_descr, "-", discr_descr, acq, "-", vector, score]"""
                    else:
                        vector, score = algo(
                            data=data_training, target=target_training, learning_method=learn, discretization_method=discr, estimator=estimator, acq_func=acq, metric=metric, n_calls=n_calls, cross_validation=config.n_splits_bay_opt)
                        df_results.loc[len(df_results)] = [
                            algo_descr, learn_descr, "-", discr_descr, acq, "-", vector, score]

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

    """
    # Define result dataframe
    df_results = pd.DataFrame(
        columns=approaches.comparison_parameters+["Vector", "Training Score"])
    for approach, approach_descr in approaches.comparison_approaches.items():
        for algo, algo_descr in approach_descr.items():
            if (algo == rfe or algo == sfm or algo == sfs) and estimator == "k_neighbours_classifier":
                # k nearest neighbors does not support weights (needed for some wrapper and embedded approaches)
                break
            else:
                for n_features in range(config.min_nr_features, config.max_nr_features+1, config.iter_step_nr_features):
                    vector = algo(data=data_training, target=target_training,
                                  n_features=n_features, estimator=estimator)
                    score = get_score(data_training, data_training, target_training, target_training, vector,
                                      estimator, metric)
                    df_results.loc[len(df_results)] = [
                        approach, algo_descr, n_features, vector, score]
            queue.put(1)  # increase progress bar

    # generate test scores
    df_results_with_test_scores = add_testing_score(
        data_training, data_test, target_training, target_test, df_results, estimator, metric)

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
    df_bay_opt = pd.DataFrame(columns=approaches.bay_opt_parameters +
                              ["Vector", "Training Score", "Testing Score"])
    df_comparison = pd.DataFrame(
        columns=approaches.comparison_parameters+["Vector", "Training Score", "Testing Score"])

    # Split dataset into testing and training data then run all approaches in parallel
    kf = KFold(n_splits=config.n_splits, shuffle=True).split(data)
    pool = mp.Pool(processes=config.n_processes)
    mp_results = []
    for train_index, test_index in kf:
        mp_results.append(("comparison", pool.apply_async(__run_all_comparison, args=(data.loc[train_index], data.loc[test_index],
                                                                                      target[train_index], target[test_index], estimator, metric, queue))))
        mp_results.append(("bayesian", pool.apply_async(__run_all_bayesian, args=(data.loc[train_index], data.loc[test_index],
                                                                                  target[train_index], target[test_index], estimator, metric, n_calls, queue))))

    # receive results from multiprocessing
    results_comparison = []
    results_bay_opt = []
    for approach, r in mp_results:
        if approach == "comparison":
            results_comparison.append(r.get())
        else:
            results_bay_opt.append(r.get())

    # Concat bayesian and comparison result-arrays to combined dataframes
    df_comparison = pd.concat(results_comparison, ignore_index=True)
    df_bay_opt = pd.concat(results_bay_opt, ignore_index=True)

    # add column with number of selected features
    df_bay_opt["actual_features"] = df_bay_opt.apply(
        lambda row: sum(row["Vector"]), axis=1)
    df_comparison["actual_features"] = df_comparison.apply(
        lambda row: sum(row["Vector"]), axis=1)

    # Group results of cross-validation runs and determine min, max and mean of score-vaules and selected features
    df_bay_opt_grouped = df_bay_opt.groupby(approaches.bay_opt_parameters, as_index=False).agg(
        {"actual_features": ["mean"], "Training Score": ["mean"], "Testing Score": ["mean"]})
    df_comparison_grouped = df_comparison.groupby(approaches.comparison_parameters, as_index=False).agg(
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
                    estimators = approaches.classification_estimators
                else:
                    estimators = approaches.regression_estimators

                for estimator, metrics in estimators.items():
                    for metric, _ in metrics.items():
                        bayesian, comparison = __run_all_bayesian_comparison(
                            dataset_id, estimator, metric, config.n_calls, queue)
                        # Write grouped results to csv-file
                        bayesian.to_csv("results/comparison_bayesian_experiment/" + task + "/bay_opt_" +
                                        str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)
                        comparison.to_csv(
                            "results/comparison_bayesian_experiment/" + task + "/comparison_"+str(dataset_id)+"_"+estimator+"_"+metric+".csv", index=False)

    queue.put(None)
    proc.join()


#experiment_all_datasets_and_estimators()


def debug():
    data, target = fetch_openml(
        data_id=1137, return_X_y=True, as_frame=True)
    print("Downloaded")
    #
    #vector1,x,y = skopt(data, target, estimator="svc_linear", metric="accuracy", n_calls=20, n_features=10, learning_method = "RF", discretization_method="binary", acq_func="PI", intermediate_results=True)
    #vector1, x = skopt(data, target, discretization_method="round", n_features=5, estimator="k_neighbours_classifier", cross_validation=2,
    #                      metric="accuracy", learning_method="GP", kernel="MATERN", acq_func="PI", n_calls=10, intermediate_results=False)
    vector1 = sfs(data, target, n_features=1, estimator="svc_linear")
    print(vector1)
    #print(x)
    # print(y)
    # print(sum(vector1))
    #kf = KFold(n_splits=5, shuffle=True).split(data)
    #for train_index, test_index in kf:
    #    print(get_score(data_training=data.loc[train_index], data_test=data.loc[test_index], target_training=target.loc[train_index], target_test=target.loc[test_index], mask=[
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], estimator="random_forest", metric="accuracy"))
    # print(sett)


debug()
