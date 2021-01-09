import numpy as np
from sklearn.datasets import fetch_openml
from comparison_algorithms import sfs, rfe, sfm
from bayesian_algorithms import skopt
from sklearn.model_selection import KFold
import pandas as pd
from utils import get_score, add_cv_score
from sklearn.linear_model import LinearRegression

##################
# Settings
##################
# number of splits for cross-validation
n_splits = 2
# number of iterations in bayesian optimization
n_calls = 20
# openml.org dataset id (30 features: 1510, 10000 features: 1458, 500 features: 1485); # IMPORTANT: classification datasets must have numeric target classes only
# TODO: put datasets into dictionary
data_id = 1510
# estimator used for training process
estimator_training = LinearRegression()
# estimator used for test process
estimator_test = LinearRegression()


# Bayesian Optimization Properties
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
comparison_approaches = {
    "wrapper": {
        sfs: "Sequential Feature Selection",
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm: "Select From Model"
    }
}


# Import dataset
X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
print("Dataset downloaded")
nr_of_features = len(X.columns)


def run_all_bayesian(data, target):
    # Define result dataframes
    dataframe = pd.DataFrame(columns=["Algorithm", "Learning Method", "Kernel",
                                      "Discretization Method", "n_features", "Vector", "Training Score"])

    for algo, algo_descr in bayesian_approaches.items():
        for learn, learn_descr in learning_methods.items():
            for discr, discr_descr in discretization_methods.items():
                if learn == "GP":
                    for kernel, kernel_descr in kernels.items():
                        if discr == "n_highest":
                            # TODO: with more than one n_features
                            n_features = round(nr_of_features/2)
                            vector = algo(data=data, target=target, learning_method=learn,
                                          kernel=kernel, discretization_method=discr, n_features=n_features)
                        else:
                            vector = algo(data=data, target=target, learning_method=learn,
                                          kernel=kernel, discretization_method=discr)
                            n_features = "-"
                        score = get_score(
                            data, target, vector, estimator_training)
                        dataframe.loc[len(dataframe)] = [
                            algo_descr, learn_descr, kernel_descr, discr_descr, n_features, vector, score]
                else:
                    if discr == "n_highest":
                        n_features = round(nr_of_features/2)
                        vector = algo(data=data, target=target, learning_method=learn,
                                      discretization_method=discr, n_features=n_features)
                    else:
                        vector = algo(
                            data=data, target=target, learning_method=learn, discretization_method=discr)
                        n_features = "-"
                    score = get_score(data, target, vector, estimator_training)
                    dataframe.loc[len(dataframe)] = [
                        algo_descr, learn_descr, "-", discr_descr, n_features, vector, score]
    return dataframe


def run_all_comparison(data, target):
    # Define result dataframe
    dataframe = pd.DataFrame(
        columns=["Approach", "Algorithm", "n_features", "Vector", "Training Score"])

    for approach, approach_descr in comparison_approaches.items():
        for algo, algo_descr in approach_descr.items():
            for n_features in range(5, nr_of_features+1, 5):
                vector = algo(data=data, target=target, n_features=n_features)
                score = get_score(data=data, target=target,
                                  mask=vector, estimator=estimator_training)
                dataframe.loc[len(dataframe)] = [
                    approach, algo_descr, n_features, vector, score]
    return dataframe


# Initialize result dataframe
df_bay_opt = pd.DataFrame(columns=["Algorithm", "Learning Method", "Kernel",
                                   "Discretization Method", "n_features", "Vector", "Training Score", "Testing Score"])
df_comparison = pd.DataFrame(columns=["Approach", "Algorithm", "n_features", "Vector", "Training Score", "Testing Score"])

# Split dataset into testing and training data
kf = KFold(n_splits=n_splits, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #
    # run all bayesian approaches
    #
    df_current_bayesian = run_all_bayesian(X_train, y_train)
    df_current_bayesian = add_cv_score(
        X_test, y_test, df_current_bayesian, estimator_test)
    df_bay_opt = pd.concat(
        [df_bay_opt, df_current_bayesian], ignore_index=True)
    #
    # run all comparison approaches
    #
    df_current_comparison = run_all_comparison(X_train, y_train)
    df_current_comparison = add_cv_score(
        X_test, y_test, df_current_comparison, estimator_test)
    df_comparison = pd.concat(
        [df_comparison, df_current_comparison], ignore_index=True)

print(df_bay_opt)
print(df_comparison)
df_bay_opt.to_csv("results/bay_opt.csv", index=False)
df_comparison.to_csv("results/comparison.csv", index=False)


# TODO Question: Why does RBF kernel work with binary search space?

# def __debug():
#    score1, vector1 = run_comparison_algorithm(type="SFM", data=X, target=y)
#    print(score1)
#    print(vector1)
# __debug()
