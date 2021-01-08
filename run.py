from sklearn.datasets import fetch_openml
from comparison_algorithms import sfs, rfe, sfm
from bayesian_algorithms import skopt
import pandas as pd

#
# Bayesian Optimization Settings
#
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
n_calls = 20  # number of iterations (bayesian optimization)

#
# Comparison Approaches Settings
#
comparison_approaches = {
    "wrapper": {
        sfs: "Sequential Feature Selection",
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm: "Select From Model"
    }
}


# import dataset
# IMPORTANT: classification datasets must have numeric target classes only
# Dataset with 30 features (https://www.openml.org/d/1510)
X, y = fetch_openml('wdbc', return_X_y=True, as_frame=True)

# alternative datasets
# Dataset with 10000 features (https://www.openml.org/d/1458)
# X, y = fetch_openml(data_id=1458, return_X_y=True, as_frame=True)
# dataset with 500 features (https://www.openml.org/d/1485)
# X, y = fetch_openml(data_id=1485, return_X_y=True, as_frame=True)

print('Dataset downloaded')
nr_of_features = len(X.columns)

# TODO Question: Why does RBF kernel work with binary search space?


def __run_all():
    # run all bayesian optimization approaches
    df_bay_opt = pd.DataFrame(columns=['Score', 'Vector'])
    for func, func_descr in bayesian_approaches.items():
        for lm, lm_descr in learning_methods.items():
            for dm, dm_descr in discretization_methods.items():
                if lm == "GP":  # kernels only for gaussian processes
                    for k, k_descr in kernels.items():
                        if dm == "n_highest":
                            for n_features in range(5, nr_of_features, 5):
                                score, vector = func(
                                    data=X, target=y, kernel=k, n_features=n_features, n_calls=n_calls, learning_method=lm, discretization_method=dm)
                                row_name = func_descr + \
                                    " (" + lm_descr + ", " + k_descr + ", " + \
                                    dm_descr + ", n_features=" + \
                                    str(n_features) + ")"
                                df_bay_opt.loc[row_name] = [score, vector]
                        else:
                            score, vector = func(
                                data=X, target=y, kernel=k, n_calls=n_calls, learning_method=lm, discretization_method=dm)
                            row_name = func_descr + \
                                " (" + lm_descr + ", " + \
                                k_descr + ", " + dm_descr + ")"
                            df_bay_opt.loc[row_name] = [score, vector]

                else:
                    if dm == "n_highest":
                        for n_features in range(5, nr_of_features, 5):
                            score, vector = func(
                                data=X, target=y, kernel=None, n_features=n_features, n_calls=n_calls, learning_method=lm, discretization_method=dm)
                            row_name = func_descr + \
                                " (" + lm_descr + ", " + dm_descr + \
                                ", n_features=" + str(n_features) + ")"
                            df_bay_opt.loc[row_name] = [score, vector]
                    else:
                        score, vector = func(
                            data=X, target=y, kernel=None, n_features=None, n_calls=n_calls, learning_method=lm, discretization_method=dm)
                        row_name = func_descr + \
                            " (" + lm_descr + ", " + dm_descr + ")"
                        df_bay_opt.loc[row_name] = [score, vector]

    # run all comparison approaches
    df_comparison = pd.DataFrame(columns=['Score', 'Vector'])
    for a, a_descr in comparison_approaches.items():
        for func, func_desc in a_descr.items():
            if func == sfm:
                score, vector = func(
                    data=X, target=y)
                row_name = a + ": " + func_desc
                df_comparison.loc[row_name] = [score, vector]
            else:
                for n_features in range(5, nr_of_features+1, 5):
                    score, vector = func(
                        data=X, target=y, n_features=n_features)
                    row_name = a + ": " + func_desc + \
                        " (n_features=" + str(n_features) + ")"
                    df_comparison.loc[row_name] = [score, vector]

    # save results in file and print to console
    df_bay_opt.to_csv('results/bay_opt.csv', index=True)
    df_comparison.to_csv('results/comparison.csv', index=True)
    print(df_bay_opt)
    print(df_comparison)


__run_all()

# def __debug():
#    score1, vector1 = run_comparison_algorithm(type="SFM", data=X, target=y)
#    print(score1)
#    print(vector1)
# __debug()
