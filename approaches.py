from bayesian_algorithms import skopt
from comparison_algorithms import rfe, sfs, sfm, vt, skb

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
                      "Kernel", "Discretization Method", "Acquisition Function", "n_features"]
bayesian_approaches = {
    skopt: "Scikit Optimize"
}
learning_methods = {
    "GP": "Gaussian Process",
    "RF": "Random Forest",
    #"ET": "Extra Trees",
    #"GBRT": "Gradient Boosting Quantile"
}
kernels = {
    # kernels for Gaussian Processes only
    "MATERN": "Matern",
    "HAMMING": "Hamming",
    #"RBF": "Radial Basis Functions"  # squared-exponential kernel
}
discretization_methods = {
    "round": "round",
    "probabilistic_round": "probabilistic round",
    "n_highest": "n highest",
    #"binary": "binary"
}
acquisition_functions = {
    #"LCB": "lower confidence bound",
    #"EI": "expected improvement",
    "PI": "probability of improvement"
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