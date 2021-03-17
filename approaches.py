from bayesian_algorithms import skopt, gpyopt
from comparison_algorithms import rfe, sfs, sfm, n_best_anova_f, n_best_mutual, n_best_pearsonr, pymrmr_fs, binary_swarm

# Estimator and metric properties (choosing estimator cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
classification_estimators = {
    "svc_linear": {
        "accuracy": "Support Vector Classification - Accuracy Score",
        "matthews": "Support Vector Classification - Matthews correlation"
    },
    "k_neighbours_classifier": {
         "accuracy": "k Neighbours Classification - Accuracy Score",
         "matthews": "k Neighbours Classification - Matthews correlation"
    },
    "random_forest": {
        "accuracy": "Random Forest - Accuracy Score",
        "matthews": "Random Forest - Matthews correlation"
    },
    "logistic_regression": {
        "accuracy": "Logistic Regression Classifier - Accuracy Score",
        "matthews": "Logistic Regression Classifier - Matthews correlation"
    }
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
    skopt: "Scikit Optimize",
    #gpyopt: "GPyOpt"
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
    "RBF": "Radial Basis Functions"  # squared-exponential kernel
}
discretization_methods = {
    "round": "round",
    "probabilistic_round": "probabilistic round",
    "n_highest": "n highest",
    "categorical": "Categorical",
    "binary": "binary"
}
acquisition_functions = {
    # TODO: maybe just gp_hedge? - depends on performance
    "LCB": "lower confidence bound",
    "EI": "expected improvement",
    "PI": "probability of improvement",
    "gp_hedge": "combination of PI, EI and LCB"
}

# Comparison Approaches Properties
comparison_parameters = ["Approach", "Algorithm", "n_features"]
comparison_approaches = {
    "filter": {
        #vt: "Variance Threshold",
        n_best_anova_f: "ANOVA f-values",
        n_best_mutual: "Highest mutual score",
        #n_best_pearsonr: "Highest pearson correlation", # probably not useful for classification targets
        pymrmr_fs: "mRMR",
        binary_swarm: "Binary particle swarm optimization"
    },
    "wrapper": {
        sfs: "Sequential Feature Selection",
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm: "Select From Model"
    }
}