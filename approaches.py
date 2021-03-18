from bayesian_algorithms import skopt
from comparison_algorithms import rfe, sfs, n_best_anova_f, n_best_mutual, pymrmr_fs_mid, pymrmr_fs_miq, sfm_svc, sfm_logistic_regression, sfm_random_forest

# Estimator and metric properties (choosing estimator cheatsheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
classification_estimators = {
    "svc_linear": {
        #"accuracy": "Support Vector Classification - Accuracy Score",
        "matthews": "Support Vector Classification - Matthews correlation"
    },
    #"k_neighbours_classifier": {
    #     #"accuracy": "k Neighbours Classification - Accuracy Score",
    #     "matthews": "k Neighbours Classification - Matthews correlation"
    #},
    #"random_forest": {
    #    #"accuracy": "Random Forest - Accuracy Score",
    #    "matthews": "Random Forest - Matthews correlation"
    #},
    "logistic_regression": {
        #"accuracy": "Logistic Regression Classifier - Accuracy Score",
        "matthews": "Logistic Regression Classifier - Matthews correlation"
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
    "RF": "Random Forest"
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
    #"LCB": "lower confidence bound",
    #"EI": "expected improvement",
    #"PI": "probability of improvement",
    "gp_hedge": "combination of PI, EI and LCB"
}

# Comparison Approaches Properties
comparison_parameters = ["Approach", "Algorithm", "n_features"]
comparison_approaches = {
    "filter": {
        #vt: "Variance Threshold",
        n_best_anova_f: "ANOVA f-values",
        n_best_mutual: "Highest mutual score",
        pymrmr_fs_mid: "mRMR MID scheme",
        pymrmr_fs_miq: "mRMR MIQ scheme"
    },
    "wrapper": {
        sfs: "Sequential Feature Selection",
        rfe: "Recursive Feature Selection"
    },
    "embedded": {
        sfm_svc: "SFM SVC Linear",
        sfm_random_forest: "SFM Random Forest",
        sfm_logistic_regression: "SFM Logistic Regression"
    }
}