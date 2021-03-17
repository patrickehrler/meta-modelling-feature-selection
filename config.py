##################
# Settings File
##################

##
# 1. Experiment-specific properties
##
# 1.1. Comparison experiment
# 1.1.1. Bayesian optimization max iterations
n_calls = 100
# 1.1.2 Bayesian optimization convergence: stop optimization after n iterations without new optimum
n_convergence = 20

# 1.2. Iteration Experiment
# 1.2.1. Maximum number of iterations
max_calls = 200



##
# 2. General properties
##
# 2.1. Number of processes for parallelization
n_processes = 4

# 2.2. Number of splits for cross-validation (outside)
n_splits = 5

# 2.3. number of cross-validation splits inside of bayesian optimization
n_splits_bay_opt = 2

# 2.4. Number of features to be selected
min_nr_features = 5
max_nr_features = 20
iter_step_nr_features = 5

# 2.5 Number of points to evaluate of the acquisition function
n_acq_points = 10000

# 2.5. Datasets used
# numbers represent openml.org dataset ids
# booleans represent if a dataset is used for the experiment or not
data_ids = {
    "classification": {
        # experiment datasets
        12: False,
        312: False,
        978: False,
        1038: False,
        1039: False,
        1041: False,
        1042: False,
        1233: False,
        1457: False,
        1458: False,
        1468: False,
        1485: False,
        1501: False,
        1514: False,
        1515: False,
        4134: False,
        40588: False,
        40593: False,
        40594: False,
        40595: False,
        40596: False,
        40910: False,
        40979: False,
        41083: False,
        41144: False,
        41158: False,
        41964: False,
        41966: False,
        41967: False,
        41973: False,
        # very small test dataset
        1510: True # 30 features
    }
}