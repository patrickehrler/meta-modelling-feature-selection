##################
# Settings File
##################

# Experiment-specific properties
# 1. Comparison experiment
# 1.1 Bayesian optimization max iterations
n_calls = 50
# 2. Iteration Experiment
# 2.1 Maximum number of iterations
max_calls = 200

# number of processes for parallelization
n_processes = 4

# number of splits for cross-validation (outside)
n_splits = 5
# number of cross-validation splits inside of bayesian optimization
n_splits_bay_opt = 2

# number of features to be selected
min_nr_features = 5
max_nr_features = 20
iter_step_nr_features = 5

# openml.org dataset id 
data_ids = {
    "classification": {
        # experiment datasets
        12: True,
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
        # very small test datasets
        1510: False # 30 features
    },
    "regression": {
        1510: False, # 30 features
        1485: False, # 500 features
        1458: False, # 10000 features
        1079: False # 22278 features
    }
}