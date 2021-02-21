##################
# Settings File
##################

# Experiment-specific properties
# 1. Comparison experiment
# 1.1 Bayesian optimization max iterations
n_calls = 50
# 2. Iteration Experiment
# 2.1 Maximum number of iterations
max_calls = 5

# number of processes for parallelization
n_processes = 32

# number of splits for cross-validation
n_splits = 5

# number of features to be selected
min_nr_features = 5
max_nr_features = 20
iter_step_nr_features = 5

# openml.org dataset id 
data_ids = {
    "classification": {
        # experiment datasets
        1039: True,
        1128: False,
        1130: False,
        1134: False,
        1137: False,
        1138: False,
        1139: False,
        1142: False,
        1145: False,
        1146: False,
        1158: False,
        1161: False,
        1165: False,
        1166: False,
        4134: False,
        41142: False,
        # test datasets
        1510: False, # 30 features
        1485: False, # 500 features
        1458: False, # 10000 features
        1079: False # 22278 features
    },
    "regression": {
        1510: False, # 30 features
        1485: False, # 500 features
        1458: False, # 10000 features
        1079: False # 22278 features
    }
}