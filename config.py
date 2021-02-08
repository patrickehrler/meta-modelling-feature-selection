##################
# Settings File
##################

# number of processes for parallelization
n_processes = 32

# number of splits for cross-validation
n_splits = 5

# number of iterations in bayesian optimization
n_calls = 50

# number of features to be selected
min_nr_features = 5
max_nr_features = 30
iter_step_nr_features = 5

# openml.org dataset id 
# IMPORTANT: datasets must have numeric target classes only
data_ids = {
    "classification": {
        1510: False, # 30 features
        1485: False, # 500 features
        1039: True,
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