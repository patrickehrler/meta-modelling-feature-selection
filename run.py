import sys
from sklearn.datasets import fetch_openml

print(sys.argv)

# algorithm dictionary
bayesian_approaches = {
    "test": "TEST"
}
comparison_approaches = {
    "sfs": "todo description",
    "rfe": "todo description"
}

# import dataset
dataset = fetch_openml('duke-breast-cancer', return_X_y=False) # https://www.openml.org/d/1434

# TODO: call algorithms using dataset