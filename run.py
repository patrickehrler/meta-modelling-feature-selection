import sys
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

from comparison_algorithms import sfs, rfe

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
# IMPORTANT: classification datasets must have numeric classes
# currently basic dataset is used to have a good performance for development
X, y = fetch_openml('pendigits', return_X_y=True, as_frame=True) # https://www.openml.org/d/1106
print('Dataset downloaded')
res = rfe(data=X, target=y, n_features=5)
print(res)


# TODO: call algorithms using dataset