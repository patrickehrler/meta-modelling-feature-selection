import sys
from sklearn.datasets import fetch_openml
from comparison_algorithms import sfs, rfe
from bayesian_algorithms import skopt

print(sys.argv)

# algorithm dictionary
bayesian_approaches = {
    "skopt": {
        "round",
        "n_highest",
        "binary"
    }
}
comparison_approaches = {
    "sfs": "todo description",
    "rfe": "todo description"
}

# import dataset
# IMPORTANT: classification datasets must have numeric classes only
X, y = fetch_openml('wdbc', return_X_y=True, as_frame=True) # https://www.openml.org/d/1510
#print('Dataset downloaded')

res_rfe = rfe(data=X, target=y, n_features=5)
print("RFE:")
print(res_rfe)
res_skopt1 = skopt(data=X, target=y, discretization_method="round")
print("skopt Bay. Opt. (round)")
print(res_skopt1)
res_skopt2 = skopt(data=X, target=y, n_features=5, discretization_method="n_highest")
print("skopt Bay. Opt. (5 highest)")
print(res_skopt2)
res_skopt3 = skopt(data=X, target=y, n_features=5, discretization_method="binary")
print("skopt Bay. Opt. (binary space)")
print(res_skopt3)
