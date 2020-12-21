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

learning_methods = {
    "GP": "Gaussian Process",
    "RF": "Random Forest",
    "ET": "Extra Trees",
    "GBRT": "Gradient Boosting Quantile"
}

kernels = {
    "MATERN": "Matern",
    "HAMMING": "Hamming"
    
}

# import dataset
# IMPORTANT: classification datasets must have numeric classes only
# https://www.openml.org/d/1510
X, y = fetch_openml('wdbc', return_X_y=True, as_frame=True)
#print('Dataset downloaded')


# todo create result dataframe with all scores (learning method, kernels, discretization methods, comparison approaches, ...)

for lm, lm_descr in learning_methods.items():
    if lm is "GP":
        for k, k_descr in kernels.items():
            print(lm_descr + " : " + k_descr)
            res_test = skopt(data=X, target=y, kernel=k, n_calls=20, learning_method=lm)
            print(res_test)
    else:
        print(lm_descr)
        res_test = skopt(data=X, target=y, kernel=None, n_calls=20, learning_method=lm)
        print(res_test)


res_rfe = rfe(data=X, target=y, n_features=5)
print("RFE:")
print(res_rfe)
res_skopt1 = skopt(data=X, target=y, discretization_method="round")
print("skopt Bay. Opt. (round)")
print(res_skopt1)
res_skopt2 = skopt(data=X, target=y, n_features=5,
                   discretization_method="n_highest")
print("skopt Bay. Opt. (5 highest)")
print(res_skopt2)
res_skopt3 = skopt(data=X, target=y, n_features=5,
                   discretization_method="binary")
print("skopt Bay. Opt. (binary space)")
print(res_skopt3)
