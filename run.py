import sys
from sklearn.datasets import fetch_openml
from comparison_algorithms import run_comparison_algorithm
from bayesian_algorithms import skopt
import pandas as pd

# algorithm dictionaries
bayesian_approaches = {
    "skopt"
}
comparison_approaches = {
    "SFS": "Sequential Feature Selection",
    "RFE": "Recursive Feature Selection"
}
discretization_methods = {
    "round": "round",
    "n_highest": "n highest",
    "binary": "binary"
}
# Bayesian Optimization learning methods
learning_methods = {
    "GP": "Gaussian Process",
    "RF": "Random Forest",
    "ET": "Extra Trees",
    "GBRT": "Gradient Boosting Quantile"
}
# Gaussian Process kernels
kernels = {
    "MATERN": "Matern",
    "HAMMING": "Hamming"
    
}
n_calls = 20 # number of iterations (bayesian optimization)




# import dataset
# IMPORTANT: classification datasets must have numeric classes only
# https://www.openml.org/d/1510
X, y = fetch_openml('wdbc', return_X_y=True, as_frame=True)
#print('Dataset downloaded')

nr_of_features = len(X.columns)

# run all bayesian optimization techniques (TODO: call different n_features for n_highest)
df_bay_opt = pd.DataFrame(columns=['Score', 'Vector'])
for lm, lm_descr in learning_methods.items():
    for dm, dm_descr in discretization_methods.items(): 
        if lm is "GP": # kernels only for gaussian processes
            for k, k_descr in kernels.items():
                score, vector = skopt(data=X, target=y, kernel=k, n_calls=n_calls, learning_method=lm, discretization_method=dm, n_features=10)
                row_name = lm_descr + " (" + k_descr + ", " + dm_descr + ")"
                df_bay_opt.loc[row_name] = [score, vector]
        else:
            score, vector = skopt(data=X, target=y, kernel=None, n_calls=n_calls, learning_method=lm, discretization_method=dm, n_features=10)
            row_name = lm_descr + "(" + dm_descr + ")"
            df_bay_opt.loc[row_name] = [score, vector]



# run all comparison approaches
df_comparison = pd.DataFrame(columns=['Score', 'Vector'])
for a, a_descr in comparison_approaches.items():
    for n_features in range(10, nr_of_features+1, 5):
        score, vector = run_comparison_algorithm(type=a, data=X, target=y, n_features=n_features)
        row_name = a_descr + " (n_features=" + str(n_features) + ")"
        df_comparison.loc[row_name] = [score, vector]


print(df_bay_opt)
print(df_comparison)
