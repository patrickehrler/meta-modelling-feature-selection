from bayesian_algorithms import skopt, gpyopt
from comparison_algorithms import rfe, sfs, sfm, n_best_anova_f, n_best_mutual, n_best_pearsonr, pymrmr_fs, binary_swarm

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold

from utils import get_score, add_testing_score

def debug():
    data, target = fetch_openml(
        data_id=1510, return_X_y=True, as_frame=True)
    print("Downloaded")

    vector1, x = skopt(data, target, estimator="svc_linear", penalty_weight=0, metric="accuracy", n_calls=10, n_features=20, learning_method = "GP", discretization_method="categorical", acq_func="PI", intermediate_results=False, acq_optimizer="n_sampling", cross_validation=2)
    print(vector1)
    print(x)

    #vector1 = pymrmr_fs(data, target, n_features=20, estimator="logistic_regression")
    #print(vector1)
    #print(sum(vector1))

    # cross-validated result scores
    kf = KFold(n_splits=5, shuffle=True).split(data)
    for train_index, test_index in kf:
        print(get_score(data_training=data.loc[train_index], data_test=data.loc[test_index], target_training=target.loc[train_index], target_test=target.loc[test_index], mask=vector1, estimator="logistic_regression", metric="accuracy"))


debug()
