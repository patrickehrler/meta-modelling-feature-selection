from bayesian_algorithms import discretize
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from utils import convert_vector, get_estimator
import numpy as np
from scipy.stats import pearsonr
import pymrmr
import pandas as pd
import pyswarms as ps
import pandas as pd


def sfs(data, target, n_features=None, estimator="linear_regression", metric=None):
    """ Run Sequential Feature Selection (a wrapper method)
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best cross-validation result is selected
    estimator -- estimator used to determine score
    metric -- metric used to calculate score

    """
    print("sfs start")
    if n_features is None:
        n_features = "best"

    # https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    sfs_selection = SFS(get_estimator(estimator),
                        k_features=n_features,
                        forward=True,
                        verbose=0,
                        cv=2,  # disable cross validation
                        scoring=metric
                        )
    sfs_selection.fit(data, target)

    # convert feature-id-array to 0/1 vector
    result_vector = [0 for _ in range(len(data.columns))]
    for index in sfs_selection.k_feature_idx_:
        result_vector[index] = 1

    return result_vector


def rfe(data, target, n_features=10, estimator="linear_regression"):
    """ Run Recursive Feature Selection (a wrapper method)
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then 10 features will be selected
    estimator -- estimator used to determine score

    """
    print("rfe start")
    rfe_selection = RFE(estimator=get_estimator(estimator),
                        n_features_to_select=n_features,
                        verbose=0
                        )

    rfe_selection.fit(data, target)

    # calculate result vector
    result_vector = convert_vector(rfe_selection.support_)

    return result_vector


def sfm(data, target, n_features=None, estimator="linear_regression"):
    """ Run Select From Model (an embedded method)
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel.get_support

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- estimator used to determine score

    """
    print("sfm start")
    sfm_selection = SelectFromModel(estimator=get_estimator(
        estimator), max_features=n_features, threshold=-np.inf).fit(data, target)

    # calculate result vector
    result_vector = convert_vector(sfm_selection.get_support())

    return result_vector


"""def vt(data, target=None):
     Run Variance Threshold feature selection. Threshhold is 0, which means only features with zero-variance are removed.

    Keyword arguments:
    data -- feature matrix
    target -- ignored (only for compatibility)

    
    print("vt start")
    # TODO: try different tresholds
    vt_selection = VarianceThreshold(threshold=0)
    vt_selection.fit(data)

    result_vector = convert_vector(vt_selection.get_support())

    return result_vector
"""

def n_best_anova_f(data, target, n_features, estimator=None):
    """ Run SelectKBest feature selection.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- ignored (only for compatibility)

    """
    print("anova start")

    skb_selection = SelectKBest(score_func=f_classif, k=n_features).fit(data, target)

    result_vector = convert_vector(skb_selection.get_support())

    return result_vector

def n_best_mutual(data, target, n_features, estimator=None):
    """ Calculate mutual score for each feature, then select n highest features.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- ignored (only for compatibility)

    """
    print("mutual start")
    mutual_selection = SelectKBest(score_func=mutual_info_classif, k=n_features).fit(data,target)

    # select n features with highest mutual score
    result_vector = convert_vector(mutual_selection.get_support())

    return result_vector

def n_best_pearsonr(data, target, n_features, estimator=None):
    """ Calculate pearson correlation for each feature, then select n highest features.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- ignored (only for compatibility)

    """
    print("pearson start")
    pearson_selection = [abs(pearsonr(data.loc[:,feature], target.astype("float"))[1]) for feature in data.columns]
    result_vector = discretize(pearson_selection, "n_highest", n_features)

    return result_vector

def pymrmr_fs(data, target, n_features, estimator=None):
    # discretize data to integers
    target_data = pd.concat([target, data], axis=1)
    target_data = target_data.apply(lambda x: pd.factorize(x)[0])

    result = pymrmr.mRMR(target_data, 'MIQ', n_features)

    # convert feature names to 0/1 vector
    result_vector = [0 for _ in data.columns]
    for i in range(0,len(result_vector)):
        if data.columns[i] in result:
            result_vector[i] = 1

    return result_vector

def binary_swarm(data, target, n_features, estimator=None):
    """ Binary Particle Swarm optimization
        Source: https://pyswarms.readthedocs.io/en/development/examples/feature_subset_selection.html

        Keyword arguments:
        data -- feature matrix
        target -- regression or classification targets
        n_features -- number of features to select
        estimator -- estimator used to determine score
    """

    # TODO: include number of desired features

    estimator = get_estimator(estimator)
    total_features = len(data.columns)
    # Define objective function
    def f_per_particle(mask, alpha):
        # Get the subset of the features from the binary mask
        if np.count_nonzero(mask) == 0:
            X_subset = data.values
        else:
            X_subset = data.values[:,mask==1]
        # Perform classification and store performance in P
        estimator.fit(X_subset, target)
        P = (estimator.predict(X_subset) == target).mean()
        # Compute for the objective function
        j = (alpha * (1.0 - P)
            + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]

        return np.array(j)
    
    # Call instance of PSO
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
    dimensions = total_features # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

    # Perform optimization
    _, pos = optimizer.optimize(f, iters=100, verbose=2)

    result_vector = convert_vector(pos)

    return result_vector