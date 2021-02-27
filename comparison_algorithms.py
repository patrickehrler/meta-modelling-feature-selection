from bayesian_algorithms import discretize
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE, SelectFromModel, VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from utils import convert_vector, get_estimator
import numpy as np


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
    if n_features is None:
        n_features = "best"

    # https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    sfs_selection = SFS(get_estimator(estimator),
                        k_features=n_features,
                        forward=True,
                        verbose=2,
                        cv=0,  # disable cross validation
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
    rfe_selection = RFE(estimator=get_estimator(estimator),
                        n_features_to_select=n_features,
                        verbose=2
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
    sfm_selection = SelectFromModel(estimator=get_estimator(
        estimator), max_features=n_features, threshold=-np.inf).fit(data, target)

    # calculate result vector
    result_vector = convert_vector(sfm_selection.get_support())

    return result_vector


def vt(data, target=None):
    """ Run Variance Threshold feature selection. Threshhold is 0, which means only features with zero-variance are removed.

    Keyword arguments:
    data -- feature matrix
    target -- ignored (only for compatibility)

    """
    # TODO: try different tresholds
    vt_selection = VarianceThreshold(threshold=0)
    vt_selection.fit(data)

    result_vector = convert_vector(vt_selection.get_support())

    return result_vector


def skb(data, target, n_features, estimator=None):
    """ Run SelectKBest feature selection.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- ignored (only for compatibility)

    """
    # TODO: try different score_func (chi2)
    skb_selection = SelectKBest(score_func=f_classif, k=n_features).fit(data, target)

    result_vector = convert_vector(skb_selection.get_support())

    return result_vector

def mutual(data, target, n_features, estimator=None):
    """ Calculate mutual score for each feature, then select n highest features.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to select
    estimator -- ignored (only for compatibility)

    """

    mutual_selection = mutual_info_classif(data, target)

    # select n features with highest mutual score
    result_vector = discretize(mutual_selection, "n_highest", n_features)

    return result_vector