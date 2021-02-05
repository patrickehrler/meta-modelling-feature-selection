import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from utils import convert_vector, get_estimator


def sfs(data, target, n_features=None, estimator="linear_regression", metric=None):
    """ Run Sequential Feature Selection (a wrapper method)
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best cross-validation result is selected
    estimator -- estimator used to determine score

    """
    print("start sfs: " + str(n_features) + " " + str(estimator) + " " + str(metric))
    if n_features is None:
        n_features = "best"

    # https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    sfs_selection = SFS(get_estimator(estimator),
                        k_features=n_features,
                        forward=True,
                        verbose=0,
                        cv=0,  # disable cross validation
                        scoring=metric
                        )
    sfs_selection.fit(data, target)

    # convert feature-id-array to 0/1 vector
    result_vector = [0 for _ in range(len(data.columns))]
    for index in sfs_selection.k_feature_idx_:
        result_vector[index] = 1

    print("stop sfs: " + str(n_features) + " " + str(estimator) + " " + str(metric))    

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
    print("start rfe: " + str(n_features) + " " + str(estimator))
    rfe_selection = RFE(estimator=get_estimator(estimator),
                        n_features_to_select=n_features
                        )

    rfe_selection.fit(data, target)

    # calculate result vector
    result_vector = convert_vector(rfe_selection.support_)
    print("stop rfe: " + str(n_features) + " " + str(estimator))

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
    print("start sfm: " + str(n_features) + " " + str(estimator))
    sfm_selection = SelectFromModel(estimator=get_estimator(
        estimator), max_features=n_features, threshold=-np.inf).fit(data, target)

    # calculate result vector
    result_vector = convert_vector(sfm_selection.get_support())
    print("stop sfm: " + str(n_features) + " " + str(estimator))

    return result_vector


# TODO implement more algorithms
