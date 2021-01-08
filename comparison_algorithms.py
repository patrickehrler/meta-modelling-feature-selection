from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from utils import get_score, convert_vector


def sfs(data, target, n_features=None, estimator=LinearRegression()):
    """ Run Sequential Feature Selection (a wrapper method)
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best cross-validation result is selected
    estimator -- estimator used to determine score

    """
    if n_features is None:
        n_features = "best"

    # https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    sfs_selection = SFS(estimator,
                        k_features=n_features,
                        forward=True,
                        verbose=0,
                        cv=0  # no cross validation # TODO: if cross-validation is on, score is smaller than expected
                        )
    sfs_selection.fit(data, target)

    # convert feature-id-array to 0/1 vector
    result_vector = [0 for _ in range(len(data.columns))]
    for index in sfs_selection.k_feature_idx_:
        result_vector[index] = 1

    result_score = get_score(data, target, result_vector, estimator)

    return result_score, result_vector


def rfe(data, target, n_features=10, estimator=LinearRegression()):
    """ Run Recursive Feature Selection (a wrapper method)
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then 10 features will be selected
    estimator -- estimator used to determine score

    """
    rfe_selection = RFE(estimator=estimator,
                        n_features_to_select=n_features
                        )

    rfe_selection.fit(data, target)

    # calculate result vector
    result_vector = convert_vector(rfe_selection.support_)

    result_score = get_score(data, target, result_vector, estimator)

    return result_score, result_vector


def sfm(data, target, n_features=None, estimator=LinearRegression()):
    """ Run Select From Model (an embedded method)
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel.get_support

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- maximum number of features to select
    estimator -- estimator used to determine score

    """
    sfm_selection = SelectFromModel(estimator=estimator, max_features=n_features).fit(data, target)

    # calculate result vector
    result_vector = convert_vector(sfm_selection.get_support())

    # calculate score
    result_score = get_score(
        data, target, result_vector, estimator)

    return result_score, result_vector


# TODO implement more algorithms
