from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def run_comparison_algorithm(type="RFE", **kwargs):
    if type == "RFE":
        return __rfe(**kwargs)
    elif type == "SFS":
        return __sfs(**kwargs)
    else:
        print("Error: Undefined Comparison Algorithm")


def __sfs(data, target, n_features=None, estimator=LinearRegression()):
    """ Run Sequential Feature Selection
    
    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best cross-validation result is selected
    estimator -- estimator used to determine score
    
    """
    if n_features == None:
        n_features = "best"

    # https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    sfs_selection = SFS(estimator,
          k_features=n_features,
          forward=True,
          verbose=0,
          #scoring = 'r2', # use sklearn regressor 
          cv = 0 # no cross validation # TODO: if cross-validation is on, score is smaller than expected
          )
    sfs_selection.fit(data,target)
    
    result_score = sfs_selection.k_score_
    
    result_vector = [0 for element in range(len(data.columns))]
    for index in sfs_selection.k_feature_idx_:
        result_vector[index] = 1

    return result_score, result_vector


def __rfe(data, target, n_features=10, estimator=LinearRegression()):
    """ Run Recursive Feature Selection
    
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

    result_score = rfe_selection.score(data, target)
    result_vector=[]
    for el in rfe_selection.support_:
        if el == False:
            result_vector.append(0)
        else:
            result_vector.append(1)

    return result_score, result_vector



# TODO implement more algorithms
