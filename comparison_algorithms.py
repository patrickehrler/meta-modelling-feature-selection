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


def __sfs(data, target, n_features=10, step_size=1, estimator=LinearRegression()):
    sfs_selection = SFS(estimator,
          k_features=n_features,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
    sfs_selection.fit(data,target)
    
    result_score = sfs_selection.k_score_
    
    result_vector = [0 for element in range(len(data.columns))]
    for index in sfs_selection.k_feature_idx_:
        result_vector[index] = 1

    return result_score, result_vector


def __rfe(data, target, n_features=10, step_size=1, estimator=LinearRegression()):
    rfe_selection = RFE(estimator=estimator,
                        n_features_to_select=n_features,
                        step=step_size)
    
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
