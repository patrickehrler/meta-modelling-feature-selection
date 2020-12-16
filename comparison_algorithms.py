from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def sfs():
    print('sfs')


def rfe(data, target, n_features=10, step_size=1, estimator=LinearRegression()):
    rfe_selection = RFE(estimator=estimator,
                        n_features_to_select=n_features,
                        step=step_size)
    
    rfe_selection.fit(data, target)

    return rfe_selection.support_



# TODO implement more algorithms
