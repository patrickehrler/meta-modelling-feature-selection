from itertools import compress
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, explained_variance_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_score(data_training, data_test, target_training, target_test, mask, estimator="linear_regression", metric="r2"):
    """ Returns score for a given feature selection and an estimator.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    mask -- 0/1 vector of unselected/selected features
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score

    """

    if data_test is None:
        data_test = data_training
    if target_test is None:
        target_test = target_training


    selected_features = list(compress(data_training.columns, mask))
    filtered_data_training = data_training[selected_features]
    filtered_data_test = data_test[selected_features]
    y_pred = get_estimator(estimator).fit(filtered_data_training, target_training).predict(filtered_data_test)
    
    # regression metrics
    if metric == "r2":
        score = r2_score(y_true=target_test, y_pred=y_pred)
    elif metric == "explained_variance":
        score = explained_variance_score(y_true=target_test, y_pred=y_pred)
    # classification metrics
    elif metric == "accuracy":
        score = accuracy_score(y_true=target_test, y_pred=y_pred)
    else:
        raise ValueError("Invalid metric")

    return score


def convert_vector(vector):
    """ Convert True/False vector to 0/1 vector

    Keyword arguments:
    vector -- vector of True/False values

    """
    return [int(x) for x in vector]


def add_testing_score(data_training, data_test, target_training, target_test, dataframe, estimator, metric):
    """ Calculate score for each row in dataframe and add it as a column

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    dataframe -- pandas dataframe with column 'Vector'
    estimator -- estimator used to predict target-values
    metric -- metric used to calculate score 

    """
    for row in dataframe.index:
        vector = dataframe.loc[row]["Vector"]
        score = get_score(data_training, data_test, target_training, target_test, vector, estimator, metric)
        dataframe.loc[row, "Testing Score"] = score

    return dataframe


def get_estimator(estimator):
    """ Returns estimator object.

    Keyword arguments:
    estimator -- estimator -- estimator name

    """
    if estimator == "linear_regression":
        return LinearRegression()
    elif estimator == "svr_linear":
        return svm.LinearSVR(dual=False, loss="squared_epsilon_insensitive", max_iter=1000) # dual false, because n_samples > n_features
    elif estimator == "svc_linear":
        return svm.LinearSVC(dual=False, max_iter=1000) 
    elif estimator == "k_neighbours_classifier":
        return KNeighborsClassifier(n_neighbors=5)
    elif estimator == "random_forest":
        return RandomForestClassifier()
    elif estimator == "logistic_regression":
        return LogisticRegression()
    else:
        raise ValueError("Invalid estimator.")

