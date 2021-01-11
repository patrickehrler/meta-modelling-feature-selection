from itertools import compress
from sklearn.linear_model import LinearRegression

def get_score(data, target, mask, estimator=LinearRegression()):
    """ Returns score for a given feature selection and an estimator.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    mask -- 0/1 vector of unselected/selected features
    estimator -- estimator used to determine score

    """
    selected_features = list(compress(data.columns, mask))
    filtered_data = data[selected_features]
    score = estimator.fit(filtered_data, target).score(filtered_data, target)

    return score

def convert_vector(vector):
    """ Convert True/False vector to 0/1 vector

    Keyword arguments:
    vector -- vector of True/False values

    """
    return [int(x) for x in vector]

def add_testing_score(data, target, dataframe, estimator):
    """ Calculate score for each row in dataframe and add it as a column

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    dataframe -- pandas dataframe with column 'Vector'

    """
    for row in dataframe.index:
        vector = dataframe.loc[row]["Vector"]
        score = get_score(data, target, vector, estimator)
        dataframe.loc[row, "Testing Score"] = score

    return dataframe
