from itertools import compress


def get_score(data, target, mask, estimator):
    """ Returns score for a given feature selection and an estimator.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    mask -- 0/1 vector of unselected/selected features
    estimator -- estimator used to determine score

    """
    selected_features = list(compress(data.columns, mask))
    filtered_data = data[data.columns[data.columns.isin(selected_features)]]
    score = estimator.fit(filtered_data, target).score(filtered_data, target)

    return score


def convert_vector(vector):
    """ Convert True/False vector to 0/1 vector

    Keyword arguments:
    vector -- vector of True/False values

    """
    return [int(x) for x in vector]
