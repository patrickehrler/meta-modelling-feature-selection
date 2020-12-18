from skopt.space import Integer, Real
from skopt import gp_minimize
from sklearn.linear_model import LinearRegression
from itertools import compress
import numpy as np


def skopt(data, target, n_features=None, discretization_method="round", estimator=LinearRegression(), acq_func="PI", n_calls=20, n_random_starts=5, random_state=123):
    # create feature space
    space = []
    if discretization_method == "binary":  # use binary skopt Integer values
        for feature_name in data.columns:
            space.append(Integer(0, 1, name=feature_name))
    elif discretization_method == "round" or discretization_method == "n_highest":  # round real number (0,1)
        for feature_name in data.columns:
            space.append(Real(0, 1, name=feature_name))

    # define black box function
    def black_box_function(*args):
        input = args[0]
        mask = [0 for i in range(len(data.columns))]
        if discretization_method == "round":
            mask = list(map(lambda x: round(x), input))
        elif discretization_method == "n_highest":
            if n_features != None:
                for i in range(n_features):
                    # detect maximum; in case of multiple occurrences the first is used
                    max_index = np.argmax(args[0])
                    # set mask value
                    mask[max_index] = 1
                    # remove input value
                    input[max_index] = 0
            else:
                print("Error: n_features is not defined")
        else:
            mask = input
        # TODO: more methods (kernels, ...)

        # create list of selected features
        selected_features = list(compress(data.columns, mask))
        # remove all unselected feature columns
        filtered_data = data[data.columns[data.columns.isin(
            selected_features)]]
        model = estimator.fit(filtered_data, target)
        # coefficient of determination
        score = 1-model.score(filtered_data, target)

        return score

    # run bayesian optimization
    optimizer = gp_minimize(black_box_function,  # the function to minimize
                                  space,      # the bounds on each dimension of x
                                  acq_func=acq_func,      # the acquisition function
                                  n_calls=n_calls,         # the number of evaluations of f
                                  n_random_starts=n_random_starts,  # the number of random initialization points
                                  random_state=random_state,  # the random seed
                                  verbose=False)

    result_score = 1-optimizer.fun

    if discretization_method == "round":
        result_vector = list(map(lambda x: round(x), optimizer.x))
    elif discretization_method == "n_highest":
        result_vector = [0 for i in range(len(data.columns))]
        result_x = optimizer.x
        for i in range(n_features):
            # detect maximum; in case of multiple occurrences the first is used
            max_index = np.argmax(result_x)
            # set mask value
            result_vector[max_index] = 1
            # remove input value
            result_x[max_index] = 0
    else:
        result_vector = optimizer.x
    
    return result_score, result_vector