from sklearn.utils.validation import check_random_state
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                                     ExpSineSquared, DotProduct,
                                                     ConstantKernel, HammingKernel)
from skopt.space import Integer, Real
from skopt.optimizer import base_minimize
from sklearn.linear_model import LinearRegression
from itertools import compress
import numpy as np
from skopt.utils import cook_estimator


def skopt(data, target, n_features=None, kernel=None, learning_method="GP", discretization_method="round", estimator=LinearRegression(), acq_func="PI", n_calls=20, n_random_starts=5, random_state=123, noise="gaussian"):
    # define black box function
    def black_box_function(*args):
        input = args[0]

        # apply discretization method on value to be evaluated (TODO: put to separate function)
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

        # create list of selected features
        selected_features = list(compress(data.columns, mask))
        # remove all unselected feature columns
        filtered_data = data[data.columns[data.columns.isin(
            selected_features)]]
        model = estimator.fit(filtered_data, target)
        # coefficient of determination
        score = 1-model.score(filtered_data, target)

        return score
    
    # create feature space
    space = []
    if discretization_method == "binary":  # use binary (Integer) values
        for feature_name in data.columns:
            space.append(Integer(0, 1, name=feature_name)) # use real values in range (0,1)
    elif discretization_method == "round" or discretization_method == "n_highest":
        for feature_name in data.columns:
            space.append(Real(0, 1, name=feature_name))

    # define base estimator
    if kernel is not None:
        if learning_method is not "GP": # gaussian processes
             # TODO: Can random forests, ... also use different kernels?
            print('Error: Kernels can only be used with Gaussian Processes. GP will be used.')
        if kernel == "MATERN":
            base_estimator=GaussianProcessRegressor(Matern())
        elif kernel == "HAMMING":
            base_estimator=GaussianProcessRegressor(HammingKernel())
        else:
            print('Error: Invalid kernel. Matern Kernel is used.')
            base_estimator=GaussianProcessRegressor(Matern())
    else:
        rng = check_random_state(random_state)
        if learning_method == "GP":
            # As implemented in gp_minimize (https://github.com/scikit-optimize/scikit-optimize/blob/de32b5f/skopt/optimizer/gp.py#L12)
            base_estimator=cook_estimator(
                learning_method, space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
                noise=noise)
        else:
            base_estimator=cook_estimator(
                learning_method, space=space, random_state=rng.randint(0, np.iinfo(np.int32).max))
    
    # run bayesian optimization
    optimizer = base_minimize(
        func=black_box_function,  # the function to minimize
        base_estimator=base_estimator,
        dimensions=space,      # the bounds on each dimension of x
        acq_func=acq_func,      # the acquisition function
        acq_optimizer="auto",   # configured on the basis of the space searched over
        n_calls=n_calls,         # the number of evaluations of f
        n_random_starts=n_random_starts,  # the number of random initialization points
        random_state=random_state,  # the random seed
        verbose=False
    )

    result_score = 1-optimizer.fun

    # apply discretization method on result (TODO: put to separate function)
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