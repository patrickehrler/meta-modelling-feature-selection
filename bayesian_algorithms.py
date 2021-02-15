from numpy.lib.function_base import average
from sklearn.utils.validation import check_random_state
from skopt.learning.forest import ExtraTreesRegressor, RandomForestRegressor
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                                     ExpSineSquared, DotProduct,
                                                     ConstantKernel, HammingKernel)
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.optimizer import base_minimize
from skopt.space import Integer, Real
from skopt.utils import cook_estimator
from utils import get_score
import numpy as np
import config
from sklearn.model_selection import KFold


def skopt(data, target, n_features=None, kernel=None, learning_method="GP", discretization_method="round", estimator="linear_regression", metric="r2", acq_func="PI", n_calls=20, intermediate_results=False, cross_validation=0, n_random_starts=5, random_state=123, noise="gaussian"):
    """ Run Scikit-Optimize Implementation of Bayesian Optimization

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    data_test -- feature matrix for test data (can be None, then no score is returned)
    target_test -- target vector (can be None, then no score is returned)
    n_features -- number of features to be selected; if 'None' then best result is selected
    estimator -- estimator used to determine score
    discretization_method -- define method on how to work with search space
    acq_func -- aquisition function to be used
    n_calls -- number of iterations
    intermediate_results -- if True a set of result vectors of each iteration step will be returned
    n_random_starts -- 
    random_state -- 
    noise -- 

    """

    # define black box function
    def black_box_function(*args):
        # apply discretization method on value to be evaluated
        mask = discretize(args[0], discretization_method, n_features)
        
        if cross_validation != 0:
            kf = KFold(n_splits=cross_validation, shuffle=True).split(data)
            score_list = []
            for train_index, _ in kf:
                # get score from estimator
                score_list.append(1 - get_score(data.loc[train_index], target.loc[train_index], mask, estimator, metric))
            score = sum(score_list) / len(score_list)
        else:
            # get score from estimator
            score = 1 - get_score(data, target, mask, estimator, metric)

        return score

    # create feature space
    space = []
    if discretization_method == "binary":  # use binary (Integer) values
        for feature_name in data.columns:
            # use real values in range (0,1)
            space.append(Integer(0, 1, name=feature_name))
    elif discretization_method == "round" or discretization_method == "n_highest" or discretization_method == "probabilistic_round":
        for feature_name in data.columns:
            space.append(Real(0, 1, name=feature_name))
    else:
        raise ValueError(
                "Undefined discretizetion method.")

    # define base estimator
    base_estimator = None
    if kernel is not None:
        if learning_method == "GP":
            if kernel == "MATERN":
                base_estimator = GaussianProcessRegressor(1.0 * Matern())
            elif kernel == "HAMMING":
                base_estimator = GaussianProcessRegressor(
                    1.0 * HammingKernel())
            elif kernel == "RBF":
                # https://blogs.sas.com/content/iml/2018/09/26/radial-basis-functions-gaussian-kernels.html (basically squared distance)
                base_estimator = GaussianProcessRegressor(1.0 * RBF())
            else:
                raise ValueError("Invalid kernel.")
        else:
            raise ValueError(
                "Kernels can only be used with Gaussian Processes.")
    else:
        if learning_method == "RF":
            base_estimator = RandomForestRegressor()
        elif learning_method == "ET":
            base_estimator = ExtraTreesRegressor()
        elif learning_method == "GBRT":
            base_estimator = GradientBoostingQuantileRegressor()
        elif learning_method == "GP":
            # As implemented in gp_minimize (https://github.com/scikit-optimize/scikit-optimize/blob/de32b5f/skopt/optimizer/gp.py#L12)
            rng = check_random_state(random_state)
            base_estimator = cook_estimator(
                learning_method, space=space, random_state=rng.randint(
                    0, np.iinfo(np.int32).max),
                noise=noise)
            print(
                "Warning: No kernel defined for Gaussian Process. Standard kernel is used.")
        else:
            raise ValueError("Undefined learning method.")

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

    if discretization_method == "round" and n_features is not None:
        # to limit the number of selected features on "round" we use the n highest features after the last bayesian iteration step
        result_vector = discretize(optimizer.x, "n_highest", n_features)
    elif discretization_method == "binary" and n_features is not None:
        result_vector = discretize(optimizer.x, "n_highest", n_features) # select n first features
    else:
        result_vector = discretize(
            optimizer.x, discretization_method, n_features)

    if intermediate_results == True:
        result_vector_set = list(map(lambda x: discretize(x, discretization_method, n_features), optimizer.x_iters))
        result_fun_set = list(map(lambda x: 1-x, optimizer.func_vals))
        return result_vector, result_vector_set, result_fun_set
    else:
        return result_vector


def discretize(data, discretization_method, n_features=None):
    """ Apply discretization method on vector

    Keyword arguments:
    data -- vector to be discretized
    discretization_method -- method on how to make vector discrete
    n_features -- number of features to be selected for discretization_method="n_highest"

    """
    if discretization_method == "round":
        return list(map(lambda x: round(x), data))
    elif discretization_method == "probabilistic_round":
        # 0 has probability 1-x, 1 has probability x
        return list(map(lambda x: np.random.choice(a=[0, 1], p=[1-x, x]), data))
    elif discretization_method == "n_highest":
        if n_features is not None:
            vector = [0 for _ in range(len(data))]
            for _ in range(n_features):
                # detect maximum; in case of multiple occurrences the first is used
                max_index = np.argmax(data)
                # set mask value
                vector[max_index] = 1
                # remove input value
                data[max_index] = 0
            return vector
        else:
            raise ValueError("Undefined n_features parameter.")
    elif discretization_method == "binary":
        return data
    else:
        raise ValueError("Undefined discretization method.")
