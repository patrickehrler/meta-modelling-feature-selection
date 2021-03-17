from GPyOpt.methods import BayesianOptimization
from numpy.random import seed
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_random_state
from skopt.learning.forest import RandomForestRegressor
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, HammingKernel, DotProduct, ConstantKernel)
from skopt.optimizer import base_minimize
from skopt.plots import plot_convergence
from skopt.space import Integer, Real, Categorical
from skopt.utils import cook_estimator
import GPy
import numpy as np
import pandas as pd
import tempfile

from callback import NImprovementStopper
from utils import get_score


def skopt(data, target, n_features=None, kernel=None, learning_method="GP", discretization_method="round", estimator="linear_regression", metric="r2", acq_func="PI", n_calls=20, intermediate_results=False, penalty_weight=0, cross_validation=0, acq_optimizer="sampling", n_convergence=20, n_acq_points=10000, n_random_starts=5, random_state=123, noise="gaussian"):
    """ Run Scikit-Optimize Implementation of Bayesian Optimization (only works with Gaussian processes and Matern or RBF kernel)

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best result is selected
    kernel -- kernel type for Gaussian processes (unused otherwise)
    learning_method -- model used for Bayesian optimization (GP or RF)
    discretization_method -- define method on how to work with search space
    estimator -- estimator used to determine score
    acq_func -- aquisition function to be used
    n_calls -- number of iterations
    intermediate_results -- if True a set of result vectors of each iteration step will be returned
    penalty_weight -- weight of penalty (0 to disable penalty)
    cross_validation - number of folds to perform cross validation inside bayesian optimization on
    acq_optimizer -- strategy to sample points of aqcuisition function 
            (default: "sampling", use "n_sampling" to only consider points where n_features are selected)
    n_convergence -- stop optimization if for n_convergence iterations the optimum did not change (to disable set None)
    n_acq_points -- number of points of the acquisition function to evaluate in each iteration
    n_random_starts -- number of initial random evaluations
    random_state -- seed for randomizer
    noise -- 

    Return:
    if intermediate_results is True: returns a tuple of the result vector, a set of all intermediate vectors and a set of all intermediate function values
    otherwise: returns tuple of result vector, final training score and number of iterations (black box evaluations)

    """
    # define black box function
    def black_box_function(*args):
        # apply discretization method on value to be evaluated
        mask = discretize(args[0], discretization_method, n_features)
        # calculate penalty score
        penalty_score = abs(sum(mask)-n_features)/len(data.columns)
        if penalty_weight < 0:
            raise ValueError("Undefined penalty weight.")

        if cross_validation > 1:
            # perform cross validation
            kf = KFold(n_splits=cross_validation, shuffle=True).split(data)
            score_list = []
            for train_index, test_index in kf:
                # get score from estimator
                evaluation_score = 1-get_score(data.iloc[train_index], data.iloc[test_index],
                                               target.iloc[train_index], target.iloc[test_index], mask, estimator, metric)
                score_list.append((evaluation_score) +
                                  penalty_weight * penalty_score)
            # calculate average validation score
            score = sum(score_list) / len(score_list)
        elif cross_validation == 0 or cross_validation == 1:
            # no cross validation
            evaluation_score = 1 - \
                get_score(data, data, target, target, mask, estimator, metric)
            score = (evaluation_score) + penalty_weight * \
                penalty_score  # minimize scores
        else:
            raise ValueError("Undefined cross-validation value.")

        #print(score)
        #print(sum(mask))

        return score

    # create feature space
    space = []
    if discretization_method == "binary":
        # integer space in range [0,1]
        for feature_name in data.columns:
            space.append(Integer(0, 1, name=feature_name))
    elif discretization_method == "categorical":
        # space of two categories (0,1)
        for feature_name in data.columns:
            space.append(Categorical([0, 1], name=feature_name))
    elif discretization_method == "round" or discretization_method == "n_highest" or discretization_method == "probabilistic_round":
        # real-valued space in range [0,1]
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
                base_estimator = GaussianProcessRegressor(
                    1.0 * Matern())
            elif kernel == "HAMMING":
                base_estimator = GaussianProcessRegressor(
                    1.0 * HammingKernel())
            elif kernel == "RBF":
                # https://blogs.sas.com/content/iml/2018/09/26/radial-basis-functions-gaussian-kernels.html (basically squared distance)
                base_estimator = GaussianProcessRegressor(1.0 * RBF())
            elif kernel == "DOT":
                base_estimator = GaussianProcessRegressor(1.0 * DotProduct())
            elif kernel == "CONST":
                base_estimator = GaussianProcessRegressor(
                    1.0 * ConstantKernel())
            else:
                raise ValueError("Invalid kernel.")
        else:
            raise ValueError(
                "Kernels can only be used with Gaussian Processes.")
    else:
        if learning_method == "RF":
            base_estimator = RandomForestRegressor()
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
        acq_optimizer=acq_optimizer,   # strategy to evaluate acquisition function
        n_features=n_features,  # only used if acq_optimizer is "n_sampling"
        n_calls=n_calls,         # the number of evaluations of f
        n_initial_points=n_random_starts,  # the number of random initialization points
        random_state=random_state,  # the random seed
        callback=NImprovementStopper(n_iterations=n_convergence), # convergence criterion
        initial_point_generator="random",
        # kappa=10000000, # do more exploration (LCB)
        # xi=0.0000001, # do more exploration (PI, EI)
        n_points=n_acq_points,  # number of points evaluated of the acquisition function per iteration
        verbose=False
    )

    if (discretization_method == "round" or discretization_method == "probabilistic_round") and n_features is not None:
        # to limit the number of selected features on "round" or "probabilistic_round" we use the n highest features after the last bayesian iteration step
        result_vector = discretize(optimizer.x, "n_highest", n_features)
    else:
        result_vector = discretize(
            optimizer.x, discretization_method, n_features)

    if intermediate_results == True:
        # return set of all vectors and all function values
        result_vector_set = optimizer.x_iters
        result_fun_set = [1-x for x in optimizer.func_vals]
        return result_vector, result_vector_set, result_fun_set
    else:
        # only return final result vector and final function value
        return result_vector, 1 - optimizer.fun, len(optimizer.x_iters)


# Alternative implementation of Bayesian optimization feature selection
# Not used for the experiment
def gpyopt(data, target, n_features=None, kernel=None, learning_method="GP", discretization_method="round", estimator="svc_linear", metric="accuracy", acq_func="PI", n_calls=15, intermediate_results=False, penalty_weight=10, cross_validation=0, n_random_starts=5, random_state=123):
    """ Run Scikit-Optimize Implementation of Bayesian Optimization
    Alternative to scikit-optimize. Currently not used in the experiment.

    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best result is selected
    kernel -- kernel type for Gaussian processes (unused otherwise)
    learning_method -- model used for Bayesian optimization (GP or RF)
    estimator -- estimator used to determine score
    discretization_method -- define method on how to work with search space
    acq_func -- aquisition function to be used
    n_calls -- number of iterations
    intermediate_results -- if True a set of result vectors of each iteration step will be returned
    penalty_weight -- weight of penalty
    n_random_starts -- number of initial random evaluations
    random_state -- seed for randomizer
    noise -- 

    Return:
    if intermediate_results is True: returns a tuple of the result vector, a set of all intermediate vectors and a set of all intermediate function values
    otherwise: returns tuple of result vector and final training score

    """
    # define black box function
    def black_box_function(*args):
        # apply discretization method on value to be evaluated
        mask = discretize(args[0][0], discretization_method, n_features)
        penalty_score = abs(sum(mask)-n_features)/len(data.columns)
        if penalty_weight < 0:
            raise ValueError("Undefined penalty weight.")

        if cross_validation > 1:
            # perform cross validation
            kf = KFold(n_splits=cross_validation, shuffle=True).split(data)
            score_list = []
            for train_index, test_index in kf:
                # get score from estimator
                evaluation_score = get_score(data.iloc[train_index], data.iloc[test_index],
                                             target.iloc[train_index], target.iloc[test_index], mask, estimator, metric)
                score_list.append((evaluation_score) -
                                  penalty_weight * penalty_score)
            # calculate average validation score
            score = sum(score_list) / len(score_list)
        elif cross_validation == 0 or cross_validation == 1:
            # no cross validation
            evaluation_score = get_score(
                data, data, target, target, mask, estimator, metric)
            score = (evaluation_score) - penalty_weight * penalty_score
        else:
            raise ValueError("Undefined cross-validation value.")

        print(score)
        print(sum(mask))
        return score

    # cast acquisition function (to get the same interface as with skopt)
    if acq_func == "PI":
        acq_func = "MPI"

    # define search-space
    space = []
    if discretization_method == "binary":  # use binary (Integer) values
        for feature_name in data.columns:
            # use real values in range (0,1)
            space.append(
                {'name': feature_name, 'type': 'discrete', 'domain': (0, 1)})
    elif discretization_method == "categorical":
        for feature_name in data.columns:
            space.append({'name': feature_name, 'type': 'categorical',
                          'domain': (0, 1), 'dimensionality': 1})
    elif discretization_method == "round" or discretization_method == "n_highest" or discretization_method == "probabilistic_round":
        for feature_name in data.columns:
            space.append(
                {'name': feature_name, 'type': 'continuous', 'domain': (0, 1)})
    else:
        raise ValueError(
            "Undefined discretizetion method.")

    # define base estimator
    kernel_object = None
    if learning_method == "GP":
        if kernel is not None:
            if kernel == "MATERN":
                kernel_object = GPy.kern.Matern52(len(data.columns))
            elif kernel == "RBF":
                kernel_object = GPy.kern.RBF(len(data.columns))
            else:
                raise ValueError("Invalid kernel.")
        else:
            kernel_object = GPy.kern.RBF(len(data.columns))
            print("No kernel defined. RBF will be used.")
    else:
        raise ValueError("Undefined learning method.")

    domain = [{'name': x, 'type': 'continuous',
               'domain': (0, 1)} for x in data.columns]

    # initialize random state for reproducability
    if random_state is not None:
        seed(random_state)

    # initialize optimizer
    optimizer = BayesianOptimization(f=black_box_function,
                                     domain=domain,
                                     model_type=learning_method,
                                     acquisition_type=acq_func,
                                     kernel=kernel_object,
                                     maximize=True,
                                     initial_design_numdata=n_random_starts,
                                     random_state=123
                                     )
    optimizer.run_optimization(max_iter=n_calls, verbosity=True)

    if (discretization_method == "binary" or discretization_method == "round" or discretization_method == "probabilistic_round" or discretization_method == "categorical") and n_features is not None:
        # to limit the number of selected features on "round" we use the n highest features after the last bayesian iteration step
        result_vector = discretize(optimizer.x_opt, "n_highest", n_features)
    else:
        result_vector = discretize(
            optimizer.x_opt, discretization_method, n_features)

    if intermediate_results == True:
        # get evaluation values for each iteration
        file_buffer = tempfile.NamedTemporaryFile(delete=True)
        optimizer.save_evaluations(file_buffer.name)
        evaluations = pd.read_csv(file_buffer.name, delimiter="\t")

        # split in function values and vectors
        result_fun_set = [-x for x in evaluations.iloc[:, 1].values]
        result_vector_set = evaluations.iloc[:, 2:].values

        return result_vector, result_vector_set, result_fun_set
    else:
        return result_vector, optimizer.fx_opt


def discretize(data, discretization_method, n_features=None):
    """ Apply discretization method on vector

    Keyword arguments:
    data -- vector to be discretized
    discretization_method -- method on how to make vector discrete
    n_features -- number of features to be selected for discretization_method="n_highest"

    Return: discretized data set

    """
    if discretization_method == "round":
        return [round(x) for x in data]
    elif discretization_method == "probabilistic_round":
        # 0 has probability 1-x, 1 has probability x
        return [np.random.choice(a=[0, 1], p=[1-x, x]) for x in data]
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
    elif discretization_method == "binary" or discretization_method == "categorical":
        return data
    else:
        raise ValueError("Undefined discretization method.")
