from sklearn.utils.validation import check_random_state
from skopt.learning.forest import ExtraTreesRegressor, RandomForestRegressor
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                                     ExpSineSquared, DotProduct,
                                                     ConstantKernel, HammingKernel)
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.space import Integer, Real
from skopt.optimizer import base_minimize
from sklearn.linear_model import LinearRegression
from itertools import compress
import numpy as np
from skopt.utils import cook_estimator

def run_bayesian_algorithm(type="SKOPT", **kwargs):
    if type == "SKOPT":
        return __skopt(**kwargs)
    else:
        print("Error: Undefined Bayesian Algorithm")


def __skopt(data, target, n_features=None, kernel=None, learning_method="GP", discretization_method="round", estimator=LinearRegression(), acq_func="PI", n_calls=20, n_random_starts=5, random_state=123, noise="gaussian"):
    """ Run Scikit-Optimize Implementation of Bayesian Optimization
    
    Keyword arguments:
    data -- feature matrix
    target -- regression or classification targets
    n_features -- number of features to be selected; if 'None' then best result is selected
    estimator -- estimator used to determine score
    discretization_method -- define method on how to work with search space
    acq_func -- aquisition function to be used
    n_calls -- number of iterations
    n_random_starts -- 
    random_state -- 
    noise -- 

    """
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
    base_estimator=None
    if kernel is not None:
        if learning_method is not "GP": # gaussian processes
             # TODO: Can random forests, ... also use different kernels?
            print('Warning: Kernels can only be used with Gaussian Processes. GP will be used.')
        if kernel == "MATERN":
            base_estimator=GaussianProcessRegressor(1.0 * Matern())
        elif kernel == "HAMMING":
            base_estimator=GaussianProcessRegressor(1.0 * HammingKernel())
        elif kernel == "RBF":
            base_estimator=GaussianProcessRegressor(1.0 * RBF()) # https://blogs.sas.com/content/iml/2018/09/26/radial-basis-functions-gaussian-kernels.html (basically squared distance)
        else:
            print('Error: Invalid kernel. Matern Kernel is used.')
            base_estimator=GaussianProcessRegressor(1.0 * Matern())
    else:
        if learning_method is "RF":
            base_estimator=RandomForestRegressor()
        elif learning_method is "ET":
            base_estimator=ExtraTreesRegressor()
        elif learning_method is "GBRT":
            base_estimator=GradientBoostingQuantileRegressor()
        elif learning_method is "GP":
            # As implemented in gp_minimize (https://github.com/scikit-optimize/scikit-optimize/blob/de32b5f/skopt/optimizer/gp.py#L12)
            rng = check_random_state(random_state)
            base_estimator=cook_estimator(
                learning_method, space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
                noise=noise)
            print('Warning: No kernel defined for Gaussian Process. Standard kernel is used for Gaussian Process.') 
        else:
            print('Error: Undefined learning_method!') 
            
    
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
        verbose=True
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
    
    #print(optimizer.models)

    return result_score, result_vector