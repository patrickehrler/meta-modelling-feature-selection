from skopt.callbacks import EarlyStopper

class nIterationsStopper(EarlyStopper):
    """Stop the optimization if the optimum did not change for the last n_iterations.
    """
    def __init__(self, n_iterations=10):
        super(EarlyStopper, self).__init__()
        self.n_iterations = n_iterations

    def _criterion(self, result):
        total_iterations = len(result.func_vals)

        if total_iterations > self.n_iterations:
            start = total_iterations-self.n_iterations-1
            stop = total_iterations
            print(result.fun)
            for value in result.func_vals[start:stop]:
                if value <= result.fun:
                    # current optimum was found during last n_iterations
                    return None
            # optimium was not found during last n_iterations
            return True
        return None