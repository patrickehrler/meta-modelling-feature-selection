# Feature Selection using Bayesian Optimization 
## Instructions
### Install Python dependencies
Required Python version: 3.6.9

    pip3 install -r requirements.txt

A custom version of scikit-optimize has to be installed.
Replace files in the folder "skopt" of your local scikit-optimize installation with the following (especially the files "space/space.py", "optimizer/base.py" and "optimizer/optimizer.py"): https://github.com/patrickehrler/scikit-optimize/tree/master/skopt

### Run experiment
#### Iteration experiment
    python3 iter_experiment.py
#### Comparison Experiment
    python3 comparison_experiment.py

Results are stored in the folder "results/".

## Repository Structure
- **jupyter-notebook/** Folder that consist the experiment of the proposal presentation and visualization for the final work.
- **results/** Folder where the results of the experiments are saved.
- **approaches.py** Consists of dictionaries where all bayesian and comparison approaches are listed (including all adjustments)
- **bayesian_algorithms.py** Implements the actual Bayesian optimization based on two different libraries.
- **comparison_algorithms.py** Implements different filter, wrapper and embedded feature selection techniques based on various libraries.
- **comparison_experiment.py** Runs all possible Bayesian and comparison approaches (including CV), then stores results in "results/comparison_bayesian_experiment/".
- **config.py** Configuration file where number of iterations, number of cross-validation splits, number of features and the used datasets can be set.
- **get_dataset_list_script.py** Script that filters the datasets on openml.org based on the criteria set. Stores result in "/results/datasets/".
- **iter_experiment.py** Runs all Bayesian approaches and saves score for each iteration. Output can be used to visualize convergence.
- **utils.py** Utility functions concerning scores, convertion of vectors and estimators.
