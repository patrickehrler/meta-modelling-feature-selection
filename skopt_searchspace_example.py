from skopt.utils import normalize_dimensions

# Example of Integer to [0,1] transformation
# source: https://github.com/scikit-optimize/scikit-optimize/issues/580

# create normalized space [0,1]
space = normalize_dimensions([(9, 13)])
print(space.rvs(random_state=1))

# transform value into our [0,1] space
x= space.transform([[9]])
print(x)

# inverse transformation
print(space.inverse_transform(x))
# value is rounded and inversed
print(space.inverse_transform([[0.2]]))