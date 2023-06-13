import numpy as np
from ExPSO import ExPSOClass

# sets the lower and upper bounds, as well as the number of dimensions (D),based on the value of the func_num variable.
runs = 30
lb = -5.12
ub = 5.12
D = 1000
nPop = 30
MaxIt = 100


def ObjFunction(x):
    z = -20*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/x.shape[1])) \
        - np.exp(np.sum(np.cos(2*np.pi*x), axis=1) /
                 x.shape[1]) + 20 + np.exp(1)
    return z


# create an instance of the ExPSOClass class with the specified parameters
pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt,
                                                   lb=lb, ub=ub, runs=runs)
# optimize the function using ExPSO and retrieve the best solution
best_solution = pso.optimize()
# print the best solution found
print(f"Best solution found: {best_solution}")
