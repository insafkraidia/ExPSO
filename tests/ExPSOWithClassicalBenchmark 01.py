import numpy as np
from ExPSO import ExPSOClass
runs = 30
lb = -30
ub = 30
D = 1000
nPop = 30
MaxIt = 10


def ObjFunction(x):
    # rosenbrock_function
    n = len(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# create an instance of the ExPSOClass class with the specified parameters
pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt,
                                                   lb=lb, ub=ub, runs=runs)
# optimize the function using ExPSO and retrieve the best solution
best_solution = pso.optimize()
# print the best solution found
print(f"Best solution found: {best_solution}")
