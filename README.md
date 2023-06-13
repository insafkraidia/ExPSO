# Exponential Particle Swarm Optimization for Global Optimization (ExPSO)

<p align="center" style=""> <img src="https://raw.githubusercontent.com/insafkraidia/ExPSO/master/src/06.png"> </p>

The ExPSO package is a Python library that includes an algorithm designed to optimize machine and deep learning parameters/hyperparameters. This method involves dividing the swarm population into three subpopulations and employing a search strategy that incorporates an exponential function. By doing so, the particles are able to take significant leaps within the search space. Additionally, the algorithm dynamically adjusts the control of each particle's velocity range to strike a balance between exploration and exploitation during the search process. The leaping strategy is integrated into the velocity equation, and a new cognitive parameter that linearly decreases over time is introduced, along with a dynamic inertia weight strategy. To obtain further information, we recommend referring to the journal paper available at [Exponential Particle Swarm Optimization for Global Optimization (ExPSO)](https://ieeexplore.ieee.org/document/9837898/).

## Table of content

| Section                                | Description                                                                              |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| [Installation](#installation)          | Installing the dependencies and ExPSO                                                    |
| [Getting started](#requirements)       | Packages necessary to work with ExPSO                                                    |
| [Available parameters](#parameters)    | Modifiable parameters in API with their possible values                                  |
| [Usage](#usage)                        | Usage example data                                                                       |
| [Examples with public data](#examples) | Different examples for API                                                               |
| [Results](#results)                    | Comparative study between ExPSO and other variants of PSO for CNN,LSTM, XLNET,MLP models |
| [References](#reference)               | References to cite                                                                       |
| [License](#license)                    | Package license                                                                          |

## Flowchart of the proposed ExPSO

<p align="center" style="max-width: 100%;height: 900px;width: 600px;"> <img src="https://raw.githubusercontent.com/insafkraidia/ExPSO/master/src/01.png"> </p>

## Installation

ExPSO can be installed using [pip](https://pip.pypa.io/en/stable/), a tool
for installing Python packages. To do it, run the following command:

```
pip install ExPSO
```

## Requirements

ExPSO requires Python >= 3.6.1 or later to run. For other Python
dependencies, please check the `pyproject.toml` file included
on this repository.

Note that you should have also the following packages installed in your system:

- pytorch
- numpy
- math
- tensorflow
- keras
- scikit-learn

## Parameters

| Parameter name                    | Parameter description                                                                                                                                                                                                                                         | Possible values                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Objective function (ObjFunction)  | The ObjFunction, represents the primary target that the algorithm aims to enhance. It serves as the central objective of the algorithm and is defined by the user according to the specific problem they seek to address, with the intention of optimization. | Function with list of floats as inputs |
| Dimensions (D)                    | D corresponds to the quantity of variables or characteristics that exist within the objective function. Precise specification of the dimension is crucial for attaining precise optimization outcomes.                                                        | Integer                                |
| Number of particles (nPop)        | nPop pertains to the quantity of agents or particles utilized for exploring the solution space. Augmenting the number of particles can enhance the quality of solutions discovered, yet it also amplifies the computational burden.                           | Integer                                |
| Maximum iteration numbers (MaxIt) | MaxIt determine the upper limit for the algorithm's iterations, ensuring termination. Increasing the iteration count can enhance solution accuracy while minimizing the chances of encountering a suboptimal solution.                                        | Integer                                |
| Upper and lower bounds (ub,lb)    | These are the maximum and the lowest value in the search space                                                                                                                                                                                                | Float                                  |
| Number of runs (runs)             | runs pertains to how many times the algorithm will be executed using identical parameters.                                                                                                                                                                    | Integer                                |

## Usage

Here is a sample guide outlining the procedure for utilizing the ExPSO package:

1. Import the ExPSO class.

```
from ExPSO import ExPSOClass
```

2. Define the objective function.

```
def ObjFunction(x):
    #instructions.....
    return ....
```

3. Create an instance of the ExPSO class from the ExPSO package with specified parameters, including the objective function, dimensionality (D), population size (nPop), maximum iterations (MaxIt), lower bounds (lb), upper bounds (ub), and number of runs (runs).

```
expso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt, lb=lb, ub=ub, runs=runs)
```

4. Execute the optimization process by employing the optimize() function.

```
best_solution = expso.optimize()
```

Note: The result of this function is an object that can be accessed by the user, containing multiple items arranged in the following manner.

- GlobalBestCost refers to the highest achievable value attained by the ExPSO algorithm, which is either the optimal outcome or very close to it.
- GlobalBestPosition refers to the most favorable or nearly ideal outcome acquired through the ExPSO algorithm within the exploration range.
- MEAN: It denotes the mean or average of the optimal cost obtained from several iterations of optimization.
- WorstSol can be defined as the maximum value among the best costs discovered during the optimization procedure. This metric is utilized to assess the excellence of the achieved solutions and identify the solution with the poorest performance.
- BestSol: It denotes the minimum value of the optimal cost discovered throughout the process of optimization.
- STD, commonly known as the standard deviation, is frequently employed to evaluate the variety or convergence of the solutions acquired.
- Avg_FES (Average Function Evaluations) denotes the mean count of function evaluations executed throughout the optimization procedure.

  <a name="item1"></a>

## Examples

### Experiment 1. ExPSO with rosenbrock function

<p align="center" style=""> <img src="https://raw.githubusercontent.com/insafkraidia/ExPSO/master/src/10.png"> </p>

The following example demonstrates the optimization process of ExPSO using the rosenbrock function:

```
def ObjFunction(x):
    # rosenbrock function
    n = len(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
# create an instance of the  ExPSOClass with the specified parameters
expso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction,D=1000, nPop=30, MaxIt=30, lb=-30, ub=30, runs=30)
# optimize the function using ExPSO and retrieve the best solution
best_solution = expso.optimize()
# print the best solution found
print(f"Best solution found: {best_solution}")
```

Outputs:

```
Best solution found:
{'GlobalBestCost': 0.0,
 'GlobalBestPosition': array([[-2.32808064e+00,  3.72638986e+00,  1.18030275e+01,
        -2.80199098e+01,  2.59328519e+01, -1.49057827e+01,
        -2.77256194e+00,  4.47054248e+00,  2.44699480e+01,
        -2.68899252e+01, -9.64479003e+00,  1.09886986e+01,
        -3.16572820e+00, -1.23867646e+01, -2.06098389e+01,
         1.86093654e+00,  1.71640639e+01,  1.89470347e+01,
         .................................................]]),
'Metrics': {'ExPSO': array([0., 0., 0., ..., 0., 0., 0.]),
 'MEAN': 0.0, 'WorstSol': 0.0, 'BestSol': 0.0, 'STD': 0.0, 'Avg_FES': 0.033296337402885685}}
```

### Experiment 2. ExPSO with ackley function

<p align="center" style=""> <img src="https://raw.githubusercontent.com/insafkraidia/ExPSO/master/src/08.png"> </p>

The following example demonstrates the optimization process of ExPSO using the ackley function:

```
def ObjFunction(x):
    # ackley function version 2.22
    z = -20*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/x.shape[1])) \
        - np.exp(np.sum(np.cos(2*np.pi*x), axis=1) /
                 x.shape[1]) + 20 + np.exp(1)
    return z
# create an instance of the  ExPSOClass with the specified parameters
expso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=1000, nPop=30, MaxIt=30, lb=-5.12, ub=5.12, runs=30)
# optimize the function using ExPSO and retrieve the best solution
best_solution = expso.optimize()
# print the best solution found
print(f"Best solution found: {best_solution}")
```

### Experiment 3. ExPSO with CNN

The following example demonstrates the optimization process of ExPSO for the convolutional neural network (CNN):

```
def ObjFunction(particles):
    numberFilters = int(particles[0][0])  # FLOAT TO INT
    numberEpochs = int(particles[0][1])
    # CALL CNN FUNCTION cnn --> RETURN accuracy
    accuracy = cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                    epochs=numberEpochs, filters=numberFilters, kernel_size=kernel_size, stride=stride)

    # APPLY LOST FUNCTION --> THE MAIN OBJECTIVE IS TO MINIMIZE LOSS --> MAXIMIZE ACCURACY AND AT SAME TIME MINIMIZE THE NUMBER OF EPOCHS
    # AND FILTERS, TO REDUCE TIME AND COMPUTACIONAL POWER
    loss = 1.5 * ((1.0 - (1.0/numberFilters)) +
                    (1.0 - (1.0/numberEpochs))) + 2.0 * (1.0 - accuracy)
    return loss  # NEED TO RETURN THIS PYSWARMS NEED THIS

def main():
    nPop = 30
    runs = 20
    lb = 1
    ub = 500
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction,D=D, nPop=nPop, MaxIt=MaxIt, lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()
```

Note: The file "tests/ExPSOWithCNN.py" contains the entire code for the implementation of the ExPSO algorithm with a Convolutional Neural Network (CNN).

### Experiment 4. ExPSO with LSTM

The following example demonstrates the optimization process of ExPSO for the Long short-term memory (LSTM):

```
def ObjFunction(particle):
    neurons = int(particle[0][0])
    epochs = int(particle[0][1])
    # CALL LSTM_MODEL function
    accuracy = lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                    neurons=neurons, epochs=epochs)
    # APPLY COST FUNCTION --> THIS FUNCTION IS EQUALS TO CNN COST FUNCTION
    loss = 1.5 * ((1.0 - (1.0/neurons)) + (1.0 - (1.0/epochs))
                    ) + 2.0 * (1.0 - accuracy)
    return loss


def main():
    nPop = 30
    runs = 20
    lb = 1
    ub = 200
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction, D=D, nPop=nPop, MaxIt=MaxIt, lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()
```

Note: The file "tests/ExPSOWithLSTM.py" contains the entire code for the implementation of the ExPSO algorithm with Long short-term memory (LSTM).

### Experiment 5. ExPSO with XLNET

The following example demonstrates the optimization process of ExPSO for XLNET:

```
def ObjFunction(particles):
    oss = alexNet(particleDimensions=particles, x_train=x_train, x_test=x_test,
                       y_train=y_train, y_test=y_test)
    return loss

def main():
    nPop = 30
    runs = 10
    lb = 32
    ub = 160
    D = 5*30
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction,D=D, nPop=nPop, MaxIt=MaxIt, lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()
```

Note: The file "tests/ExPSOWithXlNET.py" contains the entire code for the implementation of the ExPSO algorithm with XLNET.

### Experiment 6. ExPSO with MLP

The following example demonstrates the optimization process of ExPSO for the multilayer perceptron (MLP):

```
def ObjFunction(particles):
    allLosses = mlp(particleDimensions=particles, x_train=x_train, x_test=x_test,
                        y_train=y_train, y_test=y_test)

    return allLosses

def main():

    nPop = 30
    runs = 10
    lb = 1
    ub = 500
    D = 2
    MaxIt = 100
    # create an instance of the ExPSOClass class with the specified parameters
    pso = ExPSOClass.ExponentialParticleSwarmOptimizer(ObjFunction,D=D, nPop=nPop, MaxIt=MaxIt, lb=lb, ub=ub, runs=runs)
    # optimize the function using ExPSO and retrieve the best solution
    cost, pos, _ = pso.optimize()
```

Note: The file "tests/ExPSOWithMLP.py" contains the entire code for the implementation of the ExPSO algorithm with multilayer perceptron (MLP).

## Results

In [Exponential Particle Swarm Optimization for Global Optimization (ExPSO)](https://ieeexplore.ieee.org/document/9837898/), several analysis methods and a comparative study are presented to demonstrate the performance of this technique. In the provided diagram, we have conducted a comparative analysis between our library and several Python libraries:

- Fuzzy Self-Tuning PSO (FST-PSO) https://pypi.org/project/fst-pso/.
- Pyswarms: a reference librarythat used pure PSO (Particle Swarm Optimization). https://github.com/ljvmiranda921/pyswarms .
- Quantum particle swarm optimization (QPSO) https://pypi.org/project/qpso/.
- FastPSO :Fast parallel Particle Swarm Optimization package (FastPSO) https://pypi.org/project/fastPSO/.
  The results show significant progress and effective improvements accomplished using ExPSO for different models including CNN, LSTM, XLNET, and MLP.

<p align="center" style=""> <img src="https://raw.githubusercontent.com/insafkraidia/ExPSO/master/src/88.png"> </p>

## Reference

If you use `ExPSO` in your research papers, please refer to it using following reference:

```
[Exponential Particle Swarm Optimization for Global Optimization (ExPSO)](https://ieeexplore.ieee.org/document/9837898/)

```

## License

`ExPSO` is released under the terms of the GNU General Public License (GPL).
