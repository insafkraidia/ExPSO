
import numpy as np

import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Particle:
    # Constructor that takes in the dimensionality of the particle
    def __init__(self, D):

      # Initialize Position to a zero-filled numpy array with shape (1, D)
        self.Position = np.zeros((1, D))

        # Initialize Cost to 0.0
        self.Cost = 0.0

        # Initialize Velocity to a zero-filled numpy array with shape (1, D)
        self.Velocity = np.zeros((1, D))

        # Initialize Best as a new instance of the Best class with dimensionality D
        self.Best = Best(D)

        # Initialize Minimum as a new instance of the Best class with dimensionality D
        self.Minimum = Best(D)

        # Initialize Worst as a new instance of the Best class with dimensionality D
        self.Worst = Best(D)


class Best:
    def __init__(self, D):
        # This line initializes the Position attribute of the instance with a 1 x D zero matrix.
        self.Position = np.zeros((1, D))
        # This line initializes the Cost attribute of the instance with a float 0.0 value.
        self.Cost = 0.0


class ExponentialParticleSwarmOptimizer:
    def __init__(self, ObjFunction, D, nPop=50, MaxIt=5000, lb=-100, ub=100, runs=10):
        self.ObjFunction = ObjFunction
        self.D = D
        self.nPop = nPop
        self.MaxIt = MaxIt
        self.lb = lb
        self.ub = ub
        self.runs = runs
        self.MeanIter = [[] for _ in range(runs)]
        self.RunBestCost = []
        self.FES = [0] * (runs * nPop)
        # Initialization
        self.empty_particle = Particle(D)
        self.empty_particle.Position = np.zeros((1, D))
        self.empty_particle.Cost = 0.0
        self.empty_particle.Velocity = np.zeros((1, D))
        self.empty_particle.Best = Best(D)
        self.empty_particle.Minimum = Best(D)
        self.empty_particle.Worst = Best(D)

        self.particle = [self.empty_particle] * nPop
        self.GlobalBest = Best(D)
        self.GlobalBest.Cost = float('inf')
        self.Worst = Best(D)
        self.Worst.Cost = float('-inf')

    def optimize(self):
        for k in range(self.runs):
            # Constriction Coefficients
            N1 = 10
            N2 = 10
            c1 = -1
            c2 = 2
            w = 0.9                      # Inertia Weight
            r = 0.9                      # Coefficient Damping Ratio 1
            a = 2                        # Personal Learning Coefficient
            b = 2                        # Global Learning Coefficient
            c = 2
            d = -1
            e = -1
            alpha = 1e-5
            VelMax = self.ub
            VelMin = -VelMax

            for i in range(self.nPop):
                # Initialize Position
                self.particle[i].Position = self.lb + \
                    (self.ub - self.lb) * np.random.rand(1, self.D)

                # Initialize Velocity
                self.particle[i].Velocity = VelMin + \
                    (VelMax - VelMin) * np.random.rand(1, self.D)

                # Evaluation
                self.particle[i].Cost = self.ObjFunction(
                    self.particle[i].Position)

                # Update Personal Best
                self.particle[i].Best.Position = self.particle[i].Position
                self.particle[i].Best.Cost = self.particle[i].Cost

                # Update Global Best
                if self.particle[i].Best.Cost < self.GlobalBest.Cost:
                    self.GlobalBest = self.particle[i].Best

                # Update Personal Worst
                self.particle[i].Minimum.Position = self.particle[i].Position
                self.particle[i].Minimum.Cost = self.particle[i].Cost

                # Update Global Worst
                if self.particle[i].Cost > self.Worst.Cost:
                    self.Worst = self.particle[i].Minimum
            import array
            #BestCost = np.zeros(MaxIt)
            BestCost = []
            #BestCost = array.array('i', [0], start=1)
            it = 0
            while it < self.MaxIt:

                it += 1

                for i in range(self.nPop):
                    if i <= N1:
                        self.particle[i].Velocity = w * self.particle[i].Velocity \
                            + a * np.exp(1 / (np.linalg.norm(self.particle[i].Best.Position - self.particle[i].Position)) + alpha) * np.random.rand(1, self.D) \
                            + b * np.random.rand() * (self.particle[i].Best.Position - self.particle[i].Position) \
                            + c * np.random.rand() * (self.GlobalBest.Position - self.particle[i].Position) \
                            + d * np.random.rand() * (self.particle[i].Minimum.Position - self.particle[i].Position) \
                            + e * np.random.rand() * (self.Worst.Position -
                                                      self.particle[i].Position)

                    elif N1 < i <= N2:
                        self.particle[i].Velocity = w * self.particle[i].Velocity \
                            + c1 * np.random.rand() * (self.particle[i].Best.Position - self.particle[i].Position) \
                            + c2 * np.random.rand() * (self.GlobalBest.Position -
                                                       self.particle[i].Position)

                    else:
                        self.particle[i].Velocity = w * self.particle[i].Velocity \
                            + a * np.exp(1 / (np.linalg.norm(self.particle[i].Best.Position - self.particle[i].Position)) + alpha) * np.random.rand(1, self.D) \
                            + b * np.random.rand() * (self.particle[i].Best.Position - self.particle[i].Position) \
                            + c * np.random.rand() * (self.GlobalBest.Position -
                                                      self.particle[i].Position)
                    # Apply Velocity Limits
                    self.particle[i].Velocity = np.maximum(
                        self.particle[i].Velocity, VelMin)
                    self.particle[i].Velocity = np.minimum(
                        self.particle[i].Velocity, VelMax)

                    # Update Position
                    self.particle[i].Position = self.particle[i].Position + \
                        self.particle[i].Velocity

                    # Velocity Mirror Effect
                    IsOutside = (self.particle[i].Position < self.lb) | (
                        self.particle[i].Position > self.ub)
                    self.particle[i].Velocity[IsOutside] = - \
                        self.particle[i].Velocity[IsOutside]

                    # Apply Position Limits
                    self.particle[i].Position = np.maximum(
                        self.particle[i].Position, self.lb)
                    self.particle[i].Position = np.minimum(
                        self.particle[i].Position, self.ub)

                    # Evaluation
                    self.particle[i].Cost = self.ObjFunction(
                        self.particle[i].Position)

                    # Update Personal Best
                    if self.particle[i].Cost < self.particle[i].Best.Cost:
                        self.particle[i].Best.Position = self.particle[i].Position
                        self.particle[i].Best.Cost = self.particle[i].Cost

                        # Update Global Best
                        if self.particle[i].Best.Cost < self.GlobalBest.Cost:
                            self.GlobalBest = self.particle[i].Best

                    # Update Personal Worst
                    if self.particle[i].Cost > self.particle[i].Minimum.Cost:
                        self.particle[i].Minimum.Position = self.particle[i].Position
                        self.particle[i].Minimum.Cost = self.particle[i].Cost

                        # Update Global Worst
                        if self.particle[i].Cost > self.Worst.Cost:
                            self.Worst = self.particle[i].Minimum

                if it > 10:
                    a = a * r
                    if it > 1000:
                        a = 0
                    if it > 30000:
                        a = b * r
                    VelMax = self.ub * self.ub
                    VelMin = -self.ub
                    w = r * ((1 - w) / (1 + w))

                # appends the current global best cost to the BestCost list.
                BestCost.append(self.GlobalBest.Cost)
                # appends the current global best cost to the MeanIter list for the current iteration k
                self.MeanIter[k].append(self.GlobalBest.Cost)
                # cost for the current function number func_num.

            self.RunBestCost.append(BestCost[-1])
            #print("run=", k, " Best Cost =", self.RunBestCost[-1])

            self.FES.append(it * self.nPop)

            for it in range(it, self.MaxIt):

                self.MeanIter[k].append(self.GlobalBest.Cost)

        # calculates various performance metrics
        ExPSO = np.mean(self.MeanIter, axis=0)
        MEAN = np.mean(self.RunBestCost)
        WorstSol = np.max(self.RunBestCost)
        BestSol = np.min(self.RunBestCost)
        STD = np.std(self.RunBestCost)
        Avg_FES = np.mean(self.FES)
        metrics = {"ExPSO": ExPSO,
                   "MEAN": MEAN,
                   "WorstSol": WorstSol,
                   "BestSol": BestSol,
                   "STD": STD,
                   "Avg_FES": Avg_FES,
                   }
        return {"GlobalBestCost": self.GlobalBest.Cost,
                "GlobalBestPosition": self.GlobalBest.Position,
                "Metrics": metrics}
