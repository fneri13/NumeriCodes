import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from NumericalCodes.riemann_problem import RiemannProblem
from NumericalCodes.shock_tube import ShockTube


filepath = 'solutions/test_tMax_2.0.pik'
with open(filepath, 'rb') as file:
    tube = pickle.load(file)

tube.ShowAnimation()
# tube.PlotSolution(iTime=tube.nTime-1)



plt.show()