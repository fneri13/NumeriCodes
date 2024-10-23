import matplotlib.pyplot as plt
import numpy as np
from NumericalCodes.advection_equation import AdvectionEquation

"""
INPUT PARAMETERS
"""
LENGTH = 5
NX = 500
TIME_MAX = 5
CFL = 0.5
A_SPEED = 2


"""
SOLUTION DRIVER
"""
x = np.linspace(0, LENGTH, NX)
dx = x[1]-x[0]
dt = dx*CFL/A_SPEED
Nt = int(TIME_MAX//dt)
t = np.linspace(0, TIME_MAX, Nt)

adv = AdvectionEquation(x, t, CFL)
adv.InstantiateInitialCondition('square')
adv.PlotSolution(0)
adv.SetBoundaryConditions('periodic', 0)
adv.SolveSystem(method='Lax-Wendroff')
adv.ShowAnimation()



plt.show()