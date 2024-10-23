import matplotlib.pyplot as plt
import numpy as np
from NumericalCodes.advection_equation import AdvectionEquation

"""
INPUT PARAMETERS
"""
LENGTH = 5
NX = 500
TIME_MAX = 2.5
CFL = 0.4
A_SPEED = 3


"""
SOLUTION DRIVER
"""
x = np.linspace(0, LENGTH, NX)
dx = x[1]-x[0]
dt = dx*CFL/A_SPEED
Nt = int(TIME_MAX//dt)
t = np.linspace(0, TIME_MAX, Nt)

adv = AdvectionEquation(x, t, CFL)
adv.InstantiateInitialCondition('saw-tooth', LENGTH/3)
adv.SetBoundaryConditions('periodic', 0)
adv.SolveSystem(method='FORCE')
adv.ShowAnimation()




plt.show()