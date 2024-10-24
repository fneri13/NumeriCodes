import numpy as np
import matplotlib.pyplot as plt
from NumericalCodes.heat_equation_2D import HeatEquation2D

"""
INPUT PARAMETERS
"""
LX = 1
NX = 50
LY = 1
NY = 50
T_LEFT = 100
T_BOTTOM = 150
T_RIGHT = 200
T_TOP = 150



"""
SOLUTION DRIVER
"""
x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
prob = HeatEquation2D(xgrid, ygrid)
prob.SetBoundaryConditions('Dirichlet', T_LEFT, T_BOTTOM, T_RIGHT, T_TOP)
prob.SetInitialConditions('Average')
prob.InitializeLinearSystem()
# prob.SolveSystemDirect()

prob.SolveSystemIterative('jacobi')
prob.PlotSolution()

plt.show()