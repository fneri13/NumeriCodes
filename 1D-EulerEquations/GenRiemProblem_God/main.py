import matplotlib.pyplot as plt
import numpy as np
import sys
from NumericalCodes.shock_tube import ShockTube


"""
INPUT PARAMETERS FOR THE SHOCK-TUBE PROBLEM
"""
LENGTH = 1
NX = 100
TIME_MAX = 2.0


""" 
Solution Driver
"""
x = np.linspace(0, LENGTH, NX)
dx = x[1]-x[0]
rho = np.ones_like(x)
u = x/np.max(x)
p = x/np.max(x)**2
approxSpeed = np.max(u) + np.sqrt(1.4*np.max(p)/np.min(rho))  # brutal approximation max eigenvalue
CFLmax = 0.5  # conservative CFL
dtMax = CFLmax* dx / approxSpeed
nt = int(TIME_MAX/dtMax)
t = np.linspace(0, TIME_MAX, nt)

# smooth initial conditions

inCond = {'Density': rho, 'Velocity': u, 'Pressure': p}


tube = ShockTube(x, t)
tube.InstantiateSolutionArrays()
tube.InstantiateSolutionArraysConservatives()
tube.InitialConditionsArrays(inCond)
tube.SetBoundaryConditions('reflective', 0)
tube.SolveSystem(flux_method='Godunov')
tube.SaveSolution(folder_name='solutions', file_name='test_tMax_%.1f' %TIME_MAX)

tube.ShowAnimation()




plt.show()