import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from src.shock_tube import ShockTube


"""
INPUT PARAMETERS FOR THE SHOCK-TUBE PROBLEM
"""
LENGTH = 1
NX = 100
TIME_MAX = 2.0
RHOL, RHOR = 1.0, 0.125
UL, UR = 0.0, 0.0
PL, PR = 1.0, 0.1




""" 
Solution Driver
"""
x = np.linspace(0, LENGTH, NX)
dx = x[1]-x[0]
approxSpeed = np.sqrt(1.4*np.max([PL, PR])/np.min([RHOL, RHOR]))  # brutal approximation max eigenvalue
CFLmax = 1  # conservative CFL
dtMax = CFLmax* dx / approxSpeed
nt = int(TIME_MAX/dtMax)
t = np.linspace(0, TIME_MAX, nt)

tube = ShockTube(x, t)
inCondDict = {'Density': np.array([RHOL, RHOR]), 'Velocity': np.array([UL, UR]), 'Pressure': np.array([PL, PR])}
tube.InstantiateSolutionArrays()
tube.InstantiateSolutionArraysConservatives()
tube.InitialConditionsLeftRight(inCondDict)
tube.SetBoundaryConditions('transparent', 0)

tube.SolveSystem(flux_method='Godunov')
tube.SaveSolution(folder_name='solutions', file_name='SodsTest_transparent_tMax_%.1f' %TIME_MAX)

# tube.ShowAnimation()




plt.show()