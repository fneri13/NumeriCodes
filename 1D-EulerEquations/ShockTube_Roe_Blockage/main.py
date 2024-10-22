import matplotlib.pyplot as plt
import numpy as np
from NumericalCodes.shock_tube import ShockTube

"""
INPUT PARAMETERS FOR THE SHOCK-TUBE PROBLEM
"""
LENGTH = 1
NX = 100
TIME_MAX = 2  # to simulate two reflections
RHOL, RHOR = 1.0, 0.125
UL, UR = 0.0, 0.0
PL, PR = 1.0, 0.1
BLOCK = 0.9



""" 
Solution Driver
"""
x = np.linspace(0, LENGTH, NX)
dx = x[1]-x[0]
b = np.ones_like(x)
for i in range(NX):
    if NX/3<=i<=2*NX/3:
        s = x[i]-x[NX//3]
        L = x[2*NX//3]-x[NX//3]
        b[i] = 1+5*(s**2-s*L)

plt.figure()
plt.plot(x, b)

Smax = np.sqrt(1.4*np.max([PL, PR])/np.min([RHOL, RHOR]))+np.max([UL, UR])  # brutal approximation max eigenvalue
CFLmax = 0.9 # conservative CFL
dtMax = CFLmax* dx / Smax
nt = int(TIME_MAX/dtMax)
t = np.linspace(0, TIME_MAX, nt)

tube = ShockTube(x, t)
inCondDict = {'Density': np.array([RHOL, RHOR]), 'Velocity': np.array([UL, UR]), 'Pressure': np.array([PL, PR])}
tube.InstantiateSolutionArrays()
tube.InstantiateSolutionArraysConservatives()
tube.InitialConditionsLeftRight(inCondDict)
tube.SetBoundaryConditions('reflective', 0)
tube.ComputeBlockagedGeometry(b)
tube.SolveSystem(flux_method='Roe')
tube.SaveSolution(folder_name='solutions', file_name='SodsTest_tMax_%.2f' %TIME_MAX)
tube.ShowAnimation()




plt.show()