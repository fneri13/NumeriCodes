import matplotlib.pyplot as plt
import numpy as np
import pickle
from NumericalCodes.riemann_problem import RiemannProblem
from NumericalCodes.shock_tube import ShockTube
import os


refFiles = ["../RiemannProblem_Analytical/solutions/Test%i.pik" %i for i in range(1,6)]
godFiles = ["solutions/Test%i.pik" %i for i in range(1,6)]
outFolder = 'pictures'
os.makedirs(outFolder, exist_ok=True)

for i in range(len(refFiles)):
    
    # reference file
    with open(refFiles[i], 'rb') as file:
        ref = pickle.load(file)
    
    # godunov file
    with open(godFiles[i], 'rb') as file:
        god = pickle.load(file)
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ref.x+0.5, ref.rho[:, -1], '-C0o', ms=2, label='Reference') # the analytical solution needs to be shifted by 0.5 in x
    ax[0, 0].plot(god.xNodes, god.solution['Density'][1:-1, -1], '-C1o', ms=2, mfc='none', label='Godunov')  # don't plot the halo nodes
    ax[0, 0].set_ylabel(r'Density')

    ax[0, 1].plot(ref.x+0.5, ref.u[:, -1], '-C0o', ms=2, label='Reference')
    ax[0, 1].plot(god.xNodes, god.solution['Velocity'][1:-1, -1], '-C1o', ms=2, mfc='none', label='Godunov')
    ax[0, 1].set_ylabel(r'Velocity')

    ax[1, 0].plot(ref.x+0.5, ref.p[:, -1], '-C0o', ms=2, label='Reference')
    ax[1, 0].plot(god.xNodes, god.solution['Pressure'][1:-1, -1], '-C1o', ms=2, mfc='none', label='Godunov')
    ax[1, 0].set_ylabel(r'Pressure')

    ax[1, 1].plot(ref.x+0.5, ref.e[:, -1], '-C0o', ms=2, label='Reference')
    ax[1, 1].plot(god.xNodes, god.solution['Energy'][1:-1, -1], '-C1o', ms=2, mfc='none', label='Godunov')
    ax[1, 1].set_ylabel(r'Energy')

    fig.suptitle('Test %i, Time %.3f' % (i+1, god.timeVec[-1]))

    for row in ax:
            for col in row:
                col.set_xlabel('x')
                col.grid(alpha=.3)
                col.legend()
    
    plt.savefig(outFolder + '/Test%i.pdf' % (i+1), bbox_inches='tight')



plt.show()