import numpy as np
import matplotlib.pyplot as plt
from src.one_dimensional import OneDimensional
from Utils.styles import *
import os

X_MIN = 0
X_MAX = 1
L_BLADE = (X_MAX-X_MIN)/3
N_POINTS = 100
u_0 = 1
BLOCKAGE_FUNCS = ['parabolic', 'constant', 'exponential', 'harmonic']


def blockage_function(N, bfunc):
    block = np.ones(N)
    for i in range(N):
        if 0 <= zeta[i] <= L_BLADE:
            if bfunc == 'parabolic':
                block[i] = 1 + 10 * (zeta[i] ** 2 - L_BLADE * zeta[i])
            elif bfunc == 'constant':
                block[i] = 0.7
            elif bfunc == 'exponential':
                block[i] = 1 - (1 - 0.6) * np.exp(-10 * zeta[i] / L_BLADE)
            elif bfunc == 'harmonic':
                block[i] = 1 - 0.5*np.sin(np.pi * zeta[i]/L_BLADE)
            else:
                raise ValueError("Blockage function not recognized")
    return block


x = np.linspace(X_MIN, X_MAX, N_POINTS)
zeta = x-((X_MAX-X_MIN)/2-L_BLADE/2)
rho = np.ones_like(x)
A = np.ones_like(x)

for dist in BLOCKAGE_FUNCS:
    b = blockage_function(N_POINTS, dist)
    domain = OneDimensional(N_POINTS)
    domain.addCoordinates(x)
    domain.addDensity(rho)
    domain.addBlockage(b)
    domain.addArea(A)


    K = u_0*rho[0]*A[0]*b[0]
    u_ref = K/(rho*A*b)
    domain.setBC(u_ref[0], u_ref[-1])

    u_original = domain.solveContinuity(method='original')
    u_separated = domain.solveContinuity(method='separated')
    u_fv_consistent = domain.solveContinuity(method='FVM')

    folder = 'pics_block_' + dist
    os.makedirs(folder, exist_ok=True)

    domain.plotData(folder)

    plt.figure()
    plt.plot(x, u_ref, label='reference')
    plt.plot(domain.x, u_original, 'o', mfc='none', label='FD-C')
    plt.plot(domain.x, u_separated, '^', mfc='none', label='FD-NC')
    plt.plot(domain.x, u_fv_consistent, 's', mfc='none', label='FV-C')
    plt.xlabel('x')
    plt.ylabel('Velocity')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(folder + '/velocity.pdf', bbox_inches='tight')


    plt.figure()
    plt.plot(x, u_ref*rho*A*b, label='reference')
    plt.plot(domain.x, u_original*rho*A*b, 'o', mfc='none', label='FD-C')
    plt.plot(domain.x, u_separated*rho*A*b, '^', mfc='none', label='FD-NC')
    plt.plot(domain.x, u_fv_consistent * rho * A * b, '^', mfc='none', label='FV-C')
    plt.xlabel('x')
    plt.ylabel('Mass Flow')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(folder + '/massflow.pdf', bbox_inches='tight')












plt.show()