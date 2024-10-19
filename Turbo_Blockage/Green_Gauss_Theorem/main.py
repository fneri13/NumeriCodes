import numpy as np
import matplotlib.pyplot as plt
from src.one_dimensional import OneDimensional
from Utils.styles import *
import os
import matplotlib.animation as animation


def blockage_function(x, zeta, bfunc, L_BLADE):
    N = len(x)
    block = np.ones(N)
    for i in range(N):
        if 0 <= zeta[i] <= L_BLADE:
            if bfunc == 'parabolic':
                block[i] = 1 + 15 * (zeta[i] ** 2 - L_BLADE * zeta[i])
            elif bfunc == 'constant':
                block[i] = 1
            elif bfunc == 'exponential':
                block[i] = 1 - (1 - 0.6) * np.exp(-10 * zeta[i] / L_BLADE)
            elif bfunc == 'harmonic':
                block[i] = 1 - 0.5 * np.sin(np.pi * zeta[i] / L_BLADE)
            else:
                raise ValueError("Blockage function not recognized")
    return block


def main():
    X_MIN = 0
    X_MAX = 1
    L_BLADE = (X_MAX - X_MIN) / 3
    N_POINTS = 1000
    # BLOCKAGE_FUNCS = ['parabolic', 'constant', 'exponential', 'harmonic']
    BLOCKAGE_FUNCS = ['parabolic']

    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    zeta = x - X_MIN - L_BLADE
    xprime = x - X_MIN
    phi = np.sin(np.pi*x/(X_MAX-X_MIN))

    for dist in BLOCKAGE_FUNCS:
        b = blockage_function(x, zeta, dist, L_BLADE)
        dx = x[1]-x[0]
        dphi_dx = np.zeros_like(x)
        for i in range(1, len(x)-1):
            deltaV = dx*b[i]*1
            dphi_dx[i] = 1/deltaV * (0.5*(phi[i+1]-phi[i])*1*(b[i+1]*b[i]) -
                                     0.5*(phi[i-1]-phi[i])*1*(b[i-1]*b[i]))

        fig, ax = plt.subplots(2, figsize=(6, 10))
        ax[0].plot(x, b, label=r'$b$')
        ax[0].plot(x, phi, label=r'$\phi$')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(x, np.pi/(X_MAX-X_MIN)*np.cos(np.pi*x/(X_MAX-X_MIN)), label=r'Analytical $\phi_x$')
        ax[1].plot(x[1:-1], dphi_dx[1:-1], label=r'Green-Gauss $\phi_x$')
        ax[1].set_xlabel(r'$x$')
        ax[1].grid(alpha=0.3)
        ax[1].legend()
        plt.savefig('sol.pdf', bbox_inches='tight')







main()
plt.show()




plt.show()
