import numpy as np
import matplotlib.pyplot as plt


def blockage_function(x, zeta, bfunc, L_BLADE):
    N = len(x)
    block = np.ones(N)
    for i in range(N):
        if 0 <= zeta[i] <= L_BLADE:
            if bfunc == 'parabolic':
                block[i] = 1 + 10 * (zeta[i] ** 2 - L_BLADE * zeta[i])
            elif bfunc == 'constant':
                block[i] = 0.75
            elif bfunc == 'exponential':
                block[i] = 1 - (1 - 0.6) * np.exp(-10 * zeta[i] / L_BLADE)
            elif bfunc == 'harmonic':
                block[i] = 1 - 0.5 * np.sin(np.pi * zeta[i] / L_BLADE)
            else:
                raise ValueError("Blockage function not recognized")
    return block


def advected_wave(x, L_BLADE):
    phi = np.zeros_like(x)
    npoints = len(x)
    for i in range(npoints):
        if x[i] < L_BLADE / 2:
            phi[i] = np.sin(2 * np.pi * x[i] / L_BLADE)
    return phi


def laxWendroff(y, c):
    ynew = np.zeros(len(y))
    yold = np.zeros(len(y) + 2)
    yold[1:-1] = y
    yold[0] = y[-2]
    yold[-1] = y[1]
    for i in range(0, len(ynew)):
        j = i + 1
        ynew[i] = 0.5 * c * (1 + c) * yold[j - 1] + (1 - c ** 2) * yold[j] - 0.5 * c * (1 - c) * yold[j + 1]
    return ynew


def laxWendroffNonCons(y, c, b, bGrad, omega):

    ynew = laxWendroff(y, c)
    ynew += (- c * y * bGrad / b)*omega
    return ynew


def laxWendroffFV(y, a, dt, dx, b):
    ynew = np.zeros(len(y))
    yold = np.zeros(len(y) + 2)
    yold[1:-1] = y
    yold[0] = y[-2]
    yold[-1] = y[1]
    bext = np.zeros(len(b) + 2)
    bext[1:-1] = b
    bext[0] = b[-2]
    bext[-1] = b[1]


    def flux_left(ul, um, ur, a, dt, dx):
        f = 0.5*a*(ul+um) - 0.5*dt/dx*a**2*(um-ul)
        return f

    def flux_right(ul, um, ur, a, dt, dx):
        f = 0.5*a*(um+ur) - 0.5*dt/dx*a**2*(ur-um)
        return f

    for i in range(0, len(ynew)):
        j = i + 1
        fleft = flux_left(yold[j-1], yold[j], yold[j+1], a, dt, dx)
        fright = flux_right(yold[j-1], yold[j], yold[j+1], a, dt, dx)
        bleft = (bext[j]+bext[j-1])/2
        bright = (bext[j]+bext[j+1])/2


        ynew[i] = yold[j] + (bleft*fleft - fright*bright)*dt/dx/b[i]
    return ynew



def main():
    X_MIN = 0
    X_MAX = 1
    L_BLADE = (X_MAX - X_MIN) / 3
    N_POINTS = 250
    T_MAX = 1
    N_TIME = 500
    # BLOCKAGE_FUNCS = ['parabolic', 'constant', 'exponential', 'harmonic']
    BLOCKAGE_FUNCS = ['parabolic']

    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    zeta = x - X_MIN - L_BLADE
    xprime = x - X_MIN
    phi = advected_wave(xprime, L_BLADE)
    time_vec = np.linspace(0, T_MAX, N_TIME)

    for dist in BLOCKAGE_FUNCS:
        b = blockage_function(x, zeta, dist, L_BLADE)
        bgrad = np.gradient(b, x)
        sol_old = b * phi
        sol_cons_old = phi
        sol_ncons_old = phi

        # plt.figure()
        # plt.plot(x, sol_cons_old, label='u')
        # plt.plot(x, b, label='b')
        # plt.legend()
        # plt.xlabel(r'$x$')
        # plt.grid(alpha=0.25)
        # plt.legend()

        fig, ax = plt.subplots(2, 1)
        for iTime in range(len(time_vec)):
            cfl = 1
            dt = time_vec[1]-time_vec[0]
            dx = x[1]-x[0]
            a = cfl*dx/dt


            # reference solution of the problem
            sol_new = laxWendroff(sol_old, cfl)

            # non-consistent formulation of the problem
            omega = 0.02  # under-relaxation
            sol_ncons_new = laxWendroffNonCons(sol_ncons_old, cfl, b, bgrad, omega)

            # finite volume formulation
            sol_cons_new = laxWendroffFV(sol_cons_old, a, dt, dx, b)

            ax[0].cla()
            # plt.plot(x, sol_new, '--', label=r'$ub$')
            ax[0].plot(x, sol_new / b, label=r'$u_{REF}$')
            # ax[0].plot(x, sol_ncons_new, label=r'$u_{NC}$')
            ax[0].plot(x, sol_cons_new, label=r'$u_{FV}$')
            ax[0].plot(np.ones_like(x)*L_BLADE, np.linspace(-0.1, 1.5, len(x)), '--k', lw=0.75)
            ax[0].plot(np.ones_like(x) * 2*L_BLADE, np.linspace(-0.1, 1.5, len(x)), '--k', lw=0.75)
            # ax[0].set_ylim([-0.1, 1.5])
            ax[0].legend()
            ax[0].set_ylabel(r'$u$')
            ax[0].grid(alpha=0.25)

            ax[1].cla()
            # ax[1].plot(x, (sol_ncons_new-sol_new/b)*100, label=r'$\varepsilon_{NC}$')
            ax[1].plot(x, (sol_cons_new-sol_new/b)/np.max(sol_new/b)*100, 'r', lw=0.75, label=r'$\varepsilon_{FV}$')
            ax[1].legend()
            ax[1].set_xlabel(r'$x$')
            ax[1].set_ylabel(r'$\varepsilon \ \rm{[\%]}$')
            ax[1].grid(alpha=0.25)
            plt.pause(0.00001)

            # sol_ncons_old = sol_ncons_new
            sol_old = sol_new
            sol_cons_old = sol_cons_new
            sol_ncons_old = sol_ncons_new





main()
plt.show()




plt.show()
