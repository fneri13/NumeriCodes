# pag 54 of the Toro book, to visualise how different waves advect for the lineatised gas dynamic equation
import numpy as np
import matplotlib.pyplot as plt

X_MIN = 0
X_MAX = 5
X_DS = (X_MAX - X_MIN) / 2
N_X = 1000
N_T = 100
T_MAX = 5
SS = 1  # advection speed

X = np.linspace(X_MIN, X_MAX, N_X)
T = np.linspace(0, T_MAX, N_T)


def density(xp):
    if xp <= X_DS:
        rho = 1
    else:
        rho = 1/2
    return rho


def velocity(xp):
    if xp <= X_DS:
        u = 0
    else:
        u = 0
    return u


rho_solution = np.zeros_like(X)
u_solution = np.zeros_like(X)

plt.figure()
for it in range(len(T)):
    plt.cla()
    for ix in range(len(X)):
        rho_solution[ix] = (1 / 2 / SS) * (SS * density(X[ix] + SS * T[it]) - density(X[ix]) * velocity(X[ix] + SS * T[it])) + (
                    1 / 2 / SS) * (SS * density(X[ix] - SS * T[it]) + density(X[ix]) * velocity(X[ix] - SS * T[it]))

        u_solution[ix] = -(1 / 2 / density(X[ix])) * (SS * density(X[ix] + SS * T[it]) - density(X[ix]) * velocity(X[ix] + SS * T[it])) + (
                    1 / 2 / density(X[ix])) * (SS * density(X[ix] - SS * T[it]) + density(X[ix]) * velocity(X[ix] - SS * T[it]))
    plt.plot(X, rho_solution, label=r'$\rho$')
    plt.plot(X, u_solution, label=r'$u$')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel('x')
    plt.pause(0.1)


plt.show()