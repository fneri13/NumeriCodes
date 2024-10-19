import numpy as np
import matplotlib.pyplot as plt

u_im = 1.3
u_i = 1
u_ip = 2
u_ipp = 2.1

x = [1, 2, 3, 4]
xint = 2.5


def MUSCL_interpolation(f_im, f_i, f_ip, f_ipp, eps, kappa):
    fr = f_ip - eps / 4 * ((1 + kappa) * (f_ip - f_i) + (1 - kappa) * (f_ipp - f_ip))
    fl = f_i + eps / 4 * ((1 + kappa) * (f_ip - f_i) + (1 - kappa) * (f_i - f_im))
    return fr, fl


EPS = [0, 1, 1, 1, 1]
KAPPA = [0, -1, 0, 1/3, 1]
TITLES = ['1st order upwind:\n',
          '1st order one-sided interpolation:\n',
          '2nd order upwind biased linear interpolation:\n',
          '3 points interpolation formula:\n',
          'Central scheme:\n']

for i in range(len(EPS)):
    eps, kappa = EPS[i], KAPPA[i]
    ur, ul = MUSCL_interpolation(u_im, u_i, u_ip, u_ipp, eps, kappa)
    plt.figure()
    plt.scatter(x, [u_im, u_i, u_ip, u_ipp])
    plt.scatter(xint-0.1, ul, label='L')
    plt.scatter(xint+0.1, ur, label='R')
    plt.legend()
    plt.title(r'%s $\varepsilon=%.1f$; $\kappa=%.1f$' %(TITLES[i], eps, kappa))
plt.show()