import matplotlib.pyplot as plt
import matplot2tikz

from Heat1D import *
from Helper import *


def naive_solve( heat_obj, y_obs, kappa ):
    # compute naive solution of inverse problem
    M = heat_obj.assemble_mat_op( kappa )

    w_i = y_obs.copy()
    for i in np.arange(nt):
        w = M @ w_i
        w_i[:] = w

    return w


tau = 1.0 # time
omega = 1.0 # space (domain Omega = [0,omega])
nx = 100 # space
nt = 100 # time steps
# nx = 20 # space (noise free)
# nt = 100 # time steps (noise free)

kappa = 0.005 # coefficient

#sigma = 0.0 # stdev of noise
sigma = 1e-4 # stdev of noise

# init object
heat_obj = Heat1D( omega, tau, nx, nt )

# compute observation
y_obs, u_true, w_true = get_yobs( heat_obj, kappa, sigma, 2 )

# compute naive solution of inverse problem
w = naive_solve( heat_obj, y_obs, kappa )

# plot result
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(w_true, "k-", label = r'$w_\text{true}$')
plt.plot(w, "r-", label = r'$w$')
plt.legend()
plt.subplot(1,2,2)
plt.plot(u_true, "k-", label = r'$u(1)$')
plt.plot(y_obs, "rx", label = r'$y_\text{obs}$')
plt.legend()

#matplot2tikz.save('heat1D-navie-noisefree.tikz')
#matplot2tikz.save('heat1D-navie-noisy.tikz')

plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# ChAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
