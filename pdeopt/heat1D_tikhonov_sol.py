import numpy as np
import matplotlib.pyplot as plt

#import matplot2tikz

from Heat1D import *
from Helper import *

# setup the problem
tau = 1.0
omega = 1.0
kappa = 0.005

nx = 100
nt = 100

alpha = 1e-3 # regularization parameter
sigma = 1e-3 # noise perturbation

# init object
heat_obj = Heat1D( omega, tau, nx, nt )

# compute observation
y_obs, u_true, w_true = get_yobs( heat_obj, kappa, sigma, 2 )

F = heat_obj.assemble_fwdop( kappa )
w_alpha = heat_obj.solve_tikhonov( y_obs, F, alpha )

plt.plot(w_true, "-r", label = 'w_true')
plt.plot(w_alpha, "-k", label = 'w_beta')
plt.title("solution")
plt.legend()

#matplot2tikz.save('heat1D-tikhonov-sol.tikz')

plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# ChAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
