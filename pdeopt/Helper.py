import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz
from Heat1D import *


def compute_eigdec( heat_obj, kappa ):
    # compute F as a dense matrix
    nx = heat_obj._nx - 1

    F = np.zeros((nx,nx))
    w_i = np.zeros(nx)

    for i in np.arange(nx):
        w_i[i] = 1.0
        F[:,i] = heat_obj.solve_fwd( w_i, kappa )
        w_i[i] = 0.0

    # solve the eigenvalue problem
    lmbda, U = np.linalg.eigh(F)

    # sort eigenpairs in decreasing order
    lmbda[:] = lmbda[::-1]
    lmbda[lmbda < 0.] = 0.0
    U[:] = U[:,::-1]

    return lmbda, U


def get_w( x, flag ):

    if flag == 1:
        w_true = np.maximum( np.zeros_like(x), 1. - np.abs(1. - 4.*x));
        w_true = w_true + 100.*np.power(x,10)*np.power(1.-x,2)
    elif flag == 2:
        w_true = 0.5 - np.abs(x-0.5)

    return w_true


def get_yobs( heat_obj, kappa, sigma, flag ):

    # compute the data y_obs by solving the forward model
    x = heat_obj.get_x()

    # get true initial condition
    w_true = get_w(x, flag)

    # solve forward problem
    u_true = heat_obj.solve_fwd( w_true, kappa )

    # compute noise perturbation
    noise = np.random.randn(u_true.shape[0])

    y_obs = u_true + sigma*noise

    return y_obs, u_true, w_true



###########################################################
# This code is part of the python toolbox termed
#
# ChAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
