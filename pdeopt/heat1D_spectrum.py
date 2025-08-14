import matplotlib.pyplot as plt
import matplot2tikz
from Helper import *
from Heat1D import *

tau = 1.0
omega = 1.0
kappa = 0.005

nx = 100
nt = 100

# init object
heat_obj = Heat1D( omega, tau, nx, nt )

# compute spectrum of continuous operator
i = np.arange(1,50)
lambdas = np.exp(-kappa*tau*np.power(np.pi/omega*i,2))


# compute eigenvector and eigenvalues of the discretized forward operator
lmbda, U = compute_eigdec( heat_obj, kappa )

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)

plt.semilogy(i, lambdas, 'or')
plt.xlabel('i')
plt.ylabel(r'$\lambda_i$')
plt.title(r'eigenvalues of continuous $f$')

plt.subplot(1,2,2)
plt.semilogy(lmbda, 'ro')
plt.title(r'eigenvalues of discrete $f$')

#matplot2tikz.save('heat1D-eigenvalues.tikz')

plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# ChAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
