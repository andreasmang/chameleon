import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")

from Data import *
from LinSolve import *


n = 128; # problem dimension

# initialize classes
dat = Data()
sol = LinSolve()

# problem setup
A = dat.get_spd_mat( n )
xtrue = np.random.rand( n )
y = np.matmul( A, xtrue )

# execute cg solver
sol.enable_debug()
xcg = sol.run_cg( A, y, 1e-12, 1000 )

# set initial guess
x0 = 0*xtrue

# plot results
z = np.linspace( 0, 1, n)
plt.plot( z, x0, marker="1", linestyle='', markersize=12)
plt.plot( z, xcg, marker="2", linestyle='', markersize=12)
plt.plot( z, xtrue )
plt.legend(['initial guess', 'cg solution', r'$x^\star$'], fontsize="20")
plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
