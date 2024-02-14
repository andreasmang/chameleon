import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")

from LineSearchOpt import Optimize
from Data import Data



# evaluate objective function
def eval_objfctn( A, x, y, alpha, flag="d2f" ):
    # compute residual
    r = np.matmul( A, x ) - y

    # evaluate objective function
    f = 0.5*np.inner( r, r ) + alpha*0.5*np.inner( x, x )

    if flag == "f":
        return f

    # evaluate gradient
    AT = A.transpose()
    df = np.matmul( AT, r ) + alpha*x

    if flag == "df":
        return f,df

    n = A.shape[0]
    # evaluate hessian
    d2f = np.matmul( AT, A ) + alpha*np.identity( n )

    return f, df, d2f



n = 128; # problem dimension

# initialize classes
opt = Optimize()
dat = Data()

A = dat._get_spd_mat( n )

xtrue = np.random.rand( n )

# compute right hand side
y = np.matmul( A, xtrue )


# define function handle
fctn = lambda x, flag: eval_objfctn( A, x, y, 1e-6, flag )

# initial guess
x = np.zeros( n )

# set parameters
opt._set_objfctn( fctn )
opt._set_maxiter( 100 )

# execture solver (gsc)
xgd = opt._run( x, "gdsc" )

# execture solver (newton)
xnt = opt._run( x, "newton" )


z = np.linspace( 0, 1, n)
plt.plot( z, xgd, marker="1", linestyle='', markersize=12)
plt.plot( z, xnt, marker="2", linestyle='', markersize=12)
plt.plot( z, xtrue )
plt.legend(['gradient descent', 'newton', r'$x^\star$'], fontsize="20")
plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
