import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import Optimize



# evaluate objective function
def eval_objfun( A, x, y, alpha, flag="d2f" ):
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

    n = A.shape[0];
    # evaluate hessian
    d2f = np.matmul( AT, A ) + alpha*np.identity( n )

    return f,df,d2f;



n = 512; # problem dimension
A = np.random.rand( n, n )
x = np.random.rand( n )

# compute right hand side
y = np.matmul(A,x)

# initialize class
opt = Optimize();

# define function handle
fctn = lambda x, flag: eval_objfun( A, x, y, 0.1, flag )

# set objective function
opt._set_objfctn( fctn )

# perform derivative check
opt._deriv_check( x )




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
