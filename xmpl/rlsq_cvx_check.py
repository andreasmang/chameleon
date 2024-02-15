import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from LineSearchOpt import *



# evaluate objective function
def eval_objfun( A, x, y, alpha, flag ):
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
xtrue = np.random.rand( n )

# compute right hand side
y = np.matmul( A, xtrue )

# initialize class
opt = Optimize();


# define function handle
fctn = lambda x, flag: eval_objfun( A, x, y, 0.03, flag )
opt.set_objfctn( fctn )

bound = np.zeros(2)
b = 5

bound[0] = -b # lower bound
bound[1] =  b # upper bound
m = 100 # number of samples

# number of random perturbations
ntrials = 10

g = np.zeros([m,ntrials])

# draw random perturbations
for i in range(ntrials):
    # draw a random point
    x = np.random.rand( n )
    # compute 1d function along line: g(t) = f( x + t v )
    g[:,i] = opt.cvx_check( x, bound, m )


# plot
t = np.linspace( bound[0], bound[1], m )
plt.plot( t, g )
plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
