import numpy as np


class LinSolve:
    def __init__(self):
        self.debug = 0



    def _enable_debug(self):
        self.debug = 1



    def _run_cg( self, A, y, tol, maxit=0 ):

        imsg = " >> cg:"

        print(imsg, "executing")

        n = y.shape[0]
        if maxit == 0:
            print(imsg, "setting max number or iterations to", n)
            maxit = n

        # allow for function handle
        if callable( A ):
            apply = lambda x: A(x)
        else:
            apply = lambda x: np.matmul(A,x)

        # initial guess
        x = np.zeros( n )

        # compute residual
        r = apply( x ) - y
        p = -r.copy()

        rtol = tol*np.linalg.norm( r )
        print( imsg, "tol={:e}".format(tol), "{:e}".format(rtol) )

        if self.debug == 1:
            print( imsg, "{:>6}".format('iter'), "{:>15}".format('||r||') )

        for k in range( maxit ):
            ap = apply( p )

            rr = np.inner( r, r )
            alpha =  rr / np.inner( p, ap )

            x = x + alpha*p
            r = r + alpha*ap

            rnorm = np.linalg.norm( r )

            if self.debug == 1:
                print(imsg, "{:>6d}".format(k), "{:>15e}".format(rnorm))

            if rnorm < tol or rnorm < rtol:
                break
            else:
                beta = np.inner( r, r ) / rr
                p = -r + beta*p

        return x



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
