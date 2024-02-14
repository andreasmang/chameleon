import numpy as np
import matplotlib.pyplot as plt

from LinSolve import LinSolve


class Optimize:
    def __init__(self):
        self.data = [ ]
        self.debug = 1
        self.df_rtol = 1e-6
        self.df_atol = 1e-12
        self.maxiter = 0
        self.objfctn = []
        self.method = []
        self.ndf0 = []
        self.sol = LinSolve()
        self.sol._enable_debug()


    def _set_objfctn( self, objfctn ):
        """
        _set_objfctn set objective function (function handle);
        objective function is assumed ot be only a function of x
        i.e., the decision variable
        """
        self.objfctn = objfctn



    def _set_maxiter( self, maxiter ):
        """
        _set_maxiter set max number of iterations
        """
        self.maxiter = maxiter



    def _get_sdir( self, x ):
        """
        _getsdir get search direction for line search optimization
        """
        # evaluate objective function

        if self.method == "gdsc":
            f,df = self.objfctn( x, "df" )
            s = -df
        elif self.method == "newton":
            f,df,d2f = self.objfctn( x, "d2f" )
            if callable(d2f):
                s = self.sol._run_cg( d2f, -df, 1e-1, 100 )
            else:
                s = np.linalg.solve(d2f, -df)

        return s;



    def _do_linesearch( self, x, s ):
        """
        _do_linesearch do line search
        """

        # evaluate objective function
        fx, dfx = self.objfctn( x, "df" );

        # initialize flag
        success = 0;

        # set max number of linesearch iterations
        maxit = 24;

        # set initial step size
        t = 1.0;
        c = 1e-4;
        descent = np.inner( dfx, s );

        for i in range(1,maxit):
            # evaluate objective function
            ftrial = self.objfctn( x + t*s, "f" )

            #if self.debug:
            #    print("{:e}".format(ftrial), "<", "{:e}".format(fx), "[ t =","{:e}".format(t),"]")

            # make sure that we have a descent direction
            if ( ftrial < fx + c*t*descent ):
                success = 1
                break

            # divide step size by 2
            t = t / 2

        if success:
            rval = t
        else:
            rval = 0.0

        return rval



    def _check_convergence( self, x, k, df ):
        converged = 0

        ndf = np.linalg.norm( df )

        if ( ndf <= self.df_rtol*self.ndf0  ):
            print(">> solver converged: {:e}".format(ndf), "<", "{:e}".format(self.df_rtol*self.ndf0))
            converged = 1

        if ( ndf <= self.df_atol ):
            print(">> solver converged: {:e}".format(ndf), "<", "{:e}".format(self.ndf0))
            converged = 1

        if ( k >= self.maxiter):
            print(">> maximum number of iterations (", self.maxiter, ") reached")
            converged = 1


        return converged



    def _get_newton_step( self, df, d2f ):
        """
        _get_newton_step do newton step
        """
        if callable(d2f):
            s = np.inner( v, d2f( v ) )
        else:
            s = np.matmul(np.linalg.inv(d2f), -df);

        # apply inverse of hessian to negative gradient

        return s;



    def _print_header( self, flag, reps ):

        print( reps*"-" )
        if flag == 'gdsc':
            print("executing gradient descent")
        elif flag == 'newton':
            print("executing newton's method")
        print( reps*"-" )
        print( "{:>6}".format('iter'), "{:>15}".format('||df||'), "{:>15}".format('||df||_rel'), "{:>15}".format('step') )
        print( reps*"-" )



    def _run( self, x, flag="gdsc" ):
        """
        _optimize run optimizer
        """

        # set optimization method
        self.method = flag;

        f,df = self.objfctn( x, "df" )
        self.ndf0 = np.linalg.norm( df )

        reps = 55
        self._print_header( flag, reps );

        k = 0
        converged = self._check_convergence( x, k, df )

        # run optimizer
        while not converged:
            # compute search direction
            s = self._get_sdir( x )

            if np.inner( s, df ) > 0.0:
                print("not a descent direction")
                break

            # do line search
            t = self._do_linesearch( x, s )

            if t == 0.0:
                print("line search failed")
                return x
            else:
                x = x + t*s

            # check for convergence
            f,df = self.objfctn( x, "df" )

            ndf = np.linalg.norm( df )
            print("{:>6d}".format(k), "{:>15e}".format(ndf), "{:>15e}".format(ndf/self.ndf0), "{:>15e}".format(t))

            converged = self._check_convergence( x, k, df )

            k = k + 1

        print( reps*"-" )

        return x


    def _cvx_check( self, x, bound, n = 1000 ):
        """
        _cvx_check basic check if function is convex
        """
        # draw random direction in R^n
        v = np.random.rand( x.shape[0] );
        v = v / np.linalg.norm( v ) # normalize

        # define 1d function along which we plot function
        t = np.linspace( bound[0], bound[1], n ); # step size

        # store g(t)
        g = np.zeros( n )
        for j in range( n ):
            y = x + t[j]*v
            g[j] = self.objfctn( y, "f" )

        return g



    def _deriv_check( self, x ):

        reps = 51
        print( reps*"-" )
        print("executing derivative check")
        print( reps*"-" )

        h = np.logspace( 0, -10, 10 ); # step size
        v = np.random.rand( x.shape[0] ); # random perturbation

        # evaluate objective function
        f, df, d2f = self.objfctn( x, "d2f" );

        # compute linear term
        dfv = np.inner( df, v )

        # compute quadratic term
        if callable(d2f):
            vtd2fv = np.inner( v, d2f( v ) )
        else:
            vtd2fv = np.inner( v, np.matmul( d2f, v ) )

        # allocate history
        m = h.shape[0]
        t0 = np.zeros( m )
        t1 = np.zeros( m )
        t2 = np.zeros( m )

        print( "{:>1}".format('h'), "{:>13}".format('t1'), "{:>12}".format('t2'), "{:>12}".format('t3') )
        print( reps*"-" )

        # do derivative check
        for j in range( 0, m ):
            hh = h[j]*h[j];

            ft = self.objfctn( x + h[j]*v, "f" ); # function value
            t0[j] = np.linalg.norm( f - ft ); # taylor poly 0
            t1[j] = np.linalg.norm( f + h[j]*dfv - ft ) # taylor poly 1
            t2[j] = np.linalg.norm( f + h[j]*dfv + 0.5*hh*vtd2fv - ft ); # taylor poly 2

            # display to user
            print("{:e}".format(h[j]), "{:e}".format(t0[j]), "{:e}".format(t1[j]), "{:e}".format(t2[j]))

        print( reps*"-" )

        # plot errors
        if self.debug:
            plt.loglog( h, t0 )
            plt.loglog( h, t1 )
            plt.loglog( h, t2 )
            plt.legend(['t0', 't1', 't2'])
            plt.show()

        return




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
