import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


class Heat1D:
    def __init__(self, omega, tau, nx, nt):
        self._omega = omega
        self._tau = tau
        self._nx = nx
        self._nt = nt
        self._hx = omega/float(nx)
        self._ht = tau/float(nt)



    def assemble_mat_op( self, kappa ):
        # assemble forward operator
        diags = np.zeros( (3, self._nx) )
        diags[0,:] = -1.0/self._hx**2
        diags[1,:] =  2.0/self._hx**2
        diags[2,:] = -1.0/self._hx**2
        K = kappa*sp.spdiags( diags, [-1,0,1], self._nx-1, self._nx-1 )
        I = sp.spdiags( np.ones(self._nx), 0, self._nx-1, self._nx-1 )

        # for efficiency map to csc_array
        M = sp.csc_array( I + self._ht*K )
        return M

    def solve_fwd( self, w, kappa ):
        # solve forward problem
        M = self.assemble_mat_op( kappa )
        u_old = w.copy()

        for i in np.arange( self._nt ):
            u = sp.linalg.spsolve( M, u_old )
            u_old[:] = u

        return u



    def get_x( self ):

        hx = self._hx
        omega = self._omega
        nx = self._nx
        # place nx-1 equispace point in the interior of [0,L] interval

        return np.linspace(0.+hx, omega - hx, nx-1)



    def assemble_fwdop( self, kappa ):
        # compute full forward operator by applying it to an identity vector
        F = np.zeros(( self._nx-1, self._nx-1 ))
        w_i = np.zeros( self._nx-1)

        for i in np.arange( self._nx-1 ):
            w_i[i] = 1.0
            F[:,i] = self.solve_fwd( w_i, kappa )
            w_i[i] = 0.0

        return F



    def solve_tikhonov( self, y_obs, F, alpha ):
        # compute the solution of the tikhonov regularized problem;
        # solve the optimality conditions using a direct method; notice
        # that this is approach does not work in general since assembling
        # F is computationally prohibitive
        H = np.dot( F.transpose(), F) + alpha*np.identity(F.shape[1])
        rhs = np.dot( F.transpose(), y_obs)

        return np.linalg.solve(H, rhs)



###########################################################
# This code is part of the python toolbox termed
#
# ChAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
