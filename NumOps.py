import numpy as np
import scipy as sp


class NumOps:
    def __init__(self):
        self.debug = 0

    def enable_debug(self):
        self.debug = 1

    def get_diffmat( self, n, dim ):
    #get_diffmat generate 1d difference matrix
        if dim == 1:
            # generate difference matrix
            e = np.ones( n )
            D = sp.sparse.spdiags([e,-e], [0,1], n, n )
        elif dim == 2:
            I1 = sp.sparse.eye( n[0] )
            I2 = sp.sparse.eye( n[1] )

            e = np.ones( n[0] )
            D1 = sp.sparse.spdiags([e,-e], [0,1], n[0], n[0] )

            e = np.ones( n[1] )
            D2 = sp.sparse.spdiags([e,-e], [0,1], n[1], n[1] )

            D1 = sp.sparse.kron(D1,I2)
            D2 = sp.sparse.kron(I1,D2)

            D = sp.sparse.bmat( [[D1], [D2]] )

        return D





###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
