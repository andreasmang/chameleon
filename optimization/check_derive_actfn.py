import numpy as np
import matplotlib.pyplot as plt
import sys

def deriv_check( fctn, x ):

    reps = 51
    print( reps*"-" )
    print("executing derivative check")
    print( reps*"-" )

    h = np.logspace( 0, -10, 10 ); # step size
    v = np.random.rand( x.shape[0] ); # random perturbation

    # evaluate objective function
    f, df, d2f = fctn( x, "d2f" );

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
    for j in range( m ):
        hh = h[j]*h[j];

        ft = fctn( x + h[j]*v, "f" ); # function value
        t0[j] = np.linalg.norm( f - ft ); # taylor poly 0
        t1[j] = np.linalg.norm( f + h[j]*dfv - ft ) # taylor poly 1
        t2[j] = np.linalg.norm( f + h[j]*dfv + 0.5*hh*vtd2fv - ft ); # taylor poly 2

        # display to user
        print("{:e}".format( h[j] ), "{:e}".format( t0[j] ), "{:e}".format( t1[j] ), "{:e}".format( t2[j] ) )

    print( reps*"-" )

    # plot errors
    plt.loglog( h, t0 )
    plt.loglog( h, t1 )
    plt.loglog( h, t2 )
    plt.legend(['t0', 't1', 't2'])
    plt.show()

    return


def sigmoid(x):
    """Numerically stable logistic sigmoid, elementwise."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


# evaluate objective function
def eval_fctn( W, x, y, flag="d2f" ):
    # forward pass
    z = W @ x       # (m,)
    s = sigmoid(z)  # sigma(z)
    r = s + y       # residual in the 2-norm

    # objective
    f = float(r @ r)

    if flag == "f":
        return f

    # derivatives of sigmoid
    sp = s * (1.0 - s)        # sigma'(z)
    s2 = sp * (1.0 - 2.0 * s) # sigma''(z)

    # Gradient: 2 * W^T ( sigma'(z) ⊙ r )
    df = 2.0 * (W.T @ (sp * r))

    if flag == "df":
        return f,df

    # Hessian: 2 * W^T diag( (sigma')^2 + sigma'' ⊙ r ) W
    weights = 2.0 * (sp * sp + s2 * r)  # (m,)
    d2f = W.T @ (weights[:, None] * W)

    return f, df, d2f


def hess_vec( x, W, b, v):
    # hessian-vector product d2f(x) @ v (more efficient than forming d2f explicitly)

    z = W @ x
    s = sigmoid(z)
    r = s + b
    sp = s * (1.0 - s)
    s2 = sp * (1.0 - 2.0 * s)
    weights = 2.0 * (sp * sp + s2 * r)  # (m,)

    Wv = W @ v                          # (m,)
    return W.T @ (weights * Wv)




n = 64; # problem dimension
K = np.random.rand( n, n )
x = np.random.rand( n )

# compute right hand side
y = sigmoid( np.matmul( K, x ) )

# define function handle
fctn = lambda x, flag: eval_fctn( K, x, y, flag )


# perform derivative check
deriv_check( fctn, x )




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
