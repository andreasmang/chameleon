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


# evaluate objective function
def eval_fctn( K, x, y, alpha, flag="d2f" ):
    # compute residual
    r = np.matmul( K, x ) - y

    # evaluate objective function
    f = 0.5*np.inner( r, r ) + alpha*0.5*np.inner( x, x )

    if flag == "f":
        return f

    # evaluate gradient
    KT = K.transpose()
    df = np.matmul( KT, r ) + alpha*x

    if flag == "df":
        return f,df

    n = K.shape[0]
    # evaluate hessian
    d2f = np.matmul( KT, K ) + alpha*np.identity( n )

    return f, df, d2f




n = 512; # problem dimension
K = np.random.rand( n, n )
x = np.random.rand( n )

# compute right hand side
y = np.matmul( K, x )

# define function handle
fctn = lambda x, flag: eval_fctn( K, x, y, 0.1, flag )


# perform derivative check
deriv_check( fctn, x )
