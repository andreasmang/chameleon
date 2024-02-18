import numpy as np

def tanh_actnfctn( x, flag="d2f" ):
    # create vector of ones
    e = np.ones( x.shape )

    # evaluate function
    f = np.tanh( x )

    if flag == "f":
        return f

    # evaluate gradient
    df  = 1 - f**2

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = -2*f + 2*(f**3)

    return f, df, d2f



n = 28*28;
p = 10
m = 60000

# map column vector to matrix
X = np.random.rand( n, p )
Y = np.random.rand( m, n )

# apply matrix Y to X
Ypred = np.matmul( Y, X )

sYX, dsYX, d2sYX = tanh_actnfctn( Ypred )


print( "shape of tanh( Z )", sYX.shape )
print( "shape of derivative of tanh( Z )", dsYX.shape )
print( "shape of hessian of tanh( Z )", d2sYX.shape )
