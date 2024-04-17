import numpy as np
import matplotlib.pyplot as plt


def eval_objfun( Un, Y ):
    # evaluate loss function
    return 0.5*np.linalg.norm( Un - Y, ord='fro' )



def eval_K( theta ):
    # evaluate weight matrix parameterized by theta
    K = np.zeros( (3, 3) )
    K[0,0] = -1.0*theta[0] - theta[1]
    K[0,1] = theta[0]
    K[0,2] = theta[1]

    K[1,0] = theta[1]
    K[1,1] = -1.0*theta[0] - theta[1]
    K[1,2] = theta[0]

    K[2,0] = theta[0]
    K[2,1] = theta[1]
    K[2,2] = -1.0*theta[0] - theta[1]

    return K


def eval_fwd_prop( U0, K, n ):
    # compute forward propagation using first order accurate time
    # integration method (resnet interpreted as ODE)
    h = 10.0 / float(n)
    U = U0
    for i in range(n):
        U = U + h * np.tanh( K @ U )

    return U

m = 512
t = np.linspace( 0.2, 2.0, m )
n_layer = 100


# select inital feature; drawn from normal distribution
U0 = np.random.randn( 3, 3 )

# generate "true parameter" and data
theta_true = np.ones( 2 )
K = eval_K( theta_true )


Y = eval_fwd_prop( U0, K, 200 )
objval = np.zeros( (m ,m) )
theta = np.zeros(2)

# compute optimization landscape
for i in range( m ):
    theta[0] = t[i]

    for j in range( m ):
        theta[1] = t[j]

        K = eval_K( theta )
        Un = eval_fwd_prop( U0, K, n_layer )

        objval[i,j] = eval_objfun( Un, Y )



# plot landscape
X, Y = np.meshgrid(t, t)
plt.contour( X, Y, objval )
plt.plot(theta_true[0], theta_true[1], 'ro')
plt.show( )
