import numpy as np
import matplotlib.pyplot as plt

def get_kernel(n, tau=0.03):
    """
    function to get discrete convolution operator/kernel matrix.

    parameters:
        n (int): number of points
        tau (float): bandwidth of kernel (default 0.03)

    returns:
        K (ndarray): kernel matrix of shape (n, n)
        kx (ndarray): 1D kernel vector (optional, returned if requested)
    """
    # spatial step size (domain is [0,1])
    h = 1.0 / n
    h2 = h * h

    # compute all-to-all distance matrix
    y = np.arange(0.5, n, 1.0)  # points at cell centers
    y = y[:, np.newaxis] - y[np.newaxis, :]  # y[i,j] = (i+0.5) - (j+0.5)

    # constants for kernel
    c = 1.0 / (np.sqrt(2 * np.pi) * tau)
    d = h2 / (2 * tau**2)

    # kernel function
    ker = lambda x: c * np.exp(-d * (x**2))

    # compute kernel matrix
    K = h * ker(y)

    # compute 1D kernel vector (for convolution)
    kx = ker(np.arange(-(n - 0.5), n, 2.0))

    return K, kx



def get_source(n):
    """
    get source (i.e., x_true) for 1D deconvolution problem

    parameters:
        n (int): number of grid points
        id (int): choice of source pattern (default is 3)

    returns:
        x (ndarray): data vector of shape (n,)
    """
    # compute coordinate grid (cell-centered points)
    h = 1.0 / n
    x_grid = np.linspace(h, 1 - h, n)

    # compute data based on id
    i1 = (x_grid > 0.10) & (x_grid < 0.25)
    i2 = (x_grid > 0.30) & (x_grid < 0.35)
    i3 = (x_grid > 0.50) & (x_grid < 1.00)
    x = 0.75 * i1 + 0.25 * i2 + (np.sin(2 * np.pi * x_grid) ** 4) * i3

    # normalize data
    x = x / np.linalg.norm(x)

    return x



def plot_data():
    n = 256  # number of points
    gamma = 50

    # get discrete convolution operator (Kernel matrix)
    K, _ = get_kernel(n, tau=0.03)
    x_true = get_source(n)

    # compute smooth data
    y = K @ x_true

    delta = np.linalg.norm(y) / (gamma * np.sqrt(n))
    eta = np.random.normal(0, delta, size=y.shape)
    y_obs = y + eta

    w = np.linspace(0, 1, n) # domain
    plt.plot(w,x_true)
    plt.plot(w,y)
    plt.plot(w,y_obs)
    plt.savefig('deconvolution-source.pdf')

    # plot singular values (log scale)
    plt.matshow(K)
    plt.savefig('convolution-operator.pdf')
    #plt.show()


if __name__ == "__main__":
    plot_data()


###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
