"""
projected subgradient for basis pursuit (inpainting)
- transform choice: dct (orthonormal, type-ii/iii pair)
- mask sampling (noise-free): projection overwrites observed entries
"""
import numpy as np
from numpy.linalg import norm
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

def run_dct(x, nside):
    X = x.reshape(nside, nside)
    return dct(dct(X, type=2, norm='ortho', axis=0), type=2, norm='ortho', axis=1).ravel()

def run_idct(u, nside):
    U = u.reshape(nside, nside)
    return idct(idct(U, type=2, norm='ortho', axis=0), type=2, norm='ortho', axis=1).ravel()


def gen_data(n):
    # synthetic image with squares and disks
    X = np.zeros((n, n))
    X[n//8:3*n//8, n//8:3*n//8] = 1.0
    X[5*n//8:7*n//8, 5*n//8:7*n//8] = 0.7
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    circ = (xx - 3*n/4)**2 + (yy - n/3)**2 <= (n/8)**2
    X[circ] = 0.9

    return X.ravel()


def gen_mask(n, obs_ratio):
    rng = np.random.default_rng(0)

    m = int(round(obs_ratio * n))
    idx = rng.choice(n, size=m, replace=False)
    M = np.zeros(n, dtype=bool)
    M[idx] = True
    return M  # boolean mask where true=observed


# main code to solve problem
def run(n=128, obs_ratio=0.8):

    # generate true data
    x_true = gen_data(n)

    # generate mask (i.e., implementation of K x)
    N = x_true.size
    M = gen_mask( N, obs_ratio )

    # y = Kx
    y = x_true[M]

    # function handles for "matvec"
    W = lambda x: run_dct(x, n)
    WT = lambda u: run_idct(u, n)

    # apply DCT matrix and transpose
    u = W( x_true )
    v = WT( u )

    # this is the initial guess for the algorithm
    x0 = np.zeros_like( x_true )
    x0[M] = y
    x_obs = x0

    fig = plt.figure()
    plt.subplot(1,4,1); plt.imshow(x_true.reshape(n,n),vmin=0.0, vmax=1.0); plt.axis('off'); plt.title(r'$x_\text{true}$')
    plt.subplot(1,4,2); plt.imshow(x_obs.reshape(n,n),vmin=0.0, vmax=1.0); plt.axis('off'); plt.title(r'$x_\text{obs}$')
    plt.subplot(1,4,3); plt.imshow(u.reshape(n,n),vmin=-0.2, vmax=0.2); plt.axis('off'); plt.title(r'$W x_\text{true}$')
    plt.subplot(1,4,4); plt.imshow(v.reshape(n,n),vmin=0.0, vmax=1.0); plt.axis('off'); plt.title(r'$W^\mathsf{T}W x_\text{true}$')

    plt.matshow(M.reshape(n,n))
    plt.show()



if __name__ == '__main__':

    n = 256 # mesh size is 256 x 256
    obs_ratio = 0.3 # ratio of data being observed (0.9 -> 90%); the smaller the more difficult
    t0 = 1.0

    run(n=n, obs_ratio=obs_ratio)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
