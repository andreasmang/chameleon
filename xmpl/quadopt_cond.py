import numpy as np
import matplotlib.pyplot as plt


def gen_random_spd_matrix(n, cond_number=10.0, seed=None):
    """
    generate a random symmetric positive definite matrix of size n×n
    with approximately the desired condition number
    """
    rng = np.random.default_rng(seed)

    # step 1: create random orthogonal matrix Q via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # step 2: create eigenvalues spaced between 1 and cond_number
    eigvals = np.linspace(1, cond_number, n)

    # shuffle eigenvalues for randomness
    rng.shuffle(eigvals)

    # step 3: form A = Q Λ Q^T
    A = (Q * eigvals) @ Q.T

    # step 4: symmetrize (not necessary; numerical stability)
    A = 0.5 * (A + A.T)

    return A


n = 100 # matrix dimensions
maxit = 2000 # number of max iterations
rng = np.random.default_rng(None)
b = rng.standard_normal((n,1)) # random right hand side

A = gen_random_spd_matrix(n,1e1) # generate random spd matrix

x_true = np.linalg.inv(A) @ b # compute true solution based on optimality cond

lambda_max = np.linalg.eigvalsh(A).max()
print('max eigenvalue: ', lambda_max)
print('condition number: ', np.linalg.cond(A))



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
