import matplotlib.pyplot as plt
import numpy as np
import math

def apply_POmega(X, Y, Omega):
    """
    gradientfor f(X) = 0.5 * ||P_Omega(X - Y)||_F^2
    returns P_Omega(X - Y).
    """
    return (X - Y) * Omega


m = 28 #300
n = 28 #300
r_true = 5 # true rank
p = 0.1 # probability
sigma = 0.05

# ground truth low-rank M* = A B^T
A = np.random.randn(m, r_true)
B = np.random.randn(n, r_true)
M_true = A @ B.T

# normalize so ||M*||_F / sqrt(mn) = 1
M_true = M_true / (np.linalg.norm(M_true, 'fro') / math.sqrt(m * n))

# sampling
Omega = (np.random.rand(m, n) < p)

# observations (zeros outside Omega)
Y = np.zeros((m, n))
nnz = np.count_nonzero(Omega)
Y[Omega] = M_true[Omega] + sigma * np.random.randn(nnz)

X = np.zeros_like(Y)

Z = apply_POmega(X, Y, Omega)


# viz the data
fig, axs = plt.subplots(1, 3, figsize=(10, 4))
axs[0].imshow(Y, cmap='viridis', origin='lower', aspect='auto')
axs[0].set_title('Y')

axs[1].imshow(Z, cmap='viridis', origin='lower', aspect='auto')
axs[1].set_title('Z')

axs[2].imshow(M_true, cmap='viridis', origin='lower', aspect='auto')
axs[2].set_title('M_true')

plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
