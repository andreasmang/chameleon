import numpy as np
import matplotlib.pyplot as plt

def get_K(n):
    h = 1.0/n
    return np.tril(np.ones((n,n))) * h  # discrete integration

def get_L(n):
    L = np.zeros((n-1, n))
    for i in range(n-1):
        L[i, i]   = -1.0
        L[i, i+1] =  1.0
    return L

def sol_KKT(K, L, ydelta, alpha):
    # Solve (K^T K + alpha L^T L) x = K^T ydelta
    A = K.T @ K + alpha * (L.T @ L)
    b = K.T @ ydelta
    return np.linalg.solve(A, b)

n = 100
t = np.linspace(0, 1, n, endpoint=False)

# ground truth
x_true = np.exp(-(t-0.3)**2/0.002) + 0.5*np.exp(-(t-0.7)**2/0.005)

# compute true (clean) y
K = get_K(n)
y = K @ x_true

# compute observation y_obs
gamma = 0.05  # relative noise (5%)
delta = gamma * np.linalg.norm(y)
rng = np.random.default_rng(0)
noise = rng.standard_normal(n)
noise = delta * noise / np.linalg.norm(noise)
y_obs = y + noise

# choose regularization operator
L = get_L(n)

# plots
plt.figure();
plt.plot(y, 'r--', label=r'$y$')
plt.plot(y_obs, 'b-.', label=r'$y_{\text{obs}}$')
plt.plot(x_true, 'k-', label=r'$x_{\text{true}}$')
plt.legend(); plt.tight_layout()
plt.savefig('integration-data.pdf')

# regularization operator
plt.figure();
plt.matshow(L)
plt.savefig('integration-regularization-operator.pdf')

# forward operator
plt.figure();
plt.matshow(K)
plt.savefig('integration-forward-operator.pdf')

#plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
