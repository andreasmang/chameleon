import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# data
K = np.array([[0.16, 0.10],
              [0.17, 0.11],
              [2.02, 1.29]], dtype=float)
x_true = np.array([1.0, 1.0])
eta    = np.array([0.01, -0.03, 0.02])

# compute data vector
y_obs = K @ x_true + eta

# objective f(x) = ||a x - b||_2
def obj_fun(x):
    r = K @ x - y_obs
    return np.linalg.norm(r, 2)

# Least-squares solution via normal equations: (A^T A) x = A^T b
x_isol = np.linalg.solve(K.T @ K, K.T @ y_obs)

# Condition number (2-norm)
condK = np.linalg.cond(K, 2)

# Print results (formatted like MATLAB)
print(f"cond(K): {condK:.6e}")

print(f"xtrue   = ({x_true[0]:+0.3f}, {x_true[1]:+0.3f}), f(x) = {obj_fun(x_true):.6f}")
print(f"xisol   = ({x_isol[0]:+0.3f}, {x_isol[1]:+0.3f}), f(x) = {obj_fun(x_isol):.6f}")

x_trail1 = np.array([1.65, 0.0])
x_trail2 = np.array([0.0, 2.58])
print(f"xtrail  = ({x_trail1[0]:+0.3f}, {x_trail1[1]:+0.3f}), f(x) = {obj_fun(x_trail1):.6f}")
print(f"xtrail  = ({x_trail2[0]:+0.3f}, {x_trail2[1]:+0.3f}), f(x) = {obj_fun(x_trail2):.6f}")

# (optional) numerically stabler alternative:
x_ls, *_ = np.linalg.lstsq(K, y_obs, rcond=None)
print(f"x_lstsq = ({x_ls[0]:+0.3f}, {x_ls[1]:+0.3f}), f(x) = {obj_fun(x_ls):.6f}")


# plot f over [-20, 20]^2
n = 256
x1 = np.linspace(-20.0, 20.0, n)
x2 = np.linspace(-20.0, 20.0, n)
X1, X2 = np.meshgrid(x1, x2, indexing="xy")

# Evaluate f on the grid without loops:
# For each (x1, x2), r = A*[x1; x2] - b in R^3, so f = sqrt(sum r_i^2)
R1 = K[0, 0] * X1 + K[0, 1] * X2 - y_obs[0]
R2 = K[1, 0] * X1 + K[1, 1] * X2 - y_obs[1]
R3 = K[2, 0] * X1 + K[2, 1] * X2 - y_obs[2]
FX = np.sqrt(R1**2 + R2**2 + R3**2)

# 2D filled contour plot
plt.figure(figsize=(7, 6))
cf = plt.contourf(X1, X2, FX, levels=40)
plt.contour(X1, X2, FX, levels=40, colors='k', linewidths=0.4)
plt.plot(x_true[0], x_true[1], 'r*', markersize=10, label='x_true')
plt.plot(x_isol[0], x_isol[1], 'rx', markersize=8, label='x_isol')
plt.plot(x_trail1[0], x_trail1[1], 'o', label='x_trail1')
plt.plot(x_trail2[0], x_trail2[1], 'o', label='x_trail2')
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title(r'$f(x)=\|Kx-y\|_2$', fontsize=12)
plt.colorbar(cf, label=r'$f(x)$')
plt.legend()
plt.tight_layout()
plt.savefig("objective-landscape-contour-stability.pdf")
#plt.show()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, FX, linewidth=0, antialiased=True)
plt.title(r'$f(x)=\|Kx-y\|_2$')
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_zlabel(r'$f(x)$', fontsize=12)
fig.colorbar(surf, shrink=0.6, aspect=10, label=r'$f(x)$')
plt.tight_layout()
plt.savefig("objective-landscape-contour-surface.pdf")
#plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################

