import numpy as np
import matplotlib.pyplot as plt

# linear system
#   x = 1
#  2x = 11/5
# is inconsistent; solve in least squares sense
#   minimize f(x) = sum_i (a_i x - b_i)^2

# problem parameters
k = np.array([1.0, 2.0])
y = np.array([1.0, 11.0/5.0])

# objective function f(x) for vectorized x
def f(x):
    # x may be scalar or array; broadcast to compute (a_i x - b_i)^2 and sum over i
    return (k[0]*x - y[0])**2 + (k[1]*x - y[1])**2

# closed-form LS solution: x* = (sum a_i b_i) / (sum a_i^2)
xsol = (k @ k) / (y @ y)

# coordinate vector to evaluate objective function
h = 0.01
x = np.arange(-16.0, 16.0, h)

# plot
plt.figure()
plt.plot(x, f(x), linewidth=2, label='function f(x)')
plt.plot([xsol], [f(xsol)], 'rx', markersize=10, linewidth=2, label='solution of problem')
plt.title(r'$f(x) = (x - 1)^2 + (2x - 2.2)^2$', fontsize=16)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$f(x)$', fontsize=16)
plt.grid(True)
plt.legend()
plt.gca().tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('objective-landscape-existence.pdf')
#plt.show()

# print solution for reference
print("least-squares minimizer x* =", xsol)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################

