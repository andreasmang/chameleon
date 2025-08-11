import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def obj(x):
    return np.linalg.norm(x, 2)

# define domain
h = 0.05
x = np.arange(-1.0, 1.0, h)
x1, x2 = np.meshgrid(x, x, indexing='xy')

# objective on the grid: f(x1,x2) = (x1 + x2 - 1)^2
fx = (x1 + x2 - 1.0) ** 2

# line of feasible points x1 + x2 = 1
lx1 = np.arange(0.0, 1.0, h)
lx2 = 1.0 - lx1
lz = np.zeros_like(lx1)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x1, x2, fx, cmap='summer', alpha=0.5, linewidth=0, antialiased=True)
cont = ax.contour(x1, x2, fx, levels=40, colors='k', linestyles='solid', offset=None)

# mark the point (1/2, 1/2, f(1/2,1/2)) = (0.5, 0.5, 0)
x_half = 0.5
f_half = (x_half + x_half - 1.0) ** 2
ax.plot([x_half], [x_half], [f_half], 'rx', markersize=10, mew=2, label='(1/2, 1/2)')

# plot feasible line on z=0 (since f=0 on x1+x2=1)
ax.plot(lx1, lx2, lz, 'ko', markersize=3, label=r'$x_1+x_2=1$')

ax.set_title(r'$f(x_1,x_2)=(x_1+x_2-1)^2$')
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_zlabel(r'$f(x_1,x_2)$', fontsize=12)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_box_aspect((1, 1, 0.6))  # nicer proportions
plt.tight_layout()
plt.savefig('objective-landscape-uniquness.pdf')
#plt.show()

# equality-constrained minimization: min ||x||_2 s.t. x1 + x2 = 1
# closed-form (lagrange multipliers) solution for reference: (1/2, 1/2)
x_star_closed = np.array([0.5, 0.5])

x0 = np.array([-1.0, 3.0])
cons = {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1.0}
# SLSQP handles linear equality constraints well
res = minimize(obj, x0, method='SLSQP', constraints=[cons])
x_star = res.x
if not res.success:
    print("scipy reported a warning:", res.message)

# final report
print(f"solution is ({x_star[0]:.2f} {x_star[1]:.2f})  # expected (0.50 0.50)")




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################


