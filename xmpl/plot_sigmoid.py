import matplotlib.pyplot as plt
import numpy as np

n = 512; # number of points

# create input values
z = np.linspace(-100, 100, n)

# evaluate sigmoid function
y = 1.0 / (1.0 + np.exp(-z))

# plot sigmoid function
plt.plot( z, y )
plt.ylabel( r'$\sigma(x)$' )
plt.xlabel( 'x' )
plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
