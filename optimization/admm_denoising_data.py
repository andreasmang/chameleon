import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import math

def get_denoise_data1D( n ):

    # generate random data
    x_true = np.ones( n )
    for i in range(3):
        # draw random sample from
        j2 = rnd.sample( range(0,n), 1 )[0]
        j1 = math.ceil(j2/2)

        k = rnd.sample( range(0,10), 1 )[0]

        x_true[ j1 : j2 ] = (k+1)*x_true[ j1 : j2 ]

    y_obs = x_true + np.random.randn( n )
    return x_true, y_obs


n = 256
dom = np.linspace(0,1,256)
x_true, y_obs = get_denoise_data1D( n )


plt.plot(dom, x_true, label=r"$x_\text{true}$")
plt.plot(dom, y_obs, label=r"$y_\text{obs}$")
plt.legend()
plt.grid(True)
plt.savefig('admm-denoising-data.pdf')

#plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
