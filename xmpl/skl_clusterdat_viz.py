import sklearn as skl
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

n_samples = 500
seed = 30

noisy_circles = skl.datasets.make_circles( n_samples=n_samples,
                    factor=0.5, noise=0.05, random_state=seed )

noisy_moons = skl.datasets.make_moons( n_samples=n_samples,
                    noise=0.05, random_state=seed )


for i in range(2):

    if i == 1:
        X, y = noisy_circles
    else:
        X, y = noisy_moons

     # normalize dataset for easier parameter selection
    X = skl.preprocessing.StandardScaler().fit_transform(X)

    # number of examples
    n = y.shape[0]


    # number of examples in class 1 and class 2
    nc1 = sum(y)
    nc2 = n - nc1

    X1 = np.zeros( (nc1,2) )
    X2 = np.zeros( (nc2,2) )

    k1 = 0
    k2 = 0
    for j in range(n):
        if (y[j] == 1 ):
            X1[k1,:] = X[j,:]
            k1 = k1 + 1
        else:
            X2[k2,:] = X[j,:]
            k2 = k2 + 1

    plt.scatter( X1[:, 0], X1[:, 1], s=10 )
    plt.scatter( X2[:, 0], X2[:, 1], s=10 )
    plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
