import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from sklearn import datasets, cluster, preprocessing

n_samples = 500
seed = 30

X,y = skl.datasets.make_circles( n_samples=n_samples,
                    factor=0.5, noise=0.05, random_state=seed )

# normalize dataset for easier parameter selection
X = skl.preprocessing.StandardScaler().fit_transform(X)

# number of examples
n = y.shape[0]

n_clusters = 2

# setup algorithm
spec_clust = skl.cluster.SpectralClustering(
                    n_clusters=n_clusters,
                    eigen_solver="arpack",
                    affinity="nearest_neighbors",
                    random_state=42)

t0 = time.time()

# ignore warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning)
    # run spectral clustering
    spec_clust.fit(X)

t1 = time.time()

# generate prediction
y_pred = spec_clust.labels_.astype(int)

# number of examples
n = y.shape[0]

# number of examples in class 1 and class 2
nc1 = sum(y)
nc2 = n - nc1


# visualize results
X1 = np.zeros( (nc1,2) )
X2 = np.zeros( (nc2,2) )
X1_pred = np.zeros( (nc1,2) )
X2_pred = np.zeros( (nc2,2) )

k1 = 0
k2 = 0
kp1 = 0
kp2 = 0
for j in range(n):
    if ( y[j] == 1 ):
        X1[k1,:] = X[j,:]
        k1 = k1 + 1
    else:
        X2[k2,:] = X[j,:]
        k2 = k2 + 1

    # the classes are flipped in the predictions
    if ( y_pred[j] == 0 ):
        X1_pred[kp1,:] = X[j,:]
        kp1 = kp1 + 1
    else:
        X2_pred[kp2,:] = X[j,:]
        kp2 = kp2 + 1


fig, ax = plt.subplots(2,1)
fig.tight_layout(pad=5.0)
ax[0].scatter( X1[:, 0], X1[:, 1], s=10 )
ax[0].scatter( X2[:, 0], X2[:, 1], s=10 )
ax[0].set_xlabel( r'$X_1$')
ax[0].set_ylabel( r'$X_2$')


ax[1].scatter( X1_pred[:, 0], X1_pred[:, 1], s=10 )
ax[1].scatter( X2_pred[:, 0], X2_pred[:, 1], s=10 )
ax[1].set_xlabel( r'$X_{\text{pred},1}$')
ax[1].set_ylabel( r'$X_{\text{pred},2}$')
plt.show()

print(y_pred)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
