import sys
sys.path.append("..")
sys.path.append("../data")

from Data import *

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# initialize class to load images
dat = Data()

# load dataset
n = 28;
Y,C,L = dat.read_mnist( )

print("Y", Y.shape)
print("C", C.shape)
print("L", L.shape)

# get number of images
m = Y.shape[0]

print("number of datasets", m);

print("labels");
print(L[1:10])

print("binary representation of labels");
print(C[1:10,:])


# select 10 random images
y = np.zeros([10,n,n])
labels = [0]*10
for i in range(0,10):
    k = rnd.randint(0,m)
    y[i,:,:] = Y[k,:].reshape(n,n)
    labels[i] = L[k]


print("random labels drawn:", labels)
# display images
ax01 = plt.subplot(2, 5, 1)
ax01.imshow(y[0,:,:], cmap='gray')
ax01.axis('off')

ax02 = plt.subplot(2, 5, 2)
ax02.imshow(y[1,:,:], cmap='gray')
ax02.axis('off')

ax03 = plt.subplot(2, 5, 3)
ax03.imshow(y[2,:,:], cmap='gray')
ax03.axis('off')

ax04 = plt.subplot(2, 5, 4)
ax04.imshow(y[3,:,:], cmap='gray')
ax04.axis('off')

ax05 = plt.subplot(2, 5, 5)
ax05.imshow(y[4,:,:], cmap='gray')
ax05.axis('off')

ax06 = plt.subplot(1, 5, 1)
ax06.imshow(y[5,:,:], cmap='gray')
ax06.axis('off')

ax07 = plt.subplot(1, 5, 2)
ax07.imshow(y[6,:,:], cmap='gray')
ax07.axis('off')

ax08 = plt.subplot(1, 5, 3)
ax08.imshow(y[7,:,:], cmap='gray')
ax08.axis('off')

ax09 = plt.subplot(1, 5, 4)
ax09.imshow(y[8,:,:], cmap='gray')
ax09.axis('off')

ax10 = plt.subplot(1, 5, 5)
ax10.imshow(y[9,:,:], cmap='gray')
ax10.axis('off')

plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
