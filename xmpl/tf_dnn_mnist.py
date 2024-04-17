import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist

import sys

sys.path.append("..")
sys.path.append("../data")

from NNData import *

# load MNIST data
X_train,Y_train = get_mnist("train")
X_test,Y_test = get_mnist("test")


# create neural network model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile( loss='categorical_crossentropy',
                optimizer='adam', metrics=['acc'])

# train neural network
model.fit(X_train, Y_train, epochs=5, validation_data=(X_test,Y_test))

# predict using our trained model
Y_pred = model.predict( X_test )
print( Y_pred[1,:] )

Y_pred = np.argmax( Y_pred, axis=1 )
print( Y_pred )

# display some predictions on test data
fig, ax = plt.subplots(ncols=5, sharex=False,
                sharey=True, figsize=(20, 4))
for i in range(5):
    ax[i].set_title(Y_pred[i])
    ax[i].imshow(X_test[i], cmap='gray')
    ax[i].get_xaxis().set_visible( False )
    ax[i].get_yaxis().set_visible( False )

plt.show()




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
