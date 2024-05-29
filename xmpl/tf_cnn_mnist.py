import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import sklearn as sk

from sklearn.model_selection import KFold
import sys

sys.path.append("..")
sys.path.append("../data")

from NNData import *


def setup():
    model = keras.models.Sequential()
    input_shape = (28, 28, 1)
    model.add( keras.Input( shape=input_shape ) )
    c2_layer = keras.layers.Conv2D( 32, (3, 3),
                                activation='relu',
                                kernel_initializer='he_uniform' )
    model.add( c2_layer )
    model.add( keras.layers.MaxPooling2D( (2, 2) ) )
    model.add( keras.layers.Flatten() )
    model.add( keras.layers.Dense(100, activation='relu',
                                kernel_initializer='he_uniform') )
    model.add( keras.layers.Dense(10, activation='softmax') )

    opt = keras.optimizers.SGD( learning_rate=0.01, momentum=0.9 )

    # compile model
    model.compile( optimizer=opt, loss='categorical_crossentropy',
                    metrics=['accuracy'] )

    return model

def report_diagnostics( logs ):
    for i in range(len(logs)):
        plt.subplot(2,1,1)
        x_loss = log[i].history['loss']
        y_loss = log[i].history['val_loss']
        plt.plot( x, color='blue', label='train' )
        plt.plot( y, color='red', label='label' )

        plt.subplot(2,1,2)
        x_acc = log[i].history['accuracy']
        y_acc = log[i].history['accuracy_loss']
        plt.plot( x_acc, color='blue', label='train' )
        plt.plot( y_acc, color='red', label='label' )


def report_performance( scores ):
    m = np.mean(scores)*100.0
    s = np.std(scores)*100.0
    print("accuracy: (mean,std) = (.3f,.3f)" % m, s)



def train_model_crossval( X, Y, n_folds=5 ):
    scores, logs = list(), list()

    # prepare cross validation
    kfold = sk.model_selection.KFold( n_folds, shuffle=True, random_state=1 )

    # enumerate splits
    for train_ix, test_ix in kfold.split( X ):
        # define model
        model = setup()

        # select rows for train and test
        X_train, Y_train = X[train_ix], Y[train_ix]
        X_test, Y_test = X[test_ix], Y[test_ix]

        # fit model
        log = model.fit(X_train, Y_train, epochs=10, batch_size=32,
                        validation_data=(X_test, Y_test), verbose=1)

        # evaluate model
        _, acc = model.evaluate( X_test, Y_test, verbose=0 )

        print('train: X=%s Y=%s' % (X_train.shape, Y_train.shape) )
        print('test: X=%s Y=%s' % (X_test.shape, Y_test.shape) )
        print('test accuracy: %.3f' % (acc * 100.0))

        # stores scores
        scores.append( acc )
        logs.append( log )

    return scores, logs


def train_model( X, Y, X_test, Y_test ):

    # define model
    model = setup()

    # fit model
    log = model.fit( X, Y, epochs=10, batch_size=32, verbose=1 )

    # evaluate model
    _, acc = model.evaluate( X_test, Y_test, verbose=0 )

    print('test accuracy %.3f' % (acc * 100.0))

    return


cross_val = False
X,Y = get_mnist()

if cross_val == True:
    # execute training with cross validation
    train_model_crossval( X, Y )
    report_diagnostics( logs )
    report_performance( scores )

else:
    # execute training
    X_test,Y_test = get_mnist("test")
    train_model( X, Y, X_test, Y_test )




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
