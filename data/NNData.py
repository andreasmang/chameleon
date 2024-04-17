from tensorflow.keras.datasets import mnist
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def get_mnist( flag="train" ):

    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    if flag == "train":
        m = X_train.shape[0]
        n = X_train.shape[1]

        X = X_train.reshape( (m, n, n, 1) )

        # create binary matrix for 10 classes
        rows = np.array( range(0,m) )
        cols = y_train

        data = np.ones( m )
        Y = sp.sparse.csr_matrix((data, (rows,cols)), shape=(m,10)).toarray()

    elif flag == "test":
        m = X_test.shape[0]
        n = X_test.shape[1]

        X = X_test.reshape( (m, n, n, 1) )

        # create binary matrix for 10 classes
        rows = np.array( range(0,m) )
        cols = y_test

        data = np.ones( m )
        Y = sp.sparse.csr_matrix((data, (rows,cols)), shape=(m,10)).toarray()

    else:
        print("flag", flag, "not defined")


    # normalize data
    X = X.astype('float32')
    X = np.divide( X, 255.0)

    # summarize loaded dataset
    print('X=%s, Y=%s' % (X.shape, Y.shape))

    return X,Y




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
