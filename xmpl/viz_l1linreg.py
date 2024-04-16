import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../data")

from Data import *

# setup data
dat = Data()
xtrue,y,A = dat.get_sparse_reg_dat()

n = xtrue.shape[0]

fig, ax = plt.subplots(2,1)
ax[0].plot( xtrue )
ax[1].plot( y )
plt.show()
