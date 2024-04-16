import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def show_graph( adj_mat ):
    # visualiztion of graph
    rows, cols = np.where(adj_mat == 1)

    # compute deges
    e = zip( rows.tolist(), cols.tolist() )

    # setup graph
    gr = nx.Graph()
    all_rows = range(0, adj_mat.shape[0])
    for n in all_rows:
        gr.add_node( n )
    gr.add_edges_from( e )
    nx.draw( gr, node_size = 50 )
    plt.show()

# generate dataset
ns = 100
X = np.random.randn( ns, 2)

# compute connectivity matrix
A = radius_neighbors_graph(X, 0.4, mode='connectivity', metric='minkowski',
                           p=2, metric_params=None, include_self=False )

# map to array
A = A.toarray()

# visualize connectivyt graph
show_graph( A )




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
