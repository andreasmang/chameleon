import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets, preprocessing, cluster, metrics

# create example based on blobs
def make_data_blob(n_samples=600, centers=[(0, 0), (4, 4)], cluster_std=0.6):
    x, y = datasets.make_blobs(n_samples=n_samples, centers=centers,
                cluster_std=cluster_std,random_state=42)
    x = preprocessing.StandardScaler().fit_transform(x)
    return x, y


# create example based on circles
def make_data_circ(n_samples=500, noise=0.05, factor=0.5):
    x, y = datasets.make_circles(n_samples=n_samples, factor=factor,
                                 noise=noise, random_state=42)
    x = preprocessing.StandardScaler().fit_transform(x)

    return x, y

# run k-means
def run_kmeans(X, n_clusters=2, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    y_pred = km.fit_predict(X)
    return y_pred

# compute performance scores
def scores(y_true, y_pred):
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

# plot results
def plot_results(x, y_true, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(x[:,0], x[:,1], c=y_true, s=10)
    ax[0].set_title("ground truth")
    ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")
    ax[1].scatter(x[:,0], x[:,1], c=y_pred, s=10)
    ax[1].set_title("predicted")
    ax[1].set_xlabel("$x_1$"); ax[1].set_ylabel("$x_2$")
    fig.tight_layout()
    plt.show()



n_samples = 500 # number of samples
noise=0.5 # noise

#x, y = make_data_circ(n_samples=n_samples, noise=noise)
x, y = make_data_blob(n_samples=n_samples)
y_pred_km = run_kmeans(x, n_clusters=2)

#scatter_two_panels(X, y, y_pred_km, os.path.join(outdir, "kmeans.png"))
ari_km, nmi_km = scores(y, y_pred_km)
print('ari', ari_km)
print('nmi', nmi_km)

plot_results(x, y, y_pred_km)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
