# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Tutorial on Spectral Clustering

# <markdowncell>

# ** Load data ** 
# 
# 1. Noisy circles
# 2. Noisy Moons
# 3. Blobs
# 4. Structure-less data

# <codecell>

import time

import numpy as np
import pylab as pl

from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

pl.rcParams['figure.figsize'] = 20,12
np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 750
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

pl.figure(figsize=(14, 9.5))
pl.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                   hspace=.01)

plot_num = 1

datasets = [noisy_circles, noisy_moons, blobs, no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # plot
    pl.subplot(2, 2, plot_num)
    pl.scatter(X[:, 0], X[:, 1], color='b', s=10)

    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)
    pl.xticks(())
    pl.yticks(())
    plot_num += 1

pl.show()

# <codecell>

plot_num = 1
n_clusters=3

# datasets = [noisy_circles, noisy_moons, blobs, no_structure]
datasets = [no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Compute distances
    #distances = np.exp(-euclidean_distances(X))
    distances = euclidean_distances(X)

    # create clustering estimators
    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    spectral = cluster.SpectralClustering(n_clusters=n_clusters,
                                          eigen_solver='arpack',
                                          affinity='nearest_neighbors',
                                          assign_labels='discretize')
    


    for algorithm in [two_means, spectral]:
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        pl.subplot(1, 2, plot_num)
        if i_dataset == 0:
            pl.title(str(algorithm).split('(')[0], size=18)
        pl.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            pl.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        pl.xlim(-2.5, 2.5)
        pl.ylim(-2.5, 2.5)
        pl.xticks(())
        pl.yticks(())
        pl.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=pl.gca().transAxes, size=15,
                horizontalalignment='right')
        plot_num += 1

pl.show()

# <codecell>

e

