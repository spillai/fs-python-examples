# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Spectral embedding

# <codecell>

%reset

# <codecell>

import numpy as np
import pylab as pl
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.manifold import SpectralEmbedding
pl.rcParams['figure.figsize'] = 20,12

###############################################################################
l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

###############################################################################

# <codecell>

###############################################################################
# 2 circles
img = circle1 + circle2
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.1 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

se = SpectralEmbedding(n_components=5,affinity='precomputed')
Y = se.fit_transform(graph)

for j in range(0,se.n_components):
    pl.subplot(1,se.n_components, j+1)
    label_im = -np.ones(mask.shape)
    label_im[mask] = Y[:,j]
    pl.title('Eigen Vec. %i' % (j+1), size=18)
    pl.imshow(label_im,cmap=pl.cm.Spectral)
pl.show()

pl.figure()
pl.scatter(Y[:, 1], Y[:, 2], c=Y[:,0], cmap=pl.cm.Spectral)

# <codecell>

# 4 circles
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.1 * np.random.randn(*img.shape)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependant from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())


se = SpectralEmbedding(n_components=5,affinity='precomputed')
Y = se.fit_transform(graph)

for j in range(0,se.n_components):
    pl.subplot(1,se.n_components, j+1)
    label_im = -np.ones(mask.shape)
    label_im[mask] = Y[:,j]
    pl.title('Eigen Vec. %i' % (j+1), size=18)
    pl.imshow(label_im,cmap=pl.cm.Spectral)
pl.show()

pl.figure()
pl.scatter(Y[:, 0], Y[:, 1], c=Y[:,2], cmap=pl.cm.jet)

# <codecell>

a = np.random.random((4,4))
print a

# <codecell>


