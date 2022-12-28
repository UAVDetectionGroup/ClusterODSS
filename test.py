from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

"""x = np.random.rand(1024, 1024)
# print(x)
X = np.random.rand(1, 1024)
pca = PCA(256)
trainX = pca.fit_transform(x)
# print(trainX)
newX = pca.transform(X)
print(newX)"""

# print(pca.explained_variance_ratio_)

feats = []
for i in range(20):
    X = np.random.rand(1, 256)
    X = X.squeeze()
    print(X.shape)
    feats.append(X)
km = KMeans(init='k-means++', n_clusters=2, max_iter=300)
y_means = km.fit_predict(feats)
print(y_means)