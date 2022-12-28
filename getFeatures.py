from PIL import Image
import os

from sklearn.cluster import KMeans
import shutil
from classification import Classification
from sklearn.decomposition import PCA
import numpy as np

classfication = Classification()
#--------------------------------------------#
#   提取原始数据特征向量作为PCA的训练集
#   数据量需要大于256
#--------------------------------------------#
img_path = 'datasets(1)/dogs'
feats = []
imgs = os.listdir(img_path)
for img in imgs:
    img = os.path.join(img_path, img)
    image = Image.open(img)
    feat = classfication.getFeature(image)
    feats.append(feat.cpu().numpy().tolist())

feats = np.array(feats)
feats = np.squeeze(feats)
print(feats.shape)

#--------------------------------------------#
#   训练PCA
#--------------------------------------------#
pca = PCA(n_components=256)
pca.fit(feats)

#--------------------------------------------#
#   提取增强数据的特征向量
#--------------------------------------------#
aug_img_path = 'datasets(1)/dogs_aug'
feats_test = []  # 增强数据降维前的特征向量
aug_imgs = os.listdir(aug_img_path)
for img in aug_imgs:
    img = os.path.join(aug_img_path, img)
    image = Image.open(img)
    feat = classfication.getFeature(image)
    feats_test.append(feat.cpu().numpy().tolist())

#--------------------------------------------#
#   PCA降维
#   1x1024 => 1x256
#--------------------------------------------#
newFeats = []  # 增强数据降维后的特征向量
for feat in feats_test:
    newFeat = pca.transform(feat)
    newFeat = newFeat.squeeze()
    newFeats.append(newFeat)
    # print(newFeat)


#--------------------------------------------#
#   K-Means聚类
#--------------------------------------------#
km = KMeans(init='k-means++', n_clusters=2, max_iter=300)
y_means = km.fit_predict(newFeats)
print(y_means)


#--------------------------------------------#
#   筛选图像
#--------------------------------------------#
valid_imgs = []
invalid_imgs = []
valid_path = 'datasets(1)/valid'
invalid_path = 'datasets(1)/invalid'
for i in range(len(y_means)):
    if y_means[i] == 1:
        valid_imgs.append(aug_imgs[i])
    else:
        invalid_imgs.append(aug_imgs[i])

for valid_img in valid_imgs:
    shutil.move(os.path.join(aug_img_path, valid_img), os.path.join(valid_path, valid_img))

for invalid_img in invalid_imgs:
    shutil.move(os.path.join(aug_img_path, invalid_img), os.path.join(invalid_path, invalid_img))
