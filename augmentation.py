import os

import cv2 as cv
import random
import matplotlib.pyplot as plt
import numpy as np

def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

img_path = 'datasets(1)/dogs'
img_save_path = 'datasets(1)/dogs_aug'
imgs = os.listdir(img_path)

for i in imgs:
    img = cv.imread(os.path.join(img_path, i))
    prob = random.randint(1, 100)/100.0
    # print(prob)
    img_sp = sp_noise(img, prob)  # 椒盐噪声
    cv.imwrite(os.path.join(img_save_path, i[0:-4]+'_SPnoise.jpg'), img_sp)
