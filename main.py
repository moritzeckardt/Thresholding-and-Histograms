'''
Created on 25.11.2017
Modified on 05.12.2020

@author: Daniel, Max, Charly, Mathias
'''
import cv2
import matplotlib.pyplot as plt
from otsu import otsu

img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)

res = otsu(img)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('Original')
if res is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(res, 'gray')
    plt.title('Otsu\'s - Threshold = 120')
plt.show()
