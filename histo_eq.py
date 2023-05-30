# Implement the histogram equalization in this file
import numpy as np
from PIL import Image
import cv2 as cv

#Open image and convert into numpy array
def convert_img(file):
    img = cv.imread(file, 0)
    return img


#Create histogram of given image
def create_histogram(img):
    hist = np.zeros(256, dtype=int)

    for i in img:
        for k in i:
            hist[k] += 1

    return hist 
    

#Calculate cumulative distribution function
def cumulative_distribution(hist):

    cumsum = 0
    total_pixels = np.sum(hist)
    cdf = np.zeros(256)

    for pixel in range(256): 
        cumsum += hist[pixel]
        cdf[pixel] = cumsum / total_pixels
    
    return cdf

def gray_values(cdf, img):
    
    for row in img:
        for pixel in row:
            pixelvalue_old = img[row, pixel]
            pixelvalue_new = ((cdf[pixelvalue_old] - np.min(cdf)) / 1 - np.min(cdf)) * 255
            img[row, pixel] = pixelvalue_new

    return img
    

if __name__ == "__main__":
    img = convert_img('hello.png')
    hist = create_histogram(img)
    cdf = cumulative_distribution(hist)
    new_img = gray_values(cdf, img)

    cv.imshow("Kitty", new_img)
    cv.waitKey()
