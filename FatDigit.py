import cv2 as cv
import numpy as np


def create_size(sizes,i, images):
    img = images[i]
    kernel = sizes[i]
    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel,kernel))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    erosion = cv.erode(thresh1, kern, iterations=1)
    _, contours1, _ = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    while len(contours1) != 0:
        kernel += 1
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel, kernel))
        erosion = cv.erode(thresh1, kern, iterations=1)
        _, contours1, _ = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return kernel
def fat_index(A):
    return A.index(max(A))
thick = []
image = cv.imread("FatDig.png")
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 100, 255, cv.THRESH_BINARY_INV)
_,contours,_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
chars = []
sizes = []
for c in contours:
    mask = np.zeros(image.shape, np.uint8)
    mask.fill(255)
    chars.append(mask)
for i in range(len(chars)):
    cv.drawContours(chars[i],contours,i,(0,0,0),-1)
    ker = 1
    sizes.append(ker)
for i in range(len(chars)):
    thick.append(create_size(sizes,i,chars))
cv.imshow("Finally", chars[fat_index(thick)])
cv.waitKey(0)
cv.destroyAllWindows()
