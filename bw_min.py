import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("Dig.png")

def take_index(a,b):
    areas = [(a[i]*b[i],i) for i in range(len(a))]
    return areas.index(min(areas))
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(imgray,100,255,cv.THRESH_BINARY_INV)
im2,contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
W = []
H = []
for c in contours:
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(c,(x,y),(x+h,y+h), (255,255,255))
    W.append(w)
    H.append(h)
mask = np.zeros(img.shape,np.uint8)
mask.fill(255)
cv.drawContours(mask, contours,take_index(W,H),(0,0,0),-1)
cv.imshow("Result", mask)
cv.waitKey(0)
cv.destroyAllWindows()
