import numpy as np
import cv2 as cv

def fat_index(a):
    max = a[0]
    for i in range(len(a)):
        if max < a[i]:
            max = a[i]
    return a.index(max)

from matplotlib import pyplot as plt
img = cv.imread("FD.png")
kernel = np.ones((3,3), np.uint8)
imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray,100,255,cv.THRESH_BINARY_INV)
#cv.drawContours(img,contours,-1,(0,0,255),-1)
im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnt =  contours[0]
cnt1 = contours[1]
Areas = []
perimeter = cv.arcLength(cnt,True)
perimeter1 = cv.arcLength(cnt1,True)
Areas.append(perimeter)
Areas.append(perimeter1)
mask = np.zeros(img.shape, np.uint8)
mask.fill(255)
cv.drawContours(mask, contours,fat_index(Areas),(0,0,0),-1)
cv.imshow("Result", mask)
cv.waitKey(0)
cv.destroyAllWindows()
