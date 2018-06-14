import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def take_index(a,b,c,d):
    areas = [(a[i] * b[i], i) for i in range(len(a))]
    return sorted(areas)[-1][1]


img = cv.imread("XXX.XXX")

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
lower_cyan = np.array([80,100,100])
upper_cyan = np.array([100,255,255])
ret, thresh = cv.threshold(imgray,100,255,cv.THRESH_TOZERO)
thresh = cv.bitwise_not(thresh)
im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)


X = []
Y = []
W = []
H = []
for c in contours:
      x,y,w,h = cv.boundingRect(c)
      cv.rectangle(img, (x, y), (x+w, y+h), (255,255,255))
      X.append(x)
      Y.append(y)
      W.append(w)
      H.append(h)
x_c = X[take_index(W,H,X,Y)]
y_c = Y[take_index(W,H,X,Y)]
w_c = W[take_index(W,H,X,Y)]
h_c = H[take_index(W,H,X,Y)]
mask = np.zeros(img.shape, np.uint8)
mask.fill(255)
cv.drawContours(mask, contours, take_index(W,H,X,Y), (0,0,255),-1)
img = cv.add(mask,img)
print(take_index(W,H,X,Y))
plt.imshow(img)
plt.show()
