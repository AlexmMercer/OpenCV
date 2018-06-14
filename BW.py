import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def take_index(a,b,c,d):
    areas = [(a[i] * b[i], i) for i in range(len(a))]
    return sorted(areas)[-1][1]
img = cv.imread('ABC.png')

imgray = np.average(img,axis=2)

ret,thresh = cv.threshold(imgray,100,255,cv.THRESH_BINARY_INV)

#thresh.reshape((imgray.shape[0],imgray.shape[1],1))*np.array([1.,1.,1.]))

thresh = thresh.astype(np.uint8)

im2,contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
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
print(take_index(W,H,X,Y))
mask = np.zeros(img.shape, np.uint8)
mask.fill(255)
cv.drawContours(mask,contours,take_index(W,H,X,Y), (0,0,0),-1)
print(imgray.shape)
print(img.shape)
print(mask.shape)
#fin_img = np.hstack((mask,img))
final = img + mask
plt.imshow(final)
plt.show()
