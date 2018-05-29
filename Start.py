import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def take_index(a,b,c,d):
    areas = [(a[i] * b[i], i) for i in range(len(a))]
    return sorted(areas)[-1][1]
img = cv.imread('color.png')

imgray = np.average(img,axis=2)

ret,thresh = cv.threshold(imgray,100,255,cv.THRESH_BINARY_INV)

#thresh.reshape((imgray.shape[0],imgray.shape[1],1))*np.array([1.,1.,1.]))


thresh = thresh.astype(np.uint8)


im2,contours, hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,contours,-1,(0,255,0),3)
X = []
Y = []
W = []
H = []
plt.imshow(img)
plt.show()
for c in contours:
      x,y,w,h = cv.boundingRect(c)
      rect = cv.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
      X.append(x)
      Y.append(y)
      W.append(w)
      H.append(h)
#plt.imshow(img)
x_c = X[take_index(W,H,X,Y)]
y_c = Y[take_index(W,H,X,Y)]
w_c = W[take_index(W,H,X,Y)]
h_c = H[take_index(W,H,X,Y)]
cut_image = img[y_c:y_c+h_c, x_c:x_c+w_c]

#cv.drawContours(cut_image,contours,(0,0,255))
print(take_index(W,H,X,Y))
plt.imshow(cut_image)
plt.show()
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print(img.shape)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
#print(w*h)
plt.show()



