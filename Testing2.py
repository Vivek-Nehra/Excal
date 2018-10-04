import cv2
import numpy as np
import math

img = cv2.imread('Img.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

_,contours,heirarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,0),2)

cv2.imshow("Original",img)
cv2.imshow("Canny",edges)
#cv2.imshow("Erode",erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
