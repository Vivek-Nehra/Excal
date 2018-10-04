import cv2
import numpy as np
import detectLines
import roi
import rectangle

img =  cv2.imread("co5.png")

cv2.GaussianBlur(img,(15,15),1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img,100,200)
cv2.GaussianBlur(canny,(15,15),1)
# print(canny.shape,canny.dtype)

sobel = cv2.Sobel(canny,cv2.CV_64F,0,1,ksize=5)
sobel = cv2.dilate(sobel,np.ones([3,3]),iterations=1)
sobel = np.array(sobel,dtype=np.uint8)
# print(sobel.shape,sobel.dtype)

lines = detectLines.detect(img,sobel)
roi.create_roi(img,lines)
rect_img = rectangle.draw(img,canny)

cv2.imshow("Rectangle",rect_img)
cv2.imshow("Original",img)
# cv2.imshow("Sobel",sobel)
# cv2.imshow("Canny",canny)


cv2.waitKey(0)
cv2.destroyAllWindows()
