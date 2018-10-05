import cv2
import numpy as np
import detectLines
import roi
import rectangle


img =  cv2.imread("co5.png")

#lines = detectLines.detect(img)
#roi.create_roi(img,lines)
rect_img = rectangle.draw(img)
lines = detectLines.detect(rect_img)
#roi.create_roi(rect_img,lines)


cv2.imshow("Rectangle",rect_img)
cv2.imshow("Original",img)
# cv2.imshow("Sobel",sobel)
# cv2.imshow("Canny",canny)


cv2.waitKey(0)
cv2.destroyAllWindows()
