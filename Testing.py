import cv2
import numpy as np
import detectLines
import vertical
import roi
import rectangle


img =  cv2.imread(input("Enter the name of the image : "))

cv2.imshow("Image",img)
img = rectangle.resize(img)
# lines = detectLines.detect(img)
# roi.create_row(img,lines)
rect_img = rectangle.draw(img)
lines = detectLines.detect(rect_img)
# lines2 = vertical.detect(rect_img)
roi.create_row(rect_img,lines)



cv2.imshow("Rectangle",rect_img)
cv2.imshow("Original",img)


cv2.waitKey(0)
cv2.destroyAllWindows()
