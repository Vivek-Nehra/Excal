import cv2
import numpy as np
import detectLines
import vertical
import roi
import rectangle

try:
    img =  cv2.imread(input("Enter the name of the image : "))
    print(img.shape)
    #cv2.imshow('img',img)
    small_img = rectangle.resize(img)
    #cv2.imshow('small_img',small_img)

    rect_img = rectangle.draw(small_img)
    horizontal_lines = detectLines.detect(rect_img)
    vertical_lines = vertical.detect(rect_img)
    roi.create_row(rect_img,horizontal_lines,vertical_lines)


    cv2.imshow("Rectangle",rect_img)
    cv2.imshow("Original",img)

except AttributeError:
    print("Enter valid Image name")

except NoneTypeError:
    print("Error Occured .. Please Try Again")
cv2.waitKey(0)
cv2.destroyAllWindows()
