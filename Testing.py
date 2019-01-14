import cv2
import numpy as np
import detectLines
import vertical
import roi
import rectangle

try:
    img =  cv2.imread(input("Enter the name of the image : "))
    #print(img.shape)

    # Resize image to appt proportion
    resized_img = rectangle.resize(img)

    # Extract table and change Perspective
    rect_img = rectangle.draw(resized_img)

    # Detect Horizontal Lines
    horizontal_lines = detectLines.detect(rect_img)
    # Detect Vertical Lines
    vertical_lines = vertical.detect(rect_img)

    # Create Cells and Call predictor on each cell
    roi.create_row(rect_img,horizontal_lines,vertical_lines)


    #cv2.imshow("Rectangle",rect_img)
    #cv2.imshow("Original",img)

except AttributeError:
    print("Enter valid Image name")

#cv2.waitKey(0)
cv2.destroyAllWindows()
