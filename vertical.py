import cv2
import numpy as np
from rectangle import edge_detect
import math

def deletelines(lines):
    for line in lines:
        for index,line2 in enumerate(lines):
            if abs(line[0][0] - line2[0][0]) < 10 or slope(line) < 80:
                #print (abs(line[0][1] - line2[0][1]))
                del lines[index]
    return lines



def slope(line):
    if line[0][0] == line[0][2]:
        return 90
    return math.degrees(math.atan(abs(line[0][1] - line[0][3])/abs(line[0][0] - line[0][2])))



def detect(img):
    canny = edge_detect(img)
    sobel = cv2.Sobel(canny,cv2.CV_64F,1,0,ksize=5)
    row,col = sobel.shape

    sobel = cv2.erode(sobel,np.ones([int(row/10),1]),iterations=1)
    sobel = cv2.dilate(sobel,np.ones([int(row/2),1]),iterations=1)
    sobel = np.array(sobel,dtype=np.uint8)
    # print(sobel.shape,sobel.dtype)

    lines = cv2.HoughLinesP(sobel,1,np.pi/180,150,None,row/2,30)
    # print("Initial lines : " , len(lines))
    val = 0
    if lines is not None:
        lines = lines.tolist()
        lines.sort(key= lambda x : x[0][1])
        print(len(lines))
        lines = deletelines(lines)

        for line in lines:

            if line[0][0] == line[0][2]:
                theta = 90
            else :
                theta = slope(line)
            print(theta)
            if theta > 80:
                cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)
                val+=1
        print(val)


        cv2.imshow("Sobel",sobel)
        cv2.imshow("Canny",canny)
        cv2.waitKey(0)

        return lines
