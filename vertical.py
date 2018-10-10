import cv2
import numpy as np
from rectangle import edge_detect
import math


def deletelines(lines):
    n = len(lines)
    idx= 0
    while idx < n:

        if slope(lines[idx]) < 80 :
            del lines[idx]
            n -= 1
            continue

        idx2 = 0
        while idx2 < n:
            if lines[idx] == lines[idx2]:
                # print(lines[idx],lines[idx2],"Continuing")
                idx2 += 1
                continue

            if abs(lines[idx][0][0] - lines[idx2][0][0]) < 20 :
                # print ("Deleted")
                del lines[idx2]
                n -= 1
                continue

            idx2 += 1
        idx += 1

    return lines



def slope(line):
    if line[0][0] == line[0][2]:
        return 90
    return math.degrees(math.atan(abs(line[0][1] - line[0][3])/abs(line[0][0] - line[0][2])))



def detect(img):
    canny = edge_detect(img)

    sobel = cv2.Sobel(canny,cv2.CV_64F,1,0,ksize=5)
    row,col = sobel.shape

    sobel = cv2.erode(sobel,np.ones([int(row/15),1]),iterations=1)
    sobel = cv2.dilate(sobel,np.ones([int(row/1.5),1]),iterations=1)
    sobel = np.array(sobel,dtype=np.uint8)
    _,sobel = cv2.threshold(sobel,200,255,cv2.THRESH_BINARY)
    # print(canny.shape)

    lines = cv2.HoughLinesP(sobel,1,np.pi/180,150,None,int(0.3*row),30)
    # print("Initial lines : " , len(lines))
    # val = 0
    if lines is not None:
        lines = lines.tolist()
        lines.sort(key= lambda x : x[0][0])
        print("Initial Lines : ",len(lines))
        lines = deletelines(lines)
        print("After deletion : ",len(lines))

        for line in lines:
            # print(line)
            cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)

            if line[0][0] == line[0][2]:
                theta = 90
            else :
                theta = slope(line)
            # print(theta)
            if theta > 80:
                cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)
                # val+=1
        # print(val)

        cv2.imshow("Test",img)
        cv2.imshow("Sobel",sobel)
        cv2.imshow("Canny",canny)
        cv2.waitKey(0)

        return lines
