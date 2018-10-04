import cv2
import numpy as np


def deletelines(lines):
    for line in lines:
        for index,line2 in enumerate(lines):
            if abs(line[0][1] - line2[0][1]) < 20:
                #print (abs(line[0][1] - line2[0][1]))
                del lines[index]
    return lines

def detect(img,sobel):
    row,col = sobel.shape

    lines = cv2.HoughLinesP(sobel,1,np.pi/180,150,None,col/2,30)
    #print(len(lines))
    val = 0
    if lines is not None:
        lines = lines.tolist()
        lines.sort(key= lambda x : x[0][1])
        #print(lines)
        lines = deletelines(lines)

        print(len(lines))
        for line in lines:
            # if abs(line[0][1] - line[0][3]) < 10:
            val +=1
            cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),5)
        print(val)

        return lines