import cv2
import numpy as np
import math


def deletelines(lines):
    for line in lines:
        for index,line2 in enumerate(lines):
            if abs(line[0][1] - line2[0][1]) < 20:
                #print (abs(line[0][1] - line2[0][1]))
                del lines[index]
    return lines



def create_roi(lines):
    for index,line in enumerate(lines) :
        if index == len(lines)-1:
            break
        roi = img[line[0][1]:lines[index+1][0][1],:,:]
        row,col,_ = roi.shape
        if row > 20:
            cv2.imshow("ROI",roi)
            cv2.waitKey(0)








img =  cv2.imread("co5.png")
cv2.GaussianBlur(img,(15,15),1)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(img,100,200)

cv2.GaussianBlur(canny,(15,15),1)
print(canny.shape,canny.dtype)

sobel = cv2.Sobel(canny,cv2.CV_64F,0,1,ksize=5)
sobel = cv2.dilate(sobel,np.ones([3,3]),iterations=1)
sobel = np.array(sobel,dtype=np.uint8)
print(sobel.shape,sobel.dtype)
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

    create_roi(lines)

cv2.imshow("Original",img)
cv2.imshow("Sobel",sobel)
cv2.imshow("Canny",canny)
#cv2.imshow("Erode",erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
