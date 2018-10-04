import cv2
import numpy as np
import math

def nothing(x):
    pass

def deletelines(lines):
    for line in lines:
        for index,line2 in enumerate(lines):
            if abs(line[0][1] - line2[0][1]) < 20:
                #print (abs(line[0][1] - line2[0][1]))
                del lines[index]
    return lines

img =  cv2.imread("att.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(img,100,200)

cv2.GaussianBlur(canny,(15,15),1)
print(canny.shape,canny.dtype)

sobel = cv2.Sobel(canny,cv2.CV_64F,0,1,ksize=5)
#sobel = cv2.dilate(sobel,np.ones([3,3]),iterations=1)
sobel = np.array(sobel,dtype=np.uint8)

sobel = cv2.dilate(sobel,np.ones((5,5)),iterations=1)
print(sobel.shape,sobel.dtype)




window = cv2.namedWindow('trackbar',cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('thresh','trackbar',0,1500,nothing)

while(1) :
    val = cv2.getTrackbarPos('thresh','trackbar')
    lines = cv2.HoughLinesP(sobel,1,np.pi/180,25,None,val,5).tolist()
    img = cv2.imread('att.png')
    if lines is not None:
        lines.sort(key= lambda x : x[0][1])
        #print(lines)
        lines = deletelines(lines)
    for a in lines:
        cv2.line(img,(a[0][0],a[0][1]),(a[0][2],a[0][3]),(0,255,0),2)     
    cv2.imshow('image',img)
    if cv2.waitKey(1)== 27 :
        break




cv2.imshow("Original",img)
cv2.imshow("Sobel",sobel)
#cv2.imshow("Canny",canny)
#cv2.imshow("Erode",erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
