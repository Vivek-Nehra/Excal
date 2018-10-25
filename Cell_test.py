import cv2
import numpy as np
import pytesseract
import tm
def printed_text(imging,model):
    # try:
    img = cv2.imread(imging)
    # print(img.shape)
    copy = img.copy()
    cv2.bilateralFilter(img,9,75,75)
    # img = cv2.resize(img,None,fx=1.5,fy=1.5,interpolation = cv2.INTER_CUBIC)
    # img = cv2.Canny(img,100,200)
    # _,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    # img = cv2.dilate(img,np.ones([3,3]),iterations=1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    row,col=img.shape[:]
    # img = img[int(row*0.1):int(row*0.9),int(col*0.1):int(col*0.9)]

    # # _,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)

    # img = cv2.dilate(img,np.ones([5,5]),iterations= 1)
    # img = cv2.erode(img,np.ones([3,3]),iterations= 1)



    #cv2.imshow("Image",copy)
    printed = pytesseract.image_to_string(img)
    #print(printed,end=' ')
    if (len(printed)==0) :
        return tm.predict(img,model)
    else :
        return printed
    counter = 0
    # if cv2.waitKey(0)==ord('w'):
        # cv2.imwrite("printed" + str(counter) + ".jpg",img)
        # counter += 1


    # except :
    #     print("Invalid Name")

# printed_text(input("Name : "))
