import cv2
import numpy as np
import pytesseract
import Predict

def printed_text(imging,model):     # Recognise Printed Text

    img = cv2.imread(imging)
    # print(img.shape)
    copy = img.copy()
    cv2.bilateralFilter(img,9,75,75)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Final",img)

    mask = np.zeros([img.shape[0]-8,img.shape[1]-8],dtype=np.uint8)
    mask = cv2.copyMakeBorder(mask,4,4,4,4,cv2.BORDER_CONSTANT,value = [255,255,255])
    img = cv2.bitwise_or(img,mask)


    row,col=img.shape[:]


    copy = img.copy()
    canny = cv2.Canny(copy,100,200)
    _, c, h = cv2.findContours(cv2.dilate(canny.copy(),(7,7),iterations=1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(c) <= 0:
    	return ""

    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
    # img = cv2.erode(img,np.ones([3,3]),iterations= 1)
  
    printed = pytesseract.image_to_string(img)

    if (len(printed)==0) :
        return Predict.predict(img,model)            # Hand-written digits not recognised by Tesseract
    else :
    	# cv2.imshow("Image",img)
    	# cv2.waitKey(0)
    	# print(printed)
    	return printed          # Cell contains printed digits