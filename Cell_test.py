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


    row,col=img.shape[:]

    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
    # img = cv2.erode(img,np.ones([3,3]),iterations= 1)
    #cv2.imshow("Image",copy)

    printed = pytesseract.image_to_string(img)
    if (len(printed)==0) :
        return Predict.predict(img,model)            # Hand-written digits not recognised by Tesseract
    else :
        return printed          # Cell contains printed digits
