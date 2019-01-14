import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import cv2
import pickle


def nothing(x):
    pass




def predict(img,model) :
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img,(5,5),True)
    img = cv2.Canny(img,100,200)
    dilate = cv2.dilate(img,(3,3),iterations = 1)


    # print("Before")
    # cv2.imshow("Empty",img)
    # cv2.waitKey(0)
    

    _, c, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(c) > 0:
        cMax = max(c, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cMax)        # Find the Bounding Rectangle
        img = img[y:y+h,x:x+w] 


        top , bottom , left , right = 0,0,0,0
        small_row,small_col = img.shape
        
        if small_row < 28:
            top = int((28-small_row)/2)
            bottom = 28 - top - small_row

        if small_col < 28:
            left = int((28-small_col)/2)
            right = 28 - left - small_col

        img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value = [0,0,0])

        if small_row > 28 or small_col > 28:
            # print(img.shape)
            img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
            # img = cv2.Canny(img,100,200)


        # print("After")
        # cv2.imshow("Empty",img)
        # cv2.waitKey(0)



    non_zero_cells = cv2.countNonZero(cv2.dilate(img.copy(),np.ones([3,3]),iterations=1))
    digit_prob = non_zero_cells /784            # Remove Empty Cells

    if digit_prob < 0.35:
        # cv2.imshow("Empty",img)
        # print(digit_prob)
        # cv2.waitKey(0)
        return ""       # Cell is empty
    else:
        img=img.reshape(1,1, 28, 28).astype('float32')      # Cell is not empty
        #img = img.reshape(1,28*28).astype('float32')
        img = img/255

        #model = load_model('Without_Canny.h5')
        weights=model.get_weights()         # Get weights

        predicted = model.predict(img,batch_size = 200,verbose = 2,steps = None)            # Predict
        max_val = -1
        # print (predicted)

        for i in range(2):
            if (predicted[0][i]>max_val):
                max_val=predicted[0][i]
                ans = i

        # print (ans)
        return str(ans)
