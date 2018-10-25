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


#(X_train,y_train),(X_test,y_test) = mnist.load_data()
"""
for i in range(0,60000):
    X_train[i] = cv2.Canny(X_train[i],100,200)

"""

def nothing(x):
    pass




def predict(img,model) :
    #print("here")
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    copy = img.copy()
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img = cv2.Canny(img,100,200)
    non_zero_cells = cv2.countNonZero(cv2.dilate(img.copy(),np.ones([3,3]),iterations=1))
    digit_prob = non_zero_cells /784
    #print(img)
    #img = cv2.Canny(img,100,200)
    #print(img)
    # canny = cv2.dilate(canny,np.ones([3,3]),iterations=1)
    # cv2.imshow('canny',img)
    # _, c, h = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("Contours length : " ,len(c))
    # cMax = max(c, key = cv2.contourArea)
    # cv2.drawContours(img,c,-1,(0,255,0,),3)
    # x,y,w,h = cv2.boundingRect(cMax)
    # cv2.rectangle(img,(x,x+h),(y,y+h),(0,0,255),2)
    #cv2.imshow('l',img)
    # cv2.waitKey(0)
    # img = img[x:x+h,y:y+w]
    # cv2.imshow('s',img)
    # img = cv2.erode(img,np.ones([2,2]),iterations=1)
    # img = cv2.dilate(img,np.ones([3,3]),iterations=1)
    #cv2.imshow('sample1',img)
    #img = cv2.dilate(img,np.ones([3,3]),iterations = 1)

    #img = cv2.resize(img,(28,28))
    #cv2.imshow('sample',img)
    # cv2.waitKey(0)
    """

    window1 = cv2.namedWindow('trackbar1',cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('thresh1','trackbar1',0,255,nothing)

    #window2 = cv2.namedWindow('trackbar2',cv2.WINDOW_AUTOSIZE)

    #cv2.createTrackbar('thresh2','trackbar2',0,255,nothing)


    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #gray = cv2.blur(gray,(7,7))

    #gray = cv2.GaussianBlur(gray,(7,7),0)



    while(1) :

        val1 = cv2.getTrackbarPos('thresh1','trackbar1')

        #val2 = cv2.getTrackbarPos('thresh2','trackbar2')

        #canny = cv2.Canny(img,val1,val2)
        _,thresh = cv2.threshold(img,val1,255,cv2.THRESH_BINARY_INV)

        cv2.imshow('thresh',thresh)
        if (cv2.waitKey(1) == 27 ) :
            break
    """

    #canny = cv2.dilate(canny,np.ones([3,3]),iterations= 1)

    """ @@@
    canny = cv2.Canny(img,100,200)

    #canny = cv2.dilate(canny,np.ones([3,3]),iterations= 1)

    cv2.imshow('2',canny)



    rows,cols = canny.shape

    #print (canny.shape)
    #canny = canny.reshape(1, 1, 28, 28).astype('float32')
    canny = canny.reshape(1,28*28).astype('float32')
    canny = canny/255
    #print(canny)

    @@@ """
    """
    #_,thresh = cv2.threshold(img,196,255,cv2.THRESH_BINARY_INV)


    rows,cols = thresh.shape

    print (thresh.shape)
    #canny = canny.reshape(1, 1, 28, 28).astype('float32')
    thresh = thresh.reshape(1,28*28).astype('float32')
    thresh = thresh/255
    print(thresh)

    """
    if digit_prob < 0.4:
        return ""
    else:
        img=img.reshape(1,1, 28, 28).astype('float32')
        #img = img.reshape(1,28*28).astype('float32')
        img = img/255

        #model = load_model('Without_Canny.h5')
        #model = pickle.load(open("208epochs_weights_model_9.pkl","rb"))
        #weights=model.get_weights()

        predicted = model.predict(img,batch_size = 200,verbose = 2,steps = None)
        max_val = -1
        #print (predicted)
        for i in range(10):
            if (predicted[0][i]>max_val):
                max_val=predicted[0][i]
                ans = i

        #print (ans)
        return str(ans)
        #print (weights)

    #img = cv2.imread(input("Enter the name of image : "))


    #predict(img)

    """
    for i in range(X_train.shape[0]):
        predict(X_train[i])
    """
