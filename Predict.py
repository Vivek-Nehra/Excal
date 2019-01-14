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
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img = cv2.Canny(img,100,200)
    non_zero_cells = cv2.countNonZero(cv2.dilate(img.copy(),np.ones([3,3]),iterations=1))
    digit_prob = non_zero_cells /784            # Remove Empty Cells

    if digit_prob < 0.4:
        return ""       # Cell is empty
    else:
        img=img.reshape(1,1, 28, 28).astype('float32')      # Cell is not empty
        #img = img.reshape(1,28*28).astype('float32')
        img = img/255

        #model = load_model('Without_Canny.h5')
        weights=model.get_weights()         # Get weights

        predicted = model.predict(img,batch_size = 200,verbose = 2,steps = None)            # Predict
        max_val = -1
        #print (predicted)
        for i in range(2):
            if (predicted[0][i]>max_val):
                max_val=predicted[0][i]
                ans = i

        #print (ans)
        return str(ans)
