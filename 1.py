import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnistS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

import cv2

(X_train,y_train),(X_test,y_test) = mnist.load_data()

"""
plt.subplot(331)    ## nrows ncols nplot , divides the axis into nrows*ncols , nplot is plot no. associated with image
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.show()

"""


#print (X_train[0])    ## Prints the gray scale value of each pixel for image 1 (a 2D matrix 28*28 (image_width*image_height))


# Since each image in X_train has 2D matrix values , therefore, as whole X_train will be a 3D matix.
# 3D matrix is therefore : instance * image_width * image_height.
# Now we need to convert this 2D matrix of images into Vector.

# Reshaping 2D matrix of each images into vector.

"""
print (X_train[0].shape)    # Output is (28,28).

print (X_train.shape[0])    # Output is 60000. (Instance or number of images)

print (X_train.shape[1])   # Output is 28.  (Width of image)

print (X_train.shape[2])    # Output is 28  (Height of image)

print (X_train.shape[3])    # Output is Tuple Index is out of range.Since X_train is 3D matrix not 4D matrix.

# Means because X_train is an 3D matrix , therefore , its shape is (60000,28,28)
# and since X_train[0] , which is an image is 2D matrix , its shape is (28,28) 
"""

for i in range(0,60000):
    X_train[i] = cv2.Canny(X_train[i],100,200)
    #cv2.imshow('img',X_train[i])
    #if (cv2.waitKey(1) ==27 ) :
        #break 

for i in range(0,10000):
    X_test[i] = cv2.Canny(X_test[i],100,200)

#print (X_train[0])

num_pixels = X_train.shape[1]*X_train.shape[2]  # num_pixels = image_width*image_height

X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')  # Converting to (60000,784) matrix 
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')


print (X_train[0].shape)    # Output is (784,_).

print (X_train.shape[0])    # Output is 60000.

print (X_train.shape[1])   # Output is 784.

#print (X_train.shape[2])    # Output is Out of range


# Converting all values to 0-1 range , form 0-255 gray scale range. 

X_train = X_train/255
X_test = X_test/255

# Since the Output is Multi class , vary from 0-9
# Therefore , we need to convert them into binary matrix.
# 0 will be 1,0,0,0,0,0,0,0,0,0
# 1 will be 0,1,0,0,0,0,0,0,0,0
# 2 will be 0,0,1,0,0,0,0,0,0,0
# 3 will be 0,0,0,1,0,0,0,0,0,0
# 4 will be 0,0,0,0,1,0,0,0,0,0
# 5 will be 0,0,0,0,0,1,0,0,0,0
# 6 will be 0,0,0,0,0,0,1,0,0,0
# 7 will be 0,0,0,0,0,0,0,1,0,0
# 8 will be 0,0,0,0,0,0,0,0,1,0
# 9 will be 0,0,0,0,0,0,0,0,0,1
"""
print (y_train.shape)   # Output : (60000,_)
"""
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

"""
print (y_train[0])  # Output : [0,0,0,0,0,1,0,0,0,0]  representing 5

print (y_train.shape) # Output : (60000,10)

print (y_train.shape[0]) # Output : (60000)

print (y_train.shape[1]) # Output : (10)
"""

num_classes = y_test.shape[1]   # numclasses = 10

#num_classes = y_train.shape[1] could also be used , simply num_classes = 10 , if we know before.

def baseline_model() :
    
    model = Sequential()

    model.add(Dense(num_pixels,input_dim = num_pixels,kernel_initializer='normal',activation='relu'))
    #model.add(Dense(num_pixels,input_dim = num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))

    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

    return model




model = baseline_model()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=128,verbose=2)
scores = model.evaluate(X_test,y_test,verbose=0)
print ("Baseline error is %.2f%%" % (100-scores[1]*100))

model.save('1.h5')
weights = model.get_weights()

print (weights)




