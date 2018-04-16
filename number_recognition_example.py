"""
Created on Mon Apr  9 20:57:07 2018

@author: Kole
"""
##############################################
#LIBRARIES

import numpy as np
import matplotlib.pyplot as pt
from tensorflow.examples.tutorials.mnist import input_data
from SVM_number_recognition import svm_num_recognition
import cv2

##############################################
# DESKEW FUNCTION

def deskew(img_single):
    img=np.reshape(img_single,(28,28))
    SZ=28
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return np.reshape(img,(1,784))

##############################################
# DESKEW DATASET

def deskew_dataset(dataset):
    ds=dataset.copy()
    for i in range (0,np.shape(ds)[0]):
        ds[i]=deskew(dataset[i])
    return ds

##############################################
# IMPORTING DATASET
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#############################################
# TESTING 
train_num = 2000
test_num = 1000
train_data = mnist.train.images[0:train_num]
train_labels = mnist.train.labels[0:train_num] 
test_data = mnist.test.images[0:test_num]
test_labels = mnist.test.labels[0:test_num]

sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(train_data, train_labels)
a = sv.svm_validation(test_data, test_labels)


sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(deskew_dataset(train_data),train_labels)
a = sv.svm_validation(deskew_dataset(test_data), test_labels)

#############################################
# CLASSIFYING SINGLE IMAGE 

num=1101
sv.svm_one_num_classification(mnist.test.images[num]) #1101,1107,1116,1119,200,175
d=np.reshape(mnist.test.images[num],(28,28)) 
pt.imshow(d,cmap='Greys_r')

#############################################
# TESTING DIFFERENT VALUES OF C

percentage = np.zeros([100])
j=0;
for i in range(0,30):
    j+=0.1
    sv = svm_num_recognition(kernel = 'poly', C = j, degree = 3, sigma = 1, threshold = 1e-7)
    sv.svm_train(mnist.train.images[0:200], mnist.train.labels[0:200])
    percentage[i] = sv.svm_validation(mnist.test.images[0:1000], mnist.test.labels[0:1000])      
percentage[np.argmax(percentage)]

#############################################
# TESTING DESKEW FUNCTIONS
    
z=deskew(mnist.test.images[4699])
sv.svm_one_num_classification(z)
zz=np.reshape(z,(28,28))
pt.imshow(zz,cmap='Greys_r')




