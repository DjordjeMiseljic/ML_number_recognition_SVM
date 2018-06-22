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
from our_deskew import our_deskew
from test import imageprepare
from PIL import Image, ImageFilter
#############################################
# IMPORTING DATASET
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#############################################
# TRAINIG + TESTING 

train_num = 1000
test_num = 100
train_data = mnist.train.images[0:train_num]
train_labels = mnist.train.labels[0:train_num] 
test_data = mnist.test.images[0:test_num]
test_labels = mnist.test.labels[0:test_num]

#naked
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(train_data, train_labels)
a = sv.svm_validation(test_data, test_labels)

#using auto deskew from class "our_deskew"
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
deskew = our_deskew();
sv.svm_train(deskew.deskew_dataset_auto(train_data),train_labels)
a = sv.svm_validation(deskew.deskew_dataset_auto(test_data), test_labels)

#using manual deskew from class "our_deskew"
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
deskew = our_deskew()
sv.svm_train(deskew.deskew_dataset_manual(train_data),train_labels)
a = sv.svm_validation(deskew.deskew_dataset_manual(test_data), test_labels)
a = sv.svm_validation((test_data), test_labels)# with this we get files for c++ code



## indexes where classified numbers differ from the labeled ones
mistakes = np.where(test_labels!=sv.classified)

#############################################
# CLASSIFYING SINGLE IMAGE 

num=5444

print("Normal classification")
sv.svm_one_num_classification(mnist.test.images[num])
d=np.reshape(mnist.test.images[num],(28,28)) 
pt.figure()
pt.imshow(d,cmap='Greys_r')

print("Classification with deskew")
sv.svm_one_num_classification(deskew.deskew_manual(mnist.test.images[num]))
d=np.reshape(deskew.deskew_manual(mnist.test.images[num]),(28,28)) 
deskew.mu02
deskew.mu11

pt.figure()
pt.imshow(d,cmap='Greys_r')

#############################################

test = open("y.txt",'w')
np.savetxt(test,mnist.test.images[num:5446],fmt='%.12e')
test.close()

#### testing picture deskewed in c++
number = np.zeros([1,784]) 
file_handle = open('number.txt', 'r') 
for i in range (0,784): 
    number[0,i] = float(file_handle.readline() ) 
number = np.reshape(number,(28,28))
pt.imshow(number,cmap='Greys_r')     
#######################################################

