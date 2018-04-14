# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:57:07 2018

@author: Nikola
"""
# to run this code you need to be in the folder where all the classes needed for this code to function are
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from tensorflow.examples.tutorials.mnist import input_data
from SVM import svm
from SVM_number_recognition import svm_num_recognition

sv = svm_num_recognition(kernel = 'poly', C = 1, degree = 3, sigma = 1, threshold = 1e-8)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)# mnist contains all the datasets (train,test,validatio images and labels)
                                                       # when running this line the first time it will take some time beacause it takes some time for data sets to be downloaded

sv.svm_train(mnist.train.images[0:5000], mnist.train.labels[0:5000])# changing the number of images for training data can give better results, but training the network takes longer

sv.svm_validation(mnist.test.images[0:10000], mnist.test.labels[0:10000])#this gives a confusion matrix that tells us how god the classifier classifed the data

sv.svm_one_num_classification(mnist.test.images[1101])# this here prints out the number that has been given as a parameter
d=np.reshape(mnist.test.images[1101],(28,28)) # this is a confirmation of a previous classification
pt.imshow(d,cmap='Greys_r')
