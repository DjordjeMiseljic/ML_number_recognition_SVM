"""
Created on Mon Apr  9 20:57:07 2018

@author: Nikola
"""
# to run this code you need to be in the folder where all the classes needed for this code to function are
import numpy as np
import matplotlib.pyplot as pt
from tensorflow.examples.tutorials.mnist import input_data
from SVM_number_recognition import svm_num_recognition




mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(mnist.train.images[0:10000], mnist.train.labels[0:10000])
a = sv.svm_validation(mnist.test.images[0:10000], mnist.test.labels[0:10000])

num=9980
sv.svm_one_num_classification(mnist.test.images[num])

#1101,1107,1116,1119,200,175
d=np.reshape(mnist.test.images[num],(28,28)) 
pt.imshow(d,cmap='Greys_r')

# ovo sam pokusavao da testiram kako se menja procenat uspesnosti kada menjam C
percentage = np.zeros([100])
j=0;
for i in range(0,100):
    j+=0.1
    sv = svm_num_recognition(kernel = 'poly', C = j, degree = 3, sigma = 1, threshold = 1e-7)
    sv.svm_train(mnist.train.images[0:200], mnist.train.labels[0:200])
    percentage[i] = sv.svm_validation(mnist.test.images[0:1000], mnist.test.labels[0:1000])

np.argmin(percentage)        
percentage[41]

##############################################
X=np.array([[1,2,3],[4,5,6],[7,8,9]])
z=np.array([0,0,0])
for i in range(0,3):
  z+=X[i]
g=z.T+1