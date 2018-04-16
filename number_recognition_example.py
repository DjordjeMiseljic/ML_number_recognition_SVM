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

# Testiranje uspjesnosti
percentage = np.zeros([39])
j=1;
for i in range(0,39):
    sv = svm_num_recognition(kernel = 'poly', C = j, degree = 3, sigma = 1, threshold = 1e-7)
    sv.svm_train(mnist.train.images[0:1000], mnist.train.labels[0:1000])
    percentage[i] = sv.svm_validation(mnist.test.images[0:3000], mnist.test.labels[0:3000])
    j+=0.05
print ("maximum value:", np.max(percentage),"for C=", (np.argmax(percentage)*0.05+1))
print ("minimum value:", np.min(percentage),"for C=", (np.argmin(percentage)*0.05+1))        
#percentage[41]
x= np.arange(1,2.95,0.05)
pt.ion()
pt.figure()
pt.plot(x,percentage)
pt.xlabel("C values")
pt.ylabel("percentage [%]")
pt.show

##############################################
X=np.array([[1,2,3],[4,5,6],[7,8,9]])
z=np.array([0,0,0])
for i in range(0,3):
  z+=X[i]
g=z.T+1