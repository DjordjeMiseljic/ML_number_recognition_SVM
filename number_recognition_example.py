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
from SVM import svm
#from numba import jit, autojit

##############################################
#function for speeding up the training
"""
@jit
def fast_svm_train(train_data, train_labels,test_data,test_labels):
    sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
    sv.svm_train(train_data,train_labels)
    a = sv.svm_validation(deskew_dataset(test_data), test_labels)
"""
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
# CALCULATE MOMENTS MU02 and MU11

def calc_moments(img_single):
    img=np.reshape(img_single,(28,28))
    SIZE=28
    #calculate spatial moments
    m00=0;
    m10=0;
    m01=0  
    for i in range (0,SIZE):
      for j in range (0,SIZE):
        m00 += (img[i,j]) 
        m10 += (img[i,j]*i)
        m01 += (img[i,j]*j)
    #calculate mass center
    x_mc=m10/m00
    y_mc=m01/m00
    #calculate central moments
    mu02=0;
    mu11=0;
    for i in range (0,SIZE):
      for j in range (0,SIZE):
        mu02 += (img[i,j]*(i-x_mc)**2) 
        mu11 += (img[i,j]*(i-x_mc)*(j-y_mc))
    print ("mu02", mu02)
    print ("mu11", mu11)

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
# TRAINIG + TESTING 

train_num = 50
test_num = 100
train_data = mnist.train.images[0:train_num]
train_labels = mnist.train.labels[0:train_num] 
test_data = mnist.test.images[0:test_num]
test_labels = mnist.test.labels[0:test_num]

#naked
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(train_data, train_labels)
a = sv.svm_validation(test_data, test_labels)

#with deskew
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train(deskew_dataset(train_data),train_labels)
a = sv.svm_validation(deskew_dataset(test_data), test_labels)

#train for one number
sv0 = svm(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
sv.svm_train_1_num(train_data, train_labels, 0, sv0);
sv0.classifier(test_data)
np.shape(sv0.Z)



############################################
#speeded up training
"""
fast_svm_train(deskew_dataset(train_data), train_labels,deskew_dataset(test_data),test_labels)
"""
#############################################
#writing validation images into a file
y = open("saved_data/test_images/y.txt",'w')
np.savetxt(y,test_data,fmt='%.12e')
y.close()

#############################################
#writing bias into file
b = sv.sv0.b
b = np.reshape(b,(1,1));    
bias = open("saved_data/bias/bias.txt",'w')
np.savetxt(bias,b,fmt='%.12e')
bias.close()
#############################################
#writing K into file
K = open("saved_data/kernel/K.txt",'w')
np.savetxt(K,sv0.Z,fmt='%.12e')
K.close()

## indexes where classified numbers differ from the labeled ones
mistakes = np.where(test_labels!=sv.classified)

#############################################
# CLASSIFYING SINGLE IMAGE 

num=685

print("Normal classification")
sv.svm_one_num_classification(mnist.test.images[num])
d=np.reshape(mnist.test.images[num],(28,28)) 
pt.figure()
pt.imshow(d,cmap='Greys_r')

print("Classification with deskew")
sv.svm_one_num_classification(deskew(mnist.test.images[num]))
d=np.reshape(deskew(mnist.test.images[num]),(28,28)) 
pt.figure()
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
#Uz svo duzno postovanje prema ovome datasetu, slika sa indeksom 582
#izgleda kao kad koca pokusa da nacrta svastiku.
#Sada shvatam zasto je preciznost mala

sii=420
y=(mnist.test.images[sii])
yy=np.reshape(y,(28,28))
pt.imshow(yy,cmap='Greys_r')  
m = cv2.moments(yy)
calc_moments(y)
m['mu02']
m['mu11']
 
skew = m['mu11']/m['mu02']

z=deskew(mnist.test.images[sii])
zz=np.reshape(z,(28,28))
pt.imshow(zz,cmap='Greys_r')
