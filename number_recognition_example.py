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


#############################################
# IMPORTING DATASET
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#############################################
# TRAINIG + TESTING 

train_num = 10000
test_num = 5100
train_data = mnist.train.images[0:train_num]
train_labels = mnist.train.labels[0:train_num] 
test_data = mnist.test.images[5000:test_num]
test_labels = mnist.test.labels[5000:test_num]

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
a = sv.svm_validation((test_data), test_labels)

##train for one number
#sv0 = svm(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
#sv = svm_num_recognition(kernel = 'poly', C = 1.6, degree = 3, sigma = 1, threshold = 1e-7)
#sv.svm_train_1_num(train_data, train_labels, 0, sv0);
#sv0.classifier(test_data)
#np.shape(sv0.Z)
y = open("y.txt",'w')
np.savetxt(y,(test_data),fmt='%.12e')
y.close()

## indexes where classified numbers differ from the labeled ones
mistakes = np.where(test_labels!=sv.classified)

#############################################
# CLASSIFYING SINGLE IMAGE 

num=5054

print("Normal classification")
sv.svm_one_num_classification(mnist.test.images[num])
d=np.reshape(mnist.test.images[num],(28,28)) 
pt.figure()
pt.imshow(d,cmap='Greys_r')

print("Classification with deskew")
sv.svm_one_num_classification(deskew.deskew_manual(mnist.test.images[num]))
d=np.reshape(deskew.deskew_manual(mnist.test.images[num]),(28,28)) 
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


sii=421
y=(mnist.test.images[sii])
yy=np.reshape(y,(28,28))
pt.imshow(yy,cmap='Greys_r')  
m = cv2.moments(yy)
deskew_test = our_deskew();
our_deskew.calc_moments(y)
m['mu02']
m['mu11']
m['m01']/m['m00']
m['m10']/m['m00'] 

skew = m['mu11']/m['mu02']
M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
# Apply affine transform
dskw_auto = cv2.warpAffine(yy, M, (28,28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
pt.imshow(dskw_auto,cmap='Greys_r')


###testing deskew class "our_deskew.py"
sii=5001  
y=(mnist.test.images[sii])  
image=np.reshape(y,(28,28))
pt.imshow(image,cmap='Greys_r')
deskew_test = our_deskew(); 
image = deskew_test.deskew_manual(y)
image=np.reshape(image,(28,28))
pt.imshow(image,cmap='Greys_r')


sii=5000
y=(mnist.test.images[sii:5005])
image = open("4.txt",'w')
np.savetxt(image,y,fmt='%.12e')
image.close()

#### testing picture deskewed in c++
number = np.zeros([1,784]) 
file_handle = open('number.txt', 'r') 
for i in range (0,784): 
    number[0,i] = float(file_handle.readline() ) 
number = np.reshape(number,(28,28)); 
pt.imshow(number,cmap='Greys_r')     

