# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:57:40 2018

@author: Nikola
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from tensorflow.examples.tutorials.mnist import input_data
from SVM import svm

class svm_num_recognition:
    def __init__(self, kernel = 'poly', C = 1, degree = 1, sigma = 1, threshold = 1e-5):
        self.kernel = kernel
        if self.kernel == 'linear':# od 18 do 24 linije mozda ne treba
            self.kernel = 'poly'
            self.degree = 1.
        self.C = C
        self.sigma = sigma
        self.degree = degree
        self.threshold = threshold
        self. sv0 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv1 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv2 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv3 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv4 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv5 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv6 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv7 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv8 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
        self. sv9 = svm(kernel,C=C,degree=degree, sigma=sigma, threshold = threshold)
    
    
        
    def svm_train_1_num(self, training_set, training_labels, number , sv):
        if(np.shape(training_labels)[0]!=np.shape(training_set)[0]):
            print("size of label matrix and data_matrix are not the same!!")
            return -1;
        t = -np.ones((np.shape(training_set)[0],1))
        for i in range (0, np.shape(training_set)[0]):
            if training_labels[i,number]==1:
                t[i] = 1;
        sv.train_svm(training_set, t)
        
    def svm_train(self, training_set, training_labels):
        if(np.shape(training_labels)[0]!=np.shape(training_set)[0]):
            print("size of training_label matrix and training_matrix are not the same!!")
            return -1;
        self.svm_train_1_num(training_set, training_labels, 0, self.sv0)
        self.svm_train_1_num(training_set, training_labels, 1, self.sv1)
        self.svm_train_1_num(training_set, training_labels, 2, self.sv2)
        self.svm_train_1_num(training_set, training_labels, 3, self.sv3)
        self.svm_train_1_num(training_set, training_labels, 4, self.sv4)
        self.svm_train_1_num(training_set, training_labels, 5, self.sv5)
        self.svm_train_1_num(training_set, training_labels, 6, self.sv6)
        self.svm_train_1_num(training_set, training_labels, 7, self.sv7)
        self.svm_train_1_num(training_set, training_labels, 8, self.sv8)
        self.svm_train_1_num(training_set, training_labels, 9, self.sv9)
        
    def confusion_matrix_func (self, classification, labels, number):
        confusion_matrix = np.ones((2,2))
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range (0, np.shape(classification)[0]):
            if(labels[i,number] == 1):
               
                
                if(classification[i] == 1):
                    true_positives = true_positives + 1
                elif(classification[i] == -1):
                    false_negatives = false_negatives + 1
                #print("label of the non-classified number is: ",10000+i)
            elif(classification[i] == 1):
                false_positives = false_positives + 1
            #print("label of the non-classified number is: ",10000+i)
            elif (classification[i] == -1):
                true_negatives = true_negatives + 1
        
        confusion_matrix [0, 0]= true_positives
        confusion_matrix [0, 1]= false_positives
        confusion_matrix [1, 0]= false_negatives
        confusion_matrix [1, 1]= true_negatives
        print("confusion matrix for num:",number,"is\n", confusion_matrix)
        print("accuracy = ", (true_positives+true_negatives)/np.shape(classification)[0])
        
    def svm_validation(self, validation_images,validation_labels):
        if(np.shape(validation_labels)[0]!=np.shape(validation_images)[0]):
            print("size of label matrix and data_matrix are not the same!!")
            return -1;
        Ytest = validation_images
        classification0=self.sv0.classifier(Ytest,soft = False)
        classification1=self.sv1.classifier(Ytest,soft = False)
        classification2=self.sv2.classifier(Ytest,soft = False)
        classification3=self.sv3.classifier(Ytest,soft = False)
        classification4=self.sv4.classifier(Ytest,soft = False)
        classification5=self.sv5.classifier(Ytest,soft = False)
        classification6=self.sv6.classifier(Ytest,soft = False)
        classification7=self.sv7.classifier(Ytest,soft = False)
        classification8=self.sv8.classifier(Ytest,soft = False)
        classification9=self.sv9.classifier(Ytest,soft = False)
        
        self.confusion_matrix_func(classification = classification0, labels = validation_labels, number = 0)
        self.confusion_matrix_func(classification = classification1, labels = validation_labels, number = 1)
        self.confusion_matrix_func(classification = classification2, labels = validation_labels, number = 2)
        self.confusion_matrix_func(classification = classification3, labels = validation_labels, number = 3)
        self.confusion_matrix_func(classification = classification4, labels = validation_labels, number = 4)
        self.confusion_matrix_func(classification = classification5, labels = validation_labels, number = 5)
        self.confusion_matrix_func(classification = classification6, labels = validation_labels, number = 6)
        self.confusion_matrix_func(classification = classification7, labels = validation_labels, number = 7)
        self.confusion_matrix_func(classification = classification8, labels = validation_labels, number = 8)
        self.confusion_matrix_func(classification = classification9, labels = validation_labels, number = 9)
    
    def svm_one_num_classification(self, image):
        Ytest=np.reshape(image, (1,784))
        classification0=self.sv0.classifier(Ytest,soft = False)
        classification1=self.sv1.classifier(Ytest,soft = False)
        classification2=self.sv2.classifier(Ytest,soft = False)
        classification3=self.sv3.classifier(Ytest,soft = False)
        classification4=self.sv4.classifier(Ytest,soft = False)
        classification5=self.sv5.classifier(Ytest,soft = False)
        classification6=self.sv6.classifier(Ytest,soft = False)
        classification7=self.sv7.classifier(Ytest,soft = False)
        classification8=self.sv8.classifier(Ytest,soft = False)
        classification9=self.sv9.classifier(Ytest,soft = False)
        if(classification0 == 1):
            print(0)
        elif(classification1 == 1):
            print("classified number is", 1)
        elif(classification2 == 1):
            print ("classified number is", 2)
        elif(classification3 == 1):
            print ("classified number is", 3)
        elif(classification4 == 1):
            print ("classified number is", 4)
        elif(classification5 == 1):
            print ("classified number is", 5)
        elif(classification6 == 1):
            print ("classified number is", 6)
        elif(classification7 == 1):
            print ("classified number is", 7)
        elif(classification8 == 1):
            print ("classified number is", 8)
        elif(classification9 == 1):
            print ("classified number is", 9)
        else:
            print ("number not recognized")
        
    


        



    
        
        
        
        
    