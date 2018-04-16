# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:57:40 2018

@author: Nikola
"""
import numpy as np
from SVM import svm

class svm_num_recognition:
    def __init__(self, kernel = 'poly', C = 1, degree = 1, sigma = 1, threshold = 1e-5):

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
            if training_labels[i]==number:
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
        classification = np.sign(classification)
        confusion_matrix = np.ones((2,2))
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for i in range (0, np.shape(classification)[0]):
            if(labels[i] == number):
                if(classification[i] == 1):
                    true_positives = true_positives + 1
                elif(classification[i] == -1):
                    false_negatives = false_negatives + 1
                    #print("label of the non-classified number is: ",10000+i)
            else:        
                if(classification[i] == 1):
                    false_positives = false_positives + 1
                    #print("label of the non-classified number is: ",10000+i)
                elif(classification[i] == -1):
                    true_negatives = true_negatives + 1
        
        confusion_matrix [0, 0]= true_positives
        confusion_matrix [0, 1]= false_positives
        confusion_matrix [1, 0]= false_negatives
        confusion_matrix [1, 1]= true_negatives
        print("confusion matrix for num:",number,"is\n", confusion_matrix)
        print("accuracy = {0:.2f}%".format(((true_positives+true_negatives)/np.shape(classification)[0])*100))
        
    
        
    def svm_validation(self, validation_images,validation_labels):
        if(np.shape(validation_labels)[0]!=np.shape(validation_images)[0]):
            print("size of label matrix and data_matrix are not the same!!")
            return -1;
        Ytest = validation_images
        self.soft = True
        self.classification0=self.sv0.classifier(Ytest,soft = self.soft)
        self.classification1=self.sv1.classifier(Ytest,soft = self.soft)
        self.classification2=self.sv2.classifier(Ytest,soft = self.soft)
        self.classification3=self.sv3.classifier(Ytest,soft = self.soft)
        self.classification4=self.sv4.classifier(Ytest,soft = self.soft)
        self.classification5=self.sv5.classifier(Ytest,soft = self.soft)
        self.classification6=self.sv6.classifier(Ytest,soft = self.soft)
        self.classification7=self.sv7.classifier(Ytest,soft = self.soft)
        self.classification8=self.sv8.classifier(Ytest,soft = self.soft)
        self.classification9=self.sv9.classifier(Ytest,soft = self.soft)
        
        self.confusion_matrix_func(classification = self.classification0, labels = validation_labels, number = 0)
        self.confusion_matrix_func(classification = self.classification1, labels = validation_labels, number = 1)
        self.confusion_matrix_func(classification = self.classification2, labels = validation_labels, number = 2)
        self.confusion_matrix_func(classification = self.classification3, labels = validation_labels, number = 3)
        self.confusion_matrix_func(classification = self.classification4, labels = validation_labels, number = 4)
        self.confusion_matrix_func(classification = self.classification5, labels = validation_labels, number = 5)
        self.confusion_matrix_func(classification = self.classification6, labels = validation_labels, number = 6)
        self.confusion_matrix_func(classification = self.classification7, labels = validation_labels, number = 7)
        self.confusion_matrix_func(classification = self.classification8, labels = validation_labels, number = 8)
        self.confusion_matrix_func(classification = self.classification9, labels = validation_labels, number = 9)
        
        return self.svm_big_confusion_matrix(validation_labels)
        
    def svm_one_num_classification(self, image):
        Ytest=np.reshape(image, (1,784))
        classification0=self.sv0.classifier(Ytest,soft = self.soft)
        classification1=self.sv1.classifier(Ytest,soft = self.soft)
        classification2=self.sv2.classifier(Ytest,soft = self.soft)
        classification3=self.sv3.classifier(Ytest,soft = self.soft)
        classification4=self.sv4.classifier(Ytest,soft = self.soft)
        classification5=self.sv5.classifier(Ytest,soft = self.soft)
        classification6=self.sv6.classifier(Ytest,soft = self.soft)
        classification7=self.sv7.classifier(Ytest,soft = self.soft)
        classification8=self.sv8.classifier(Ytest,soft = self.soft)
        classification9=self.sv9.classifier(Ytest,soft = self.soft)
        conc_classification = np.concatenate((classification0,classification1,
                                              classification2,classification3,
                                              classification4,classification5,
                                              classification6,classification7,
                                              classification8,classification9), axis=1)
        classified = np.argmax(conc_classification, axis = 1)
        
        print("predicted number is: ", classified)
        
    
    def svm_big_confusion_matrix(self, validation_labels):
        conc_classification = np.concatenate((self.classification0,self.classification1,
                                              self.classification2,self.classification3,
                                              self.classification4,self.classification5,
                                              self.classification6,self.classification7,
                                              self.classification8,self.classification9), axis=1) 
        self.classified = np.argmax(conc_classification, axis = 1)
        
        self.big_confusion=np.zeros([10,10])
        
        for i in range (0, np.shape(validation_labels)[0]):
            self.big_confusion[validation_labels[i],self.classified[i]] += 1
    
        
        print("big confusion matrix is:\n", self.big_confusion)
        print("percentage: ",np.trace(self.big_confusion)/np.shape(validation_labels)[0])
        return (np.trace(self.big_confusion)/np.shape(validation_labels)[0])

    
        
        
        
        
    