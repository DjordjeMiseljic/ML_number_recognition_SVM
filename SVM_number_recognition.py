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
        
        #treba za upis u fajl
        self.num_train=np.shape(training_set)[0]
        
        
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
        
        # treba za upis u fajl
        self.num_test=np.shape(Ytest)[0]
        
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
        #Write primary results into file
        res = open("saved_data/results/res.txt",'w')
        np.savetxt(res,conc_classification,fmt='%.12e')
        res.close()
        #classify based on primary results
        self.classified = np.argmax(conc_classification, axis = 1)
        
        #make confusion matrix based on classifications
        self.big_confusion=np.zeros([10,10])
        for i in range (0, np.shape(validation_labels)[0]):
            self.big_confusion[validation_labels[i],self.classified[i]] += 1
        
        
        
        print("big confusion matrix is:\n", self.big_confusion)
        self.percent =100*np.trace(self.big_confusion)/np.shape(validation_labels)[0]
        self.save_to_files()
        print("percentage: {0:.2f}%".format(self.percent))
        return (self.percent)

    def save_to_files(self):
        #INFO
        info = open("saved_data/info.txt",'w')
        info.write("Kernel:\n")
        info.write(self.sv0.kernel)
        info.write("\nAccuracy:\n")
        info.write(str(self.percent))
        info.write("\nSlack constant:\n")
        info.write(str(self.sv0.C))
        info.write("\nDegree:\n")
        info.write(str(self.sv0.degree))
        info.write("\nSigma:\n")
        info.write(str(self.sv0.sigma))
        info.write("\nThreshold:\n")
        info.write(str(self.sv0.threshold))
        info.write("\nNumber of training examples:\n")
        info.write(str(self.num_train))
        info.write("\nNumber of testing examples:\n")
        info.write(str(self.num_test))
        info.close()
        
        #SUPPORT VECTORS
        support_vectors0 = open("saved_data/support_vectors/sv0.txt",'w')
        np.savetxt(support_vectors0,self.sv0.X,fmt='%.12e')
        support_vectors0.close()
        
        support_vectors1 = open("saved_data/support_vectors/sv1.txt",'w')
        np.savetxt(support_vectors1,self.sv1.X,fmt='%.12e')
        support_vectors1.close()
        
        support_vectors2 = open("saved_data/support_vectors/sv2.txt",'w')
        np.savetxt(support_vectors2,self.sv2.X,fmt='%.12e')
        support_vectors2.close()
        
        support_vectors3 = open("saved_data/support_vectors/sv3.txt",'w')
        np.savetxt(support_vectors3,self.sv3.X,fmt='%.12e')
        support_vectors3.close()
        
        support_vectors4 = open("saved_data/support_vectors/sv4.txt",'w')
        np.savetxt(support_vectors4,self.sv4.X,fmt='%.12e')
        support_vectors4.close()
        
        support_vectors5 = open("saved_data/support_vectors/sv5.txt",'w')
        np.savetxt(support_vectors5,self.sv5.X,fmt='%.12e')
        support_vectors5.close()
        
        support_vectors6 = open("saved_data/support_vectors/sv6.txt",'w')
        np.savetxt(support_vectors6,self.sv6.X,fmt='%.12e')
        support_vectors6.close()
        
        support_vectors7 = open("saved_data/support_vectors/sv7.txt",'w')
        np.savetxt(support_vectors7,self.sv7.X,fmt='%.12e')
        support_vectors7.close()
        
        support_vectors8 = open("saved_data/support_vectors/sv8.txt",'w')
        np.savetxt(support_vectors8,self.sv8.X,fmt='%.12e')
        support_vectors8.close()
        
        support_vectors9 = open("saved_data/support_vectors/sv9.txt",'w')
        np.savetxt(support_vectors9,self.sv9.X,fmt='%.12e')
        support_vectors9.close()
        
        
        
        #LAMBDAS
        lambdas0 = open("saved_data/lambdas/lambdas0.txt",'w')
        np.savetxt(lambdas0,self.sv0.lambdas,fmt='%.12e')
        lambdas0.close()
        
        lambdas1 = open("saved_data/lambdas/lambdas1.txt",'w')
        np.savetxt(lambdas1,self.sv1.lambdas,fmt='%.12e')
        lambdas1.close()
        
        lambdas2 = open("saved_data/lambdas/lambdas2.txt",'w')
        np.savetxt(lambdas2,self.sv2.lambdas,fmt='%.12e')
        lambdas2.close()
        
        lambdas3 = open("saved_data/lambdas/lambdas3.txt",'w')
        np.savetxt(lambdas3,self.sv3.lambdas,fmt='%.12e')
        lambdas3.close()
        
        lambdas4 = open("saved_data/lambdas/lambdas4.txt",'w')
        np.savetxt(lambdas4,self.sv4.lambdas,fmt='%.12e')
        lambdas4.close()
        
        lambdas5 = open("saved_data/lambdas/lambdas5.txt",'w')
        np.savetxt(lambdas5,self.sv5.lambdas,fmt='%.12e')
        lambdas5.close()
        
        lambdas6 = open("saved_data/lambdas/lambdas6.txt",'w')
        np.savetxt(lambdas6,self.sv6.lambdas,fmt='%.12e')
        lambdas6.close()
        
        lambdas7 = open("saved_data/lambdas/lambdas7.txt",'w')
        np.savetxt(lambdas7,self.sv7.lambdas,fmt='%.12e')
        lambdas7.close()
        
        lambdas8 = open("saved_data/lambdas/lambdas8.txt",'w')
        np.savetxt(lambdas8,self.sv8.lambdas,fmt='%.12e')
        lambdas8.close()
        
        lambdas9 = open("saved_data/lambdas/lambdas9.txt",'w')
        np.savetxt(lambdas9,self.sv9.lambdas,fmt='%.12e')
        lambdas9.close()
        
        
        
        #TARGETS
        targets0 = open("saved_data/targets/targets0.txt",'w')
        np.savetxt(targets0,self.sv0.targets,fmt='%.2f')
        targets0.close()
        
        targets1 = open("saved_data/targets/targets1.txt",'w')
        np.savetxt(targets1,self.sv1.targets,fmt='%.2f')
        targets1.close()
        
        targets2 = open("saved_data/targets/targets2.txt",'w')
        np.savetxt(targets2,self.sv2.targets,fmt='%.2f')
        targets2.close()
        
        targets3 = open("saved_data/targets/targets3.txt",'w')
        np.savetxt(targets3,self.sv3.targets,fmt='%.2f')
        targets3.close()
        
        targets4 = open("saved_data/targets/targets4.txt",'w')
        np.savetxt(targets4,self.sv4.targets,fmt='%.2f')
        targets4.close()
        
        targets5 = open("saved_data/targets/targets5.txt",'w')
        np.savetxt(targets5,self.sv5.targets,fmt='%.2f')
        targets5.close()
        
        targets6 = open("saved_data/targets/targets6.txt",'w')
        np.savetxt(targets6,self.sv6.targets,fmt='%.2f')
        targets6.close()
        
        targets7 = open("saved_data/targets/targets7.txt",'w')
        np.savetxt(targets7,self.sv7.targets,fmt='%.2f')
        targets7.close()
        
        targets8 = open("saved_data/targets/targets8.txt",'w')
        np.savetxt(targets8,self.sv8.targets,fmt='%.2f')
        targets8.close()
        
        targets9 = open("saved_data/targets/targets9.txt",'w')
        np.savetxt(targets9,self.sv9.targets,fmt='%.2f')
        targets9.close()
        
        
        
        
    