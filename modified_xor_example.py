# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:48:34 2018

@author: Nikola
"""
import numpy as np
import pylab as pl
def modified_XOR (kernel, degree, C, sdev):
    import SVM
    from SVM import svm
    sv = svm(kernel, degree = degree, C=C)
    
    m = 100
    X = sdev * np.random.randn(m, 2) # na ovaj nacin dobijamo standardnu gausovu raspodelu, sa mean-om nula i standarndnom devijacijom 1 stim sto
                                     # mozemo da skaliramo standardnu devijaciju pomocu promenljive sdev
    X[m//2: ,0] += 1. # // ovaj operator zaokruzuje na nizu vrednost ..primer 15/4=3. I ovde smo elemente koji se nalaze u drugoj polivini nultog reda
                      # uvecali za jedan
    X[m//4:m//2, 1] += 1.
    X[3 * m//4:, 1] += 1.
    # Prethodna modifikacija za X dovela je do toga da imamo parove (0,0) od 0 do 25, (0,1) od 25 do 50 (1,0) od 50 do 75 i (1,1) od 75 do 100
    targets = -np.ones((m, 1))        
    targets[:m//4, 0] = 1.  #od nultog do 25
    targets[3*m//4:, 0] = 1.#od 75 do kraja
    #prethodna modifikacija za targete, nam govori kojoj klasi treba da pripadaju koji parovi
    
    sv.train_svm(X, targets)# treniranje mreze
    
    Y = sdev*np.random.randn(m,2)
    Y[m//2:,0] += 1.
    Y[m//4:m//2, 1] += 1.
    Y[3*m//4:m, 1] += 1.
    test = -np.ones((m, 1))
    test[:m//4, 0] = 1.
    test[3*m//4:, 0] = 1.
    
    output = sv.classifier(Y,soft=False) # klasifikovanje
    
    err1 = np.where((output==1.) & (test==-1.))[0]
    err2 = np.where((output==-1.) & (test==1.))[0]
    
    print (kernel, C)
    print ("Class 1 errors ",len(err1)," from ",len(test[test==1]))
    print ("Class 2 errors ",len(err2)," from ",len(test[test==-1]))
    print ("Test accuracy ",1. -(float(len(err1)+len(err2)))/ (len(test[test==1]) + len(test[test==-1])))
    
    pl.ion()
    pl.figure()
    l1 =  np.where(targets==1)[0]
    l2 =  np.where(targets==-1)[0]
    pl.plot(X[sv.sv,0],X[sv.sv,1],'o',markeredgewidth=5)
    pl.plot(X[l1,0],X[l1,1],'ko')
    pl.plot(X[l2,0],X[l2,1],'wo')
    l1 =  np.where(test==1)[0]
    l2 =  np.where(test==-1)[0]
    pl.plot(Y[l1,0],Y[l1,1],'ks')
    pl.plot(Y[l2,0],Y[l2,1],'ws')
    
    step = 0.1
    f0,f1  = np.meshgrid(np.arange(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, step), np.arange(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, step))

    out = sv.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T

    out = out.reshape(f0.shape)
    pl.contour(f0, f1, out,2)

    pl.axis('off')
    pl.show()
    
def run_mxor():
#for sdev in [0.1]:
    for sdev in [0.1, 0.3, 0.4]:
        modified_XOR('linear',1,None,sdev)
        modified_XOR('linear',1,0.1,sdev)
        modified_XOR('poly',3,None,sdev)
        modified_XOR('poly',3,0.1,sdev)
        modified_XOR('rbf',0,None,sdev)
        modified_XOR('rbf',0,0.1,sdev)
        
run_mxor()