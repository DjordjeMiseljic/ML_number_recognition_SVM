"""
Created on Tue Sep  11 08:46:20 2001

@author: Nikola
"""
import numpy as np
import cvxopt
#from numba import jit


class svm:
    
    def __init__(self, kernel = 'linear', C = None, sigma=1., degree = 1, threshold=1e-5):
        self.kernel = kernel
        if self.kernel == 'linear':
            self.kernel = 'poly'
            self.degree = 1.
        self.C = C
        self.sigma = sigma
        self.degree = degree
        self.threshold = threshold
        print(kernel,C,sigma,degree, threshold)
        
    def build_kernel(self, X):
        self.K = np.dot(X,X.T)
        if self.kernel == 'poly': 
            self.K = (1. + 1./self.sigma*self.K)**self.degree
        elif self.kernel == 'rbf':
            self.xsquared = (np.diag(self.K)*np.ones((1, self.N))) 
            
            b = np.ones((self.N, 1))
            self.K -= 0.5*(np.dot(self.xsquared, b) + np.dot(b, self.xsquared)) # ovde je bila greska u kodu, pogledaj originalni !!!
            self.K = np.exp(self.K/(2.*self.sigma**2))
        
    def train_svm(self, X, targets):
        self.N = np.shape(X)[0] # vraca broj redova u matrici X
        self.build_kernel(X)
        # proracunavanje matrica koje prosledjujemo solveru, pogledaj u knjzi strana 180
        P = targets*targets.transpose()*self.K
        q = -np.ones((self.N, 1))
        if self.C is None: # ako ne postoji slack variabla
            G = -np.eye(self.N)
            h = np.zeros((self.N, 1))
        else:
            G = np.concatenate((np.eye(self.N),-np.eye(self.N))) # objasnjenje isto sto i za h
            h = np.concatenate((self.C*np.ones((self.N,1)),np.zeros((self.N,1))))# iz razloga sto je labmda vece od nule i manje od C potrebno je da 
                                                                                 # izvrsimo kontatanaciju 2 colon matrice nula i C-ova
        A = targets.reshape(1, self.N)                                           
        b = 0.0                                                                          
        sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        
        lambdas = np.array(sol['x'])# izvlaci lambde iz sol-a, 'x' predstavlja neki parametar kojim to radimo
        
        self.sv = np.where(lambdas > self.threshold)[0]# vraca matricu 1xm [0,1,2,4,6], gde nam broj govori koji od prosledjenih vektora je support vektor,
                                                       # odnosno ovi brojevi predstavljaju redove matrice X, odnosno ti redovi su support vektori
                                                      
        self.nsupport = len(self.sv) # ovim dobijamo ukupan broj support vectora
        print (self.nsupport, "support vectors found")
        
        self.X = X[self.sv, :]# na ovaj nacin uzimamo samo one support vektore koji nam trebaju
        self.lambdas = lambdas[self.sv] #uzimamo samo lambde koje odgovaraju support vektorima
        self.targets = targets[self.sv] #uzimamo samo targete koji odgovaraju support vektorima
        
        
        
        self.b = np.sum(self.targets)
        for n in range(self.nsupport):# loop u zavisnosti od toga koliko support vektora ima
            self.b -= np.sum(self.lambdas * self.targets * np.reshape(self.K[self.sv[n], self.sv], (self.nsupport, 1))) # proracun za bias
        self.b /= len(self.lambdas)
        
        
        if self.kernel == 'poly':
            def classifier(Y, soft = False): # ovim klasifikujemo podatke nakon sto je mreza istrenirana
                K = (1. + 1./self.sigma*np.dot(Y,self.X.T))**self.degree
                
                self.y = np.zeros((np.shape(Y)[0],1)) # pravi matricu nula cija je dimenzija broj_redova_matrice(Y)x1
                for j in range (np.shape(Y)[0]): # ova dupla for petlja obavlja jednacinu 8.11 iz knjige na strani 175
                    for i in range (self.nsupport):
                        self.y[j] += self.lambdas[i] * self.targets[i] * K[j, i]
                    self.y[j] += self.b
                    
                if soft:
                    return self.y
                else:
                    return np.sign(self.y) # vraca klasifikovan y odnosno matricu minus jedinica i jedinica
        elif self.kernel == 'rbf':
            def classifier(Y,soft=False):
                K = np.dot(Y, self.X.T)
                c = (1./self.sigma * np.sum(Y**2,axis=1) * np.ones((1, np.shape(Y)[0]))).T
                c = np.dot(c, np.ones((1, np.shape(K)[1])))
                aa = np.dot(self.xsquared.T[self.sv], np.ones((1, np.shape(K)[0]))).T # i ovde je bila greska, pa sam prepravio pogledaj original, vezano je za squared
                K = K - 0.5 * c - 0.5 * aa
                K = np.exp(K/(2. * self.sigma**2))
                
                self.y = np.zeros((np.shape(Y)[0], 1))
                for j in range(np.shape(Y)[0]):
                    for i in range(self.nsupport):
                        self.y[j] += self.lambdas[i] * self.targets[i] * K[j,i]
                    self.y[j] += self.b
                if soft:
                    return self.y
                else:
                    return np.sign(self.y)
        else:
            print ("Error -- kernel not recognised")
            return
        self.classifier = classifier
                
