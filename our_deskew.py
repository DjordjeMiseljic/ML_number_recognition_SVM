# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:01:06 2018

@author: Nikola
"""
import numpy as np
import cv2

class our_deskew:
    def __init__(self):
        pass
    
    def deskew_cv2(self, image):
        img=np.reshape(image,(28,28))
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
    
#    def deskew_dataset(self, dataset):
#        ds=dataset.copy()
#        for i in range (0,np.shape(ds)[0]):
#            ds[i]=self.deskew_cv2(dataset[i])
#        return ds
    def deskew_dataset_auto(self, dataset):
                print("auto_deskew")
                ds=dataset.copy()
                for i in range (0,np.shape(ds)[0]):
                    ds[i]=self.deskew_cv2(dataset[i])
                return ds
    def deskew_dataset_manual(self, dataset):
                ds=dataset.copy()
                for i in range (0,np.shape(ds)[0]):
                    ds[i]=self.deskew_manual(dataset[i])
                return ds
            
    def calc_moments(self, image):
        img=np.reshape(image,(28,28))
        SIZE=28
        #calculate spatial moments
        m00=0;
        m10=0;
        m01=0  
        for x in range (0,SIZE):
          for y in range (0,SIZE):
            m00 += (img[y,x]) 
            m10 += (img[y,x]*x)
            m01 += (img[y,x]*y)
        #calculate mass center
        x_mc=m10/m00
        y_mc=m01/m00
       
        #calculate central moments
        self.mu02=0;
        self.mu11=0;
        for x in range (0,SIZE):
          for y in range (0,SIZE):
            self.mu02 += (img[y,x]*(y-y_mc)**2) 
            self.mu11 += (img[y,x]*(x-x_mc)*(y-y_mc))
        
        
    def deskew_manual(self, image):
        self.calc_moments(image);
        img=np.reshape(image,(28,28))
        deskew_image = np.copy(img)
        skew  = self.mu11/self.mu02
        M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])

        for x in range(0,28):
          for y in range(0,28):
            xp=(M[0,0]*x +M[0,1]*y+ M[0,2])
            yp=(M[1,0]*x +M[1,1]*y+ M[1,2])   
            if (xp<27 and yp<27 and xp>=0 and yp>=0):
               x1=(int(xp))
               y1=(int(yp))
               x2=(x1+1)
               y2=(y1+1)
               R1 = img[y1,x1] + float((xp - x1))/(x2 - x1)*(img[y1,x2] - img[y1,x1])
               #R1= 0.6       +    0.87/1   *   -0.06
               R2 = img[y2,x1] + float((xp - x1))/(x2 - x1)*(img[y2,x2] - img[y2,x1])
               P = R2 + float(yp - y1)/(y2 - y1)*(R1 - R2)
               if (P<0):      
                 P=0
               deskew_image[y,x]=P
               
            else:
              deskew_image[y,x]=0.0
           
        return np.reshape(deskew_image,(1,784))