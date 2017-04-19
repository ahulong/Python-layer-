import numpy as np
import math
import os
import cv2
import  matplotlib.pyplot as plt
import sys
import cv2
import h5py 
import random
import caffe

class denloss(caffe.Layer):

    mlabel=0
    def getKey(self,x):
        return x[0]
    
    def setup(self, bottom, top):
        if len(bottom) != 5:
            raise Exception("Need 5 inputs to compute the loss.")
        self._name_to_bottom_map={'conv5_2_det':0,'conv5_2_loc':1,'ylabel':2,'dlabel':3,'mlabel':4}
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data[:,0,:,:]
        self.__class__.mlabel=bottom[4].data[:,0,:,:]
        
        Negloss=list()
        poscount=0
        batchloss=list()
        negnum=0
        for i in range(batch_size):
            for x in range(60):
                for y in range(60):
                
                    if self.__class__.mlabel[i,x,y]<=-0.5:
                       continue
                       
                    if ystar[i,x,y]>=0.5:
                       poscount+=1
                       continue
                       
                    cls_pred_max = np.max(cls_conv[i,:,x,y])   
                    x0=cls_conv[i,0,x,y]-cls_pred_max
                    x1=cls_conv[i,1,x,y]-cls_pred_max
                    rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                    batchloss.append([rate1,i,x,y]) 
                    negnum+=1
                    

        batchloss.sort(key=self.getKey,reverse=True)
        for k in range(int(negnum*0.4)):
              Negloss.append(batchloss[k])
        
        
        #select the negtive sample pixels
        #select the hard samples
        for j in range(poscount/2):
             
              i=Negloss[j][1]
              x=Negloss[j][2]
              y=Negloss[j][3]
              self.__class__.mlabel[i,x,y]=1.0
   

        if poscount%2==0:
             count=poscount/2
        else:
             count=poscount/2+1
          
        #random select the neg pixels
        for j in range(count):
             i=np.random.randint(0,batch_size)
             x=np.random.randint(0,60)
             y=np.random.randint(0,60)
             if self.__class__.mlabel[i,x,y]<-0.5 or self.__class__.mlabel[i,x,y]>0.5:
                continue
             self.__class__.mlabel[i,x,y]=1.0

        
        cls_loss = 0.0
        correct_count = 0.0    
        num=0
        for i in range(batch_size):
            for x in range(60):
                for y in range(60):
                
                    if  self.__class__.mlabel[i,x,y]<=0.5:
                           continue
                    num+=1       
                    cls_loss+=math.pow(cls_conv[i,x,y]-ystar[i,x,y],2)
                   
                    if ystar[i,x,y]>0.5:
                       print 'label1 prop=', cls_conv[i,x,y]
                       if cls_conv[i,x,y]>0.5:       
                          correct_count+=1
                    else:
                       print 'label0 prop=', cls_conv[i,x,y]
                       if cls_conv[i,x,y]<0.5:  
                          correct_count+=1 
                          

        if num>0:
           cls_loss =cls_loss/2.0/num   
           cls_acc = correct_count/float(num)  
        
        reg_loss = np.float32(0.0)
        count=0
        for i in range(batch_size):
            pos_count =0
            temp_loss=np.float32(0.0)
            for channel in range(4):
                for x in range(60):
                    for y in range(60):
                        if ystar[i,x,y]<0.5:
                            continue
                        reg_loss+=math.pow((reg_conv[i,channel,x,y]-float(dstar[i,channel,x,y])),2)  
                        count += 1
           
        if count > 0:  
           reg_loss = reg_loss/2.0/count       
                             
        top[0].data[...] = np.float32(cls_loss+3*reg_loss)
        print 'classification loss = ', cls_loss, ' acc = ', cls_acc, ' regression loss = ', reg_loss

    def backward(self, top, propagate_down, bottom):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data[:,0,:,:]
        
        if propagate_down[0]:
            cls_diff= np.zeros_like(cls_conv, dtype=np.float32) 
            num=0       
            for i in range(batch_size):
                for x in range(60):
                    for y in range(60):
                         if self.__class__.mlabel[i,x,y]<=0.5:
                             continue
                         num+=1
                         cls_diff[i,x,y]= cls_conv[i,x,y]-ystar[i,x,y]   

            if num>0:                    
               cls_diff/=num                               
            bottom[0].diff[:,0,:,:] = cls_diff
        
        if propagate_down[1]:
            reg_diff = np.zeros_like(reg_conv, dtype=np.float32)
            count=0
            for i in range(batch_size):
                pos_count = 0
                for channel in range(4):
                    for x in range(60):
                        for y in range(60):
                            if ystar[i,x,y]<0.5:
                               continue
                            reg_diff[i,channel,x,y]+=(reg_conv[i,channel,x,y]-dstar[i,channel,x,y])
                            count +=1
                   
            if count>0:
               reg_diff = reg_diff/count 
            bottom[1].diff[...] = 3*reg_diff
        print 'the backward is over!'  
