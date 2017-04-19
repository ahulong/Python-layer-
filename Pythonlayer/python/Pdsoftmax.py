import numpy as np
import math
import os
import cv2
import  matplotlib.pyplot as plt
import sys
import cv2
import h5py 
caffe_root='/home/apollo/Workspace/user/caffe'
sys.path.insert(0,caffe_root+'/python')
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
        cls_conv = bottom[0].data
        self.__class__.mlabel=bottom[4].data[:,0,:,:]
        

        poscount=0
        for i in range(batch_size):
            for x in range(60):
                for y in range(60):
                
                    if self.__class__.mlabel[i,x,y]<=-0.5:
                       continue
                       
                    if ystar[i,x,y]>=0.5:
                       poscount+=1
              
        #select the negtive sample pixels
        for j in range(poscount):
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
            tempcount=0.0
            temploss=np.float32(0.0)
            tempnum=0
            for x in range(60):
                for y in range(60):
                    if  self.__class__.mlabel[i,x,y]<=0.5:
                           continue
                           
                    cls_pred_max = max(cls_conv[i,0,x,y],cls_conv[i,1,x,y])
                    x0=cls_conv[i,0,x,y]-cls_pred_max
                    x1=cls_conv[i,1,x,y]-cls_pred_max
                    rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
                    rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                    tempnum+=1
                    if ystar[i,x,y]>=0.5:   
                       temploss+=x1-np.log(np.exp(x1)+np.exp(x0))
                       if rate1>=0.5:
                           tempcount+= 1
                    else:
                       temploss+= x0-np.log(np.exp(x1)+np.exp(x0))            
                       if rate0>=0.5:
                          tempcount+= 1
            if tempnum>0:
               cls_loss+=temploss/tempnum
               correct_count+=tempcount/float(tempnum) 
               num+=1              
         
        if num>0:
           cls_loss =-cls_loss/num   
           cls_acc = correct_count/np.float32(num)  
        
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
                        temp_loss+=math.pow((reg_conv[i,channel,x,y]-float(dstar[i,channel,x,y])),2)  
                        pos_count += 1
            if pos_count > 0:            
               reg_loss+=temp_loss/pos_count
               count+=1        
        if count > 0:  
           reg_loss = reg_loss/2.0/count       
                             
        top[0].data[...] = np.float32(cls_loss+3*reg_loss)
        print 'classification loss = ', cls_loss, ' acc = ', cls_acc, ' regression loss = ', reg_loss

    def backward(self, top, propagate_down, bottom):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        
        if propagate_down[0]:
            cls_diff= np.zeros_like(cls_conv, dtype=np.float32) 
            num=0       
            for i in range(batch_size):
                tempcount=0
                for x in range(60):
                    for y in range(60):
                         if self.__class__.mlabel[i,x,y]<=0.5:
                             continue
                         cls_pred_max = np.max(cls_conv[i,:,x,y])
                         x0=cls_conv[i,0,x,y]-cls_pred_max
                         x1=cls_conv[i,1,x,y]-cls_pred_max
                         rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
                         rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                         tempcount+=1 
                         if ystar[i,x,y]>0.5: 
                            cls_diff[i,0,x,y]+=rate0
                            cls_diff[i,1,x,y]+=rate1-1
                         else:
                            cls_diff[i,0,x,y]+=rate0-1
                            cls_diff[i,1,x,y]+=rate1 
                if tempcount>0:
                   cls_diff[i]/=tempcount
                   num+=1
            if num>0:                    
               cls_diff/=num                               
            bottom[0].diff[...] = cls_diff
        
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
                            pos_count +=1
                if  pos_count >0:
                    count+=1
                    reg_diff[i,:,:,:]/=pos_count        
            if count>0:
               reg_diff = reg_diff/count 
            bottom[1].diff[...] = 3*reg_diff
        print 'the backward is over!'  
