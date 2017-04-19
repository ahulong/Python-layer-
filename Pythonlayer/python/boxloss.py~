import numpy as np
import math
import os
import cv2
import  matplotlib.pyplot as plt
import sys
import cv2
import h5py 
caffe_root='/mnt/data1/hxw/caffe'
sys.path.insert(0,caffe_root+'/python')
import caffe


class loss(caffe.Layer):
    
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
        
        Negloss=list()
        poscount=0
        for i in range(batch_size):
            batchloss=list()
            negnum=0
            for x in range(60):
                for y in range(60):
                
                    if self.__class__.mlabel[i,x,y]<=-0.5:
                       continue
                       
                    if ystar[i,x,y]>=0.5:
                       poscount+=1
                       continue
                       
                    batchloss.append([math.pow((cls_conv[i,0,x,y]-0),2),i,x,y]) 
                    negnum+=1
                    
            batchloss.sort(key=self.getKey,reverse=True)
            for k in range(int(negnum*0.1)):
                Negloss.append(batchloss[k])

        
        #select the negtive sample pixels
        #select the hard samples
        for j in range(poscount/2):
              pos=np.random.randint(0,len(Negloss)) 
              i=Negloss[pos][1]
              x=Negloss[pos][2]
              y=Negloss[pos][3]
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

        
        cls_loss = np.float32(0.0)
        correct_count = np.float32(0.0)
        totalnum=0 
        cls_acc=np.float32(0.0)  
        for i in range(batch_size):
            tempcount=0.0
            temploss=np.float32(0.0)
            tempnum=0
            for x in range(60):
                for y in range(60):
                    if  self.__class__.mlabel[i,x,y]<=0.5:
                       continue
                    tempnum+=1
                    value=1.0/(1.0+np.exp(-cls_conv[i,0,x,y]))
                    if ystar[i,x,y]>=0.5:
                       temploss+=math.pow((value-ystar[i,x,y]),2)
                       if value>=0.5:
                          tempcount+= 1
                    else:
                       temploss+=math.pow((value-ystar[i,x,y]),2)
                       if value<0.5:
                          tempcount+= 1
            if tempnum>0:
               cls_loss+=temploss/tempnum
               correct_count+=tempcount/tempnum
               totalnum+=1
    
        if totalnum>0:
           cls_loss = cls_loss/2.0/np.float32(totalnum)   
           cls_acc= correct_count/np.float32(totalnum)

        
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
        mlabel=bottom[4].data[:,0,:,:]
        
        if propagate_down[0]:
            cls_diff= np.zeros_like(cls_conv, dtype=np.float32)
            totalnum=0        
            for i in range(batch_size):
                tempcount=0
                for x in range(60):
                    for y in range(60):
                        if  self.__class__.mlabel[i,x,y]<=0.5:
                           continue
                        tempcount+=1
                        if ystar[i,x,y]>=0.5:
                           cls_diff[i,0,x,y]+=cls_conv[i,0,x,y]-ystar[i,x,y]
                        else:
                           cls_diff[i,0,x,y]+=cls_conv[i,0,x,y]-ystar[i,x,y]
                if tempcount>0:
                   cls_diff[i,0,:,:]/=tempcount
                   totalnum+=1           
                            
            if totalnum>0:
                cls_diff/=totalnum
                              
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
