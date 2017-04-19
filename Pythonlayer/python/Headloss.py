from xml.etree import ElementTree
from scipy import misc
import  matplotlib.pyplot as plt
from    matplotlib.patches import Rectangle
import numpy as np
import math
import os
import cv2
import sys
import cv2
import h5py 
import caffe
from multiprocessing import Pool

#calculate the classification and regression loss for using the multiprocess
def calloss(ystar,dstar,cls_conv,reg_conv,mlabel):
    tempyloss=0.0
    tempnum=0
    tempcorrect=0
    tempdloss=0.0
    tempcount=0
    
    for x in range(60):
        for y in range(60):
            if  mlabel[x,y]<=0.5:
                     continue             
            cls_pred_max = max(cls_conv[0,x,y],cls_conv[1,x,y])
            x0=cls_conv[0,x,y]-cls_pred_max
            x1=cls_conv[1,x,y]-cls_pred_max
            rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
            rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
            tempnum+=1
            
            if ystar[x,y]>=0.5:   
               tempyloss+=x1-np.log(np.exp(x1)+np.exp(x0))
               if rate1>=0.5:
                  tempcorrect+=1
            else:
               tempyloss+= x0-np.log(np.exp(x1)+np.exp(x0))            
               if rate0>=0.5:
                     tempcorrect+=1

    for channel in range(4):
            for x in range(60):
               for y in range(60):
                        if ystar[x,y]<0.5:
                            continue
  
                        tempdloss+=math.pow((reg_conv[channel,x,y]-float(dstar[channel,x,y])),2)  
                        tempcount += 1
                        
    return [tempyloss,tempnum,tempcorrect,tempdloss,tempcount]                    

#calculate the classification  and regression's diff for using multiprocess               
def caldiff(cls_conv,reg_conv,dstar,ystar,mlabel):

    cls_diff= np.zeros((2,60,60))
    reg_diff = np.zeros((4,60,60))
    tempnum=0
    tempcount=0
    
    for x in range(60):
        for y in range(60):
            if mlabel[x,y]<=0.5:
               continue
               
            cls_pred_max = np.max(cls_conv[:,x,y])
            x0=cls_conv[0,x,y]-cls_pred_max
            x1=cls_conv[1,x,y]-cls_pred_max
            rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
            rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
            tempnum+=1 
            if ystar[x,y]>0.5: 
               cls_diff[0,x,y]+=rate0
               cls_diff[1,x,y]+=rate1-1
            else:
               cls_diff[0,x,y]+=rate0-1
               cls_diff[1,x,y]+=rate1 



    for channel in range(4):
        for x in range(60):
            for y in range(60):
                if ystar[x,y]<0.5:
                   continue
                reg_diff[channel,x,y]+=(reg_conv[channel,x,y]-dstar[channel,x,y])
                tempcount +=1
                
    return [cls_diff,reg_diff,tempnum,tempcount]               
       

class denloss(caffe.Layer):

    #mlabel=0
    def getKey(self,x):
        return x[0]
    def plot_circle(self,bb,col):
        ax=plt.gca()
        circ=plt.Circle((bb[0], bb[1]), radius=2, color=col)
        ax.add_patch(circ)
        
    def setup(self, bottom, top):
        #if len(bottom) != 5:
         #   raise Exception("Need 6 inputs to compute the loss.")
        self._name_to_bottom_map={'conv5_2_det':0,'conv5_2_loc':1,'ylabel':2,'dlabel':3,'mlabel':4,'data':5}
        
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        self.__class__.mlabel=bottom[4].data[:,0,:,:]
        img=bottom[5].data
        
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
                    rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
                    batchloss.append([rate0,i,x,y]) 
                    negnum+=1
                    

        batchloss.sort(key=self.getKey)
        for k in range(int(negnum*0.5)):
              Negloss.append(batchloss[k])
        
        
        #select the negtive sample pixels
        #select the hard samples
        for j in range(poscount/2):
             
              i=Negloss[j][1]
              x=Negloss[j][2]
              y=Negloss[j][3]
              self.__class__.mlabel[i,x,y]=2.0
   

        if poscount%2==0:
             count=poscount/2
        else:
             count=poscount/2+1
          
        #random select the neg pixels
        j=0 
        while j<count:
             i=np.random.randint(0,batch_size)
             x=np.random.randint(0,60)
             y=np.random.randint(0,60)
             if self.__class__.mlabel[i,x,y]<-0.5 or self.__class__.mlabel[i,x,y]>0.5:
                continue
             self.__class__.mlabel[i,x,y]=1.0
             j+=1
     
        cls_loss = 0.0
        correct_count = 0.0    
        num=0
        reg_loss = np.float32(0.0)
        count=0
        
        pool=Pool(processes=10)
        result=list()
        
        for i in range(batch_size):
             temp=self.__class__.mlabel[i]
             result.append(pool.apply_async(calloss,(ystar[i],dstar[i],cls_conv[i],reg_conv[i],temp)))
             
             
        pool.close()
        pool.join()   
        
           
        for i in range(batch_size): 
            temp=result[i].get()
            cls_loss+=temp[0]
            num+=temp[1]
            correct_count+=temp[2]
            reg_loss+=temp[3]
            count+=temp[4] 
        
        if num>0:
           cls_loss =-cls_loss/num   
           cls_acc = correct_count/np.float32(num)
       
        if count > 0:  
           reg_loss = reg_loss/2.0/count  
           
        savefile_root1='/home/apollo/Workspace/user/caffe/imgwithbox_v3_test/org'
        savefile_root2='/home/apollo/Workspace/user/caffe/imgwithbox_v3_test/result'  
        for i in range(batch_size):
           posimg=(img[i]+0.5)*255
           posimg=np.array(np.transpose(posimg,(2, 1, 0)),dtype=np.uint8)
           save_fn1='%s/res-%06d.jpg' % (savefile_root1,i)
           save_fn2='%s/res-%06d.jpg' % (savefile_root2,i)
           plt.figure(0)
           plt.clf()
           plt.imshow(posimg)
           cv2.imwrite(save_fn1+".jpg", posimg)
           for x in range(60):
              for y in range(60):
                   if self.__class__.mlabel[i,x,y]<=0.5:
                          continue
                   cls_pred_max = np.max(cls_conv[i,:,x,y])
                   x0=cls_conv[i,0,x,y]-cls_pred_max
                   x1=cls_conv[i,1,x,y]-cls_pred_max
                   rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))       
             
                   if self.__class__.mlabel[i,x,y]>1.5:      
                         self.plot_circle([x*4,y*4],'b')
                   if ystar[i,x,y]>0.5 and rate1>0.8:       
                      self.plot_circle([x*4,y*4],'g')  
                  # if rate1>0.8:
                  #       self.plot_circle([x*4,y*4],'r')
           plt.savefig(save_fn2,dpi=100)           
         
        top[0].data[...] = np.float32(cls_loss+3*reg_loss)
        print 'classification loss = ', cls_loss, ' acc = ', cls_acc, ' regression loss = ', reg_loss

    def backward(self, top, propagate_down, bottom):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        
        cls_diff= np.zeros_like(cls_conv, dtype=np.float32)
        reg_diff = np.zeros_like(reg_conv, dtype=np.float32)
        num=0 
        count=0
    
        pool=Pool(processes=10)
        result=list()
        
        for i in range(batch_size):
             temp=self.__class__.mlabel[i]
             result.append(pool.apply_async( caldiff,(cls_conv[i],reg_conv[i],dstar[i],ystar[i],temp)))
             
        print 'diffshape',cls_diff.shape,bottom[0].diff.shape      
        pool.close()
        pool.join() 
        for i in range(batch_size):
             temp=result[i].get()
             num+=temp[2]
             count+=temp[3]
             cls_diff[i]=temp[0]
             reg_diff[i]=temp[1]

        if propagate_down[0]:
            if num>0:                    
               cls_diff/=num                               
            bottom[0].diff = cls_diff
        
        if propagate_down[1]:
            if count>0:
               reg_diff = reg_diff/count 
            bottom[1].diff = 3*reg_diff
        print 'the backward is over!'  
