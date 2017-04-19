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


class dloss(caffe.Layer):

    def getKey(self,x):
        return x[0]
    def plot_circle(self,bb,col):
        ax=plt.gca()
        circ=plt.Circle((bb[0], bb[1]), radius=2, color=col)
        ax.add_patch(circ)
        
    def setup(self, bottom, top):
        if len(bottom) != 5:
            raise Exception("Need 6 inputs to compute the loss.")
        self._name_to_bottom_map={'conv5_2_det':0,'conv5_2_loc':1,'ylabel':2,'dlabel':3,'data':4}
        
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0]
        height=bottom[0].data.shape[3]
        width=bottom[0].data.shape[2] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        img=bottom[4].data

        cls_loss = 0.0
        correct_count = 0.0    
        num=0
        for i in range(batch_size):
            for x in range(width):
                for y in range(height):
                    if  ystar[i,x,y]<=-0.5:
                           continue
                    cls_pred_max = max(cls_conv[i,0,x,y],cls_conv[i,1,x,y])
                    x0=cls_conv[i,0,x,y]-cls_pred_max
                    x1=cls_conv[i,1,x,y]-cls_pred_max
                    rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
                    rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                    num+=1
                    if ystar[i,x,y]>=0.5:   
                       cls_loss+=x1-np.log(np.exp(x1)+np.exp(x0))
                       if rate1>=0.5:
                          correct_count+=1
                    else:
                       cls_loss+= x0-np.log(np.exp(x1)+np.exp(x0))            
                       if rate0>=0.5:
                          correct_count+=1

        if num>0:
           cls_loss =-cls_loss/num   
           cls_acc = correct_count/np.float32(num)
        else:
           savefile_root1='/home/apollo/Workspace/user/caffe/KITTI/store/orig'
           savefile_root2='/home/apollo/Workspace/user/caffe/KITTI/store/result1'
            
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
                      
                      cls_pred_max = max(cls_conv[i,0,x,y],cls_conv[i,1,x,y])
                      x0=cls_conv[i,0,x,y]-cls_pred_max
                      x1=cls_conv[i,1,x,y]-cls_pred_max
                      rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                      if ystar[i,x,y]>0.5:
                         self.plot_circle([x*4,y*4],'b')
                      if  self.__class__.mlabel[i,x,y]>=1.5:
                           #ate1>0.9:
                         x2=int(x-reg_conv[i,0,x,y])*4
                         y2=int(y-reg_conv[i,1,x,y])*4
                         x1=int(x-reg_conv[i,2,x,y])*4
                         y1=int(y-reg_conv[i,3,x,y])*4 
                         self.plot_circle([x*4,y*4],'r')
                      # self.plot_rect([x1,y1,x2,y2],'b') 
                 
               plt.savefig(save_fn2,dpi=100)     
        
        reg_loss = np.float32(0.0)
        count=0
        for i in range(batch_size):
            for x in range(width):
                for y in range(height):
                      if ystar[i,x,y]<0.5:
                            continue
                      count += 1      
                      for channel in range(4):  
                        reg_loss+=math.pow((reg_conv[i,channel,x,y]-float(dstar[i,channel,x,y])),2)  
 
        if count > 0:  
           reg_loss = reg_loss/2.0/count       
                             
        top[0].data[...] = np.float32(cls_loss+3*reg_loss)
        print 'classification loss = ', cls_loss, ' acc = ', cls_acc, ' regression loss = ', reg_loss

    def backward(self, top, propagate_down, bottom):
        ystar=bottom[2].data[:,0,:,:]
        dstar=bottom[3].data[:,:,:,:]
        batch_size=bottom[0].data.shape[0] 
        height=bottom[0].data.shape[3]
        width=bottom[0].data.shape[2] 
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        
        if propagate_down[0]:
            cls_diff= np.zeros_like(cls_conv, dtype=np.float32) 
            num=0       
            for i in range(batch_size):
                for x in range(width):
                    for y in range(height):
                         if ystar[i,x,y]<=-0.5:
                             continue
                         cls_pred_max = np.max(cls_conv[i,:,x,y])
                         x0=cls_conv[i,0,x,y]-cls_pred_max
                         x1=cls_conv[i,1,x,y]-cls_pred_max
                         rate0=np.exp(x0)/(np.exp(x0)+np.exp(x1))
                         rate1=np.exp(x1)/(np.exp(x0)+np.exp(x1))
                         num+=1 
                         if ystar[i,x,y]>0.5: 
                            cls_diff[i,0,x,y]+=rate0
                            cls_diff[i,1,x,y]+=rate1-1
                         else:
                            cls_diff[i,0,x,y]+=rate0-1
                            cls_diff[i,1,x,y]+=rate1 
         
            if num>0:                    
               cls_diff/=num                               
            bottom[0].diff[...] = cls_diff
        
        if propagate_down[1]:
            reg_diff = np.zeros_like(reg_conv, dtype=np.float32)
            count=0
            for i in range(batch_size):
                for x in range(width):
                    for y in range(height):
                        if ystar[i,x,y]<0.5:
                             continue
                        count +=1     
                        for channel in range(4):     
                            reg_diff[i,channel,x,y]+=(reg_conv[i,channel,x,y]-dstar[i,channel,x,y])
                            
                   
            if count>0:
               reg_diff = reg_diff/count 
            bottom[1].diff[...] = 3*reg_diff
        print 'the backward is over!'  
