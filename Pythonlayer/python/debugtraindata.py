import numpy as np
import math
import os
import cv2
import sys
import h5py
caffe_root+'/mnt/data1/hxw/caffe'
sys.path.insert(0,caffe_root+'/python')
import caffe

class TrData(caffe.Layer):
      
      imgbatch=10
      
      def flipimage(self,boximg):
          typ=np.random.randint(1,4)
          if typ==1:
             boximg=boximg[::,::-1,]
          elif typ==2:
             boximg=boximg[::-1,::-1,]
          elif typ==3:
             boximg=boximg[::-1,::-1,]
          
          return boximg          
      
      def contianorcross(self,r1,r2):
         
          if(r1[0]<=r2[0]and r1[1]<=r2[1] and r1[2]>=r2[2] and r1[3]>=r2[3] ):
             return True
          if(r1[0]>=r2[0] and r1[1]>=r2[1] and r1[2]<=r2[2]and r1[3]<=r2[3]):
            return True
          
          zx=abs(r1[0]+r1[2]-r2[0]-r2[2])
          x=abs(r1[0]-r1[2])+abs(r2[0]-r2[2])
          zy=abs(r1[1]+r1[3]-r2[1]-r2[3])     
          y=abs(r1[1]-r1[3])+abs(r2[1]-r2[3])
          if zx<=x and zy<=y:
             return True
          return False
          
      def cropimage(self,box,img):
          
          w=box[2]/2
          h=box[3]/2
          
          tempx=0
          tempy=0
          
          if box[0]-w<0:
             x1=0 
             if box[0]/float(w)<0.5:
                tempx+=w/2
          else:
             x1=box[0]-w  
             
          if box[1]-h<0:
             y1=0
             if box[1]/float(h)<0.5:
                tempy+=h/2
          else:
             y1=box[1]-h
          
          if box[0]+3*w>img.shape[1]:
             x2=img.shape[1]
             if float(img.shape[1]-box[0]-2*w)/w<0.5 and box[0]-w>=0:
                if x1-(3*w-img.shape[1]+box[0])>0:
                   x1-=(3*w-img.shape[1]+box[0])
                else:
                   x1=0
          else:
               x2=box[0]+3*w+tempx
               if x2>img.shape[1]:
                  x2=img.shape[1]
          if box[1]+3*h>img.shape[0]:
             y2=img.shape[0]
             if float(img.shape[0]-box[1]-2*h)/h<0.5 and box[1]-h>=0:
                if y1-(3*h-img.shape[0]+box[1])>0:
                   y1-=3*h-img.shape[0]+box[1]
                else:
                   y1=0
          else:
             y2=box[1]+3*h+tempy
             if y2>img.shape[0]:
                y2=img.shape[0]
          
          return [x1,y1,x2,y2]
          
      def caloverlap(self,box1,box2):
          
          if box1[0]>=box2[2] or box2[0]>=box1[2] or box1[1]>box2[3] or box2[1]>=box1[3]:
             return [0,0,0,0]
          
          xx1 = max(box1[0],box2[0])
          yy1 = max(box1[1],box2[1])
          xx2 = min(box1[2],box2[2])
          yy2 = min(box1[3],box2[3]) 
          
          return[xx1,yy1,xx2,yy2]
          
      def GettingdataAndlabel(self,boxs,img,box1,box2):
          
          data=np.array(np.zeros((3,240,240)),dtype=np.float32)
          label1=np.array(np.zeros((1,60,60)),dtype=np.float32)
          label2=np.array(np.zeros((4,60,60)),dtype=np.int32)
          label3=np.array(np.zeros((1,60,60)),dtype=np.float32)
          
          posboxs=list()
          posboxs.append(box2)
          
          for box in boxs:
              if box[0]==box2[0] and box[1]==box2[1] and box[2]==box2[2]-box2[0] and box[3]==box2[3]-box2[1]:
                 continue
              
              if self.containorcorss(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])==True:
                  [x1,y1,x2,y2]=self.caloverlap(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])
                  area1=float(abs((x2-x1)*(y2-y1)))
                  area2=float((box2[2]-box2[0])*(box2[3]-box2[1]))
                  area3=float(box[2]*box[3])
                  if area1/area2>=0.8 and area1/area2<1.25 and area1/area3>=0.75:
                     posboxs.append([x1,y1,x2,y2]) 
                     
          boximg=img[box1[1]:box1[3],box1[0]:box1[2],:]  
          boximg=cv2.resize(boximg,(240,240))
          data=np.transpose(boximg,(2,1,0))
          
          w=box1[2]-box1[0]
          h=box1[3]-box1[1]
          ratex=float(240)/w
          ratey=float(240)/h
          
          for box in posboxs:
              x1=int((box[0]-box1[0])*ratex)
              y1=int((box[1]-box1[1])*ratey)
              x2=int(x1+(box[2]-box[0])*ratex)
              y2=int(y1+(box[3]-box[1])*ratey)
              
              cntx=int((x2-x1)/2+x1)/4
              cnty=int((y2-y1)/2+y1)/4
              
              distance=int(min( (x2-x1)/4,(y2-y1)/4)*0.15)
              
              for i in range(cntx-distance,cntx+distance):
                  for j in range(cnty-distance,cnty+distance):
                      label1[0,i,j]=np.float32(1.0)
                      label2[0,i,j]=int
                                      
                                                                                  
