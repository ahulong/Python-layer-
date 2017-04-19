
import numpy as np
import math
import os
import cv2
import  matplotlib.pyplot as plt
import sys
import cv2
import h5py 
import random
caffe_root='/mnt/data1/hxw/caffe'
sys.path.insert(0,caffe_root+'/python')
import caffe
from multiprocessing import Pool

# we can use the python class to implement the train data layer and it is similar to the reading test data layer
# judging whether the two rectangle is contain or cross for IOU 
def containorcross(r1,r2):

        if(r1[0]<=r2[0] and r1[1]<=r2[1] and r1[2]>=r2[2] and r1[3]>=r2[3]):
           return True

        if(r1[0]>=r2[0] and r1[1]>=r2[1] and r1[2]<=r2[2] and r1[3]<=r2[3]):
           return True
        zx=abs(r1[0]+r1[2]-r2[0]-r2[2])
        x=abs(r1[0]-r1[2])+abs(r2[0]-r2[2])
        zy=abs(r1[1]+r1[3]-r2[1]-r2[3])
        y=abs(r1[1]-r1[3])+abs(r2[1]-r2[3])
        if zx<=x and zy<=y:
           return True
        return False
 #calulate the overlap of the two rectangles positions in the crop image 
def caloverlap(box1,box2):
         if box1[0]>=box2[2] or box2[0]>=box1[2] or box1[1]>=box2[3] or box2[1]>=box1[3]:
            return [0,0,0,0]
         if box1[0]>=box2[0] :
            x1=box1[0]     
         else: 
            x1=box2[0]

         if box1[2]<=box2[2]:
            x2=box1[2]
         else:
            x2=box2[2]

         if box1[1]>=box2[1]:
            y1=box1[1]
         else:
            y1=box2[1]
 
         if box1[3]>=box2[3]:
            y2=box2[3]
         else:
            y2=box1[3]  
         return [x1,y1,x2,y2]

 # using to implement data augmentation             
def preprocess(img):
         typ=np.random.randint(1,4)
         if typ==1:
            img=self.flipimage(img)
         elif typ==2:
            scale= np.random.uniform(0.8,1.25)
            img=cv2.resize(img,(int(img.shape[0]*scale),int(img.shape[1]*scale)),interpolation=cv2.INTER_LINEAR)
         return img 
       #calulate the data and label,box1 the crop image position, box2 the center image position, IOU 0.7 in kitti car detection for the postive patch                     
def GettingdataAndlabel(boxs,img,box1,box2):
     
         data=np.array(np.zeros((3,240,240)),dtype=np.float32)
         label1=np.array(np.zeros((1,60,60)),dtype=np.float32)
         label2=np.array(np.zeros((4,60,60)),dtype=np.float32)
         label3=np.array(np.zeros((1,60,60)),dtype=np.float32)
        
         # find all the positive boudingbox in the patch which should be in scale range of 0.8~1.25 relative to the input patch's width or height      
         posboxs=list()
         posboxs.append(box2)          
         for box in boxs:
            if box[0]==box2[0] and box[1]==box2[1] and box[2]==box2[2]-box2[0] and box[3]==box2[3]-box2[1]:
                continue 
            if containorcross(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])==True:
                [x1,y1,x2,y2]=caloverlap(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])
                area1=float(abs((x2-x1)*(y2-y1)))
                area2=float((box2[2]-box2[0])*(box2[3]-box2[1]))
                area3=float(box[2]*box[3])
                bw=(x2-x1)/float(box1[2]-box1[0])
                bh=(y2-y1)/float(box1[3]-box1[1])
                if bw>bh:
                   if bh>=1.0/6 and bh<=63.0/240.0 and area1/area3>=0.5:
                        posboxs.append([x1,y1,x2,y2])
                else:
                   if bw>=1.0/6 and bw<=63.0/240.0 and area1/area3>=0.5:
                        posboxs.append([x1,y1,x2,y2]) 
                   
         boximg=img[box1[1]:box1[3],box1[0]:box1[2],:]
         #boximg=self.preprocess(boximg)
         boximg=cv2.resize(boximg,(240,240),interpolation=cv2.INTER_LINEAR) 
         data=np.transpose(boximg,(2,1,0)) 
         
         w=box1[2]-box1[0]
         h=box1[3]-box1[1]
         ratex=float(240)/w
         ratey=float(240)/h 
         
         #compute the labels
         for box in posboxs:
             #relative coordinates 
             x1=int((box[0]-box1[0])*ratex)
             y1=int((box[1]-box1[1])*ratey)
             x2=int(x1+(box[2]-box[0])*ratex)
             y2=int(y1+(box[3]-box[1])*ratey)
             
             centerx=int(((x2-x1)/2+x1)/4)
             centery=int(((y2-y1)/2+y1)/4)
             
             #label the positive pixels
             distance=int(min((x2-x1)/4,(y2-y1)/4)*0.3)
             for i in range(centerx-distance,centerx+distance+1):
                 for j in range(centery-distance,centery+distance+1): 
                     label1[0,i,j]=np.float32(1.0)
                     label2[0,i,j]=i-x2/4
                     label2[1,i,j]=j-y2/4
                     label2[2,i,j]=i-x1/4
                     label2[3,i,j]=j-y1/4 
                     
             stx=max(centerx-distance-2,0)
             endx=min(centerx+distance+3,60)
             sty=max(centery-distance-2,0)
             endy=min(centery+distance+3,60)
             
             #label the nonpositve and nonnegtive pixels and label the positive pixels is selected
             for i in range(stx,endx):
                 for j in range(sty,endy):
                     if label1[0,i,j]>=np.float32(0.5):
                        label3[0,i,j]=np.float32(1.0)
                     else:
                        label3[0,i,j]=np.float32(-1.0)          
           
         label2/=12.5     
         return[data,label1,label2,label3] #data is the input patch, laebl1 is y*,label2 is the d*,label3 is the M(t) which means the pixel is selected.
         
def GetRandomPatchlabel(boxs,img,box1):
     
         data=np.array(np.zeros((3,240,240)),dtype=np.float32)
         label1=np.array(np.zeros((1,60,60)),dtype=np.float32)
         label2=np.array(np.zeros((4,60,60)),dtype=np.float32)
         label3=np.array(np.zeros((1,60,60)),dtype=np.float32)  
         
         boximg=img[box1[1]:box1[3],box1[0]:box1[2],:]
         #boximg=self.preprocess(boximg)
         boximg=cv2.resize(boximg,(240,240),interpolation=cv2.INTER_LINEAR) 
         data=np.transpose(boximg,(2,1,0)) 
         
         # search whether there exist the positive boudingbox in the random patch
         posboxs=list()
         num=0
         maxarea=0
         maxh=0
         maxw=0
         for box in boxs:
             if containorcross(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])==True:
                [x1,y1,x2,y2]=caloverlap(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])
                area1=float(abs((x2-x1)*(y2-y1)))
                area3=float(box[2]*box[3])
                if area1/area3>=0.5:  
                   ratew=(x2-x1)/float(box1[2]-box1[0]) 
                   rateh=(y2-y1)/float(box1[3]-box1[1]) 
                   if ratew>rateh:
                      if rateh>=1.0/6 and rateh<=63.0/240.0:
                          num+=1
                          posboxs.append([x1,y1,x2,y2])
                   else:
                      if ratew>=1.0/6 and ratew<=63.0/240.0:
                          num+=1
                          posboxs.append([x1,y1,x2,y2])      
                            
        
        
         #if there exists the positive smaples             
         if num>0:
             w=box1[2]-box1[0]
             h=box1[3]-box1[1]
             ratex=240.0/w
             ratey=240.0/h
             
             #compute the labels
             for box in posboxs:
                #relative coordinates 
                x1=int((box[0]-box1[0])*ratex)
                y1=int((box[1]-box1[1])*ratey)
                x2=int(x1+(box[2]-box[0])*ratex)
                y2=int(y1+(box[3]-box[1])*ratey)
              
                centerx=int(((x2-x1)/2+x1)/4)
                centery=int(((y2-y1)/2+y1)/4)
             
                distance=int(min((x2-x1)/4,(y2-y1)/4)*0.3)
                for i in range(centerx-distance,centerx+distance+1):
                    for j in range(centery-distance,centery+distance+1): 
                        label1[0,i,j]=np.float32(1.0)
                        label2[0,i,j]=np.float32(i-x2/4.0)
                        label2[1,i,j]=np.float32(j-y2/4.0)
                        label2[2,i,j]=np.float32(i-x1/4.0)
                        label2[3,i,j]=np.float32(j-y1/4.0)  

                        
                stx=max(centerx-distance-2,0)
                endx=min(centerx+distance+3,60)
                sty=max(centery-distance-2,0)
                endy=min(centery+distance+3,60)
             
                for i in range(stx,endx):
                   for j in range(sty,endy):
                      if label1[0,i,j]>=np.float32(0.5):
                         label3[0,i,j]=np.float32(1.0)
                      else:
                         label3[0,i,j]=np.float32(-1.0)
         
         return [data,label1,label2,label3]    

class TrData(caffe.Layer):
     #typ represents the tpye of flip images
     imgbatch=40
     def flipimage(self,boximg):
         typ=np.random.randint(1,4)
         if typ==1:# flip the image left and right
            boximg=boximg[::,::-1,]
         elif typ==2:#flip the image bottom and top
               boximg=boximg[::-1,::,]
         elif typ==3:
               boximg=boximg[::-1,::-1,]
         return boximg
    

    
        #computing the positions of patch in the original image we should keep the smaller width or heigth of the boduingbox has roughly 50 pixels in the 240*240 input patch         
     def cropimage(self,box,img):
         
         scale=1.0#np.random.uniform(0.8,1.25)
         if box[2]>box[3]:
            rate=(box[3]*24/(5.0*scale))/img.shape[0]
         else:
            rate=(box[2]*24/(5.0*scale))/img.shape[1]  
         
         lastw=int(rate*img.shape[0]/2)
         lasth=int(rate*img.shape[1]/2)      

         tempx=0
         if box[0]-lastw<0:
            x1=0
            tempx+=lastw-box[0]
               
         else:
            x1=box[0]-lastw
               
         if box[0]+box[2]+lastw>img.shape[1]:
         
            x2=img.shape[1]
            x1-=(lastw+box[0]+box[2]-img.shape[1])
            if x1<0:
               x1=0
         else:
              x2=box[0]+box[2]+lastw+tempx
              if x2>img.shape[1]:
                 x2=img.shape[1]   

         tempy=0
         if box[1]-lasth<0:
            y1=0
            tempy+=lasth-box[1]
               
         else:
            y1=box[1]-lasth
               
         if box[1]+box[2]+lasth>img.shape[0]:
            y2=img.shape[0]
            y1-=(box[1]+box[3]+lasth-img.shape[0])
            if y1<0:
               y1=0
         else:
              y2=box[1]+box[3]+lasth+tempy
              if y2>img.shape[0]:
                 y2=img.shape[0]     
         return [x1,y1,x2,y2]
         
     #calulate the overlap positions in the crop image
                 
             
              
     
     def setup(self,bottom,top):
         print 'set up entering'
         self._name_to_top_map={'data':0,'ylabel':1,'dlabel':2,'mlabel':3}
         self.files=os.listdir(self.param_str)
         print 'setup is over'

     def reshape(self,bottom,top):
         top[0].reshape(self.__class__.imgbatch,3,240,240)
         top[1].reshape(self.__class__.imgbatch,1,60,60)        
         top[2].reshape(self.__class__.imgbatch,4,60,60) 
         top[3].reshape(self.__class__.imgbatch,1,60,60)
             
     def forward(self,bottom,top):
         self.files=np.random.permutation(self.files)
         filesname=os.listdir(self.param_str+self.files[0])
         imagename=list()
         for subf in filesname:
             length=len(subf)
             if cmp(subf[length-4:length],'.jpg')==0:
                imagename.append(subf) 
         imagename=np.random.permutation(imagename)
  
         num=0
         cur=0
         pool=Pool(processes=10)
         result=list()
         
         while num<self.__class__.imgbatch:
              #reading the bounding box label
              img=cv2.imread(self.param_str+self.files[0]+'/'+imagename[cur])
              length=len(imagename[cur])
              pos=h5py.File(self.param_str+self.files[0]+'/bb'+imagename[cur][3:length-4]+'.h5','r')
              boxs=np.array(pos['label'])
              random.shuffle(boxs)
              cur=cur+1
              img=img/float(255)-0.5
                 
              random.shuffle(boxs)
              left=self.__class__.imgbatch-num
              
              if left<2*len(boxs):
                  count=left/2
                  if left%2==1:
                     count1=count+1
                  else:
                     count1=count    
              else:
                  count=len(boxs)
                  count1=count    
                 
              #get the positive patches  
              for i in range(count):
                 box=boxs[i]
                 x1=box[0]
                 y1=box[1]
        	 x2=x1+box[2]
        	 y2=y1+box[3]     
                 box1=self.cropimage(box,img)
                 result.append(pool.apply_async(GettingdataAndlabel,(boxs,img,box1,[x1,y1,x2,y2])))
                 #[top[0].data[num],top[1].data[num],top[2].data[num],top[3].data[num]]=self.GettingdataAndlabel(boxs,img,box1,[x1,y1,x2,y2])
                 num+=1
       
                    
          

              w=img.shape[1]
              h=img.shape[0]
              
              #get the random patches
              for i in range(count1):
              
                  #attention there curw and curh according to the w and h, if w and h changed , the rate also be changed
                  curw=np.random.randint(int(0.2*w),int(0.8*w))
                  curh=np.random.randint(int(0.4*h),int(h)) 
                  stx=np.random.randint(0,w-curw)
                  sty=np.random.randint(0,h-curh)
                  endx=stx+curw
                  endy=sty+curh
                  result.append(pool.apply_async(GetRandomPatchlabel,(boxs,img,[stx,sty,endx,endy])))
                  #[top[0].data[num],top[1].data[num],top[2].data[num],top[3].data[num]]=self.GetRandomPatchlabel(boxs,img,[stx,sty,endx,endy]) 
                  num+=1
         
         pool.close()
         pool.join()         
         for i in range(self.__class__.imgbatch): 
            [top[0].data[i],top[1].data[i],top[2].data[i],top[3].data[i]]=result[i].get() 
            #print 'poolresiul=',result[i].get()        
             
	 print 'The train ------forward is over'

     def backward(self,top,propagate_Down,bottom):
          pass
