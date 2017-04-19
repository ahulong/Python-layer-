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

class TeData(caffe.Layer):
     #typ represents the tpye of flip images
     imgbatch=20
     def flipimage(self,boximg):
         typ=np.random.randint(1,4)
         if typ==1:# flip the image left and right
            boximg=boximg[::,::-1,]
         elif typ==2:#flip the image bottom and top
               boximg=boximg[::-1,::,]
         elif typ==3:
               boximg=boximg[::-1,::-1,]
         return boximg
     #rotate the image and box is the person's position
     def rotateimage(self,img,box):
          if box[0]-30<=0:
             tempx1=0
             x1=box[0]
          else:
             tempx1=box[0]-30
             x1=30

          if box[1]-20<=0:
             tempy1=0
             y1=box[1]
          else:
             tempy1=box[1]-20
             y1=20

          if box[2]+30>img.shape[1]:
             tempx2=img.shape[1]
          else:
             tempx2=box[2]+30

          if box[3]+30>img.shape[0]:
             tempy2=img.shape[0]
          else:
             tempy2=box[3]+30

          x2=x1+(box[2]-box[0])
          y2=y1+(box[3]-box[1])

          boximg=img[tempy1:tempy2,tempx1:tempx2,]
          rows,cols,channles=boximg.shape
          angle=np.random.randint(15,165)
          M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
          boximg=cv2.warpAffine(boximg,M,(cols,rows))
          boximg=boximg[y1:y2,x1:x2,] 
          return boximg
     #resize the box image 
     def resizeimage(self,img,box):
         typ=np.random.randint(0,2)
         if typ==1:
            h=np.random.randint(5,11)
            w=np.random.randint(5,10)
            if (box[3]-box[1]<=4*h):
                h=0
            if (box[2]-box[0]<=4*w):
                w=0
            boximg=img[(box[1]+h):(box[3]-h),(box[0]+w):(box[2]-w),]
            return boximg
     
         h=np.random.randint(10,20)
         w=np.random.randint(10,40)
           
         if box[0]-w<=0:
              tempx1=0
         else:
            tempx1=box[0]-w

         if box[1]-h<=0:
            tempy1=0
         else:
            tempy1=box[1]-h

         if box[2]+w>img.shape[1]:
            tempx2=img.shape[1]
         else:
            tempx2=box[2]+w

         if box[3]+h>img.shape[0]:
            tempy2=img.shape[0]
         else:
            tempy2=box[3]+h

         boximg=img[tempy1:tempy2,tempx1:tempx2,]

         return boximg

     def containorcross(self,r1,r2):

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
     #crop the negtative images 
     def cropnegimage(self,boxs,img):
         j=0 
         number=len(boxs)
         posx1=np.ones(number)
         posx2=np.ones(number)
         posy1=np.ones(number)
         posy2=np.ones(number) 
         for box in boxs:
             posx1[j]=box[0]
             posx2[j]=box[2]+box[0]
             posy1[j]=box[1]
             posy2[j]=box[3]+box[1] 
             j=j+1
         w=np.random.randint(50,120)
         h=np.random.randint(60,100)
         k=0
         flag=0
         while(k<100):
              k=k+1
              y1=np.random.randint(0,img.shape[0]-h)
              x1=np.random.randint(0,img.shape[1]-w)
              x2=x1+w
              y2=y1+h
              if (x2>=img.shape[1] or y2>=img.shape[0]):
                  continue
              for t in range(number):
                  if self.containorcross([x1,y1,x2,y2],[posx1[t],posy1[t],posx2[t],posy2[t]]):
                      flag=0
                      break
                  flag=flag+1

              if flag==0:
                 continue
              elif flag==number:
                 break

         if flag==number:
            return [x1,y1,x2,y2]
         else:# return nonexistence position meaing can't find the negative images 
            return [-1,-1,-1,-1]  
     #box meaning the original box in the hf5file and img is the original image       
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
         
     #calulate the overlap positions in the crop image
     def caloverlap(self,box1,box2):
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
                  
     #calulate the data and label,box1 the crop image position, box2 the center image position, IOU 0.5
     def GettingdataAndlabel(self,boxs,img,box1,box2):
     
         data=np.array(np.zeros((3,240,240)),dtype=np.float32)
         label1=np.array(np.zeros((1,60,60)),dtype=np.float32)
         label2=np.array(np.zeros((4,60,60)),dtype=np.float32)
         label3=np.array(np.zeros((1,60,60)),dtype=np.float32)
        
         posboxs=list()
         posboxs.append(box2)          
         for box in boxs:
            if box[0]==box2[0] and box[1]==box2[1] and box[2]==box2[2]-box2[0] and box[3]==box2[3]-box2[1]:
                continue 
            if self.containorcross(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])==True:
                [x1,y1,x2,y2]=self.caloverlap(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])
                area1=float(abs((x2-x1)*(y2-y1)))
                area2=float((box2[2]-box2[0])*(box2[3]-box2[1]))
                area3=float(box[2]*box[3])
                if area1/area2>=0.8 and area1/area2<=1.25 and area1/area3>=0.7:
                   posboxs.append([x1,y1,x2,y2])
                   
         boximg=img[box1[1]:box1[3],box1[0]:box1[2],:]
         boximg=cv2.resize(boximg,(240,240),interpolation=cv2.INTER_CUBIC) 
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
             
             distance=int(min((x2-x1)/4,(y2-y1)/4)*0.15)
             for i in range(centerx-distance,centerx+distance):
                 for j in range(centery-distance,centery+distance): 
                     label1[0,i,j]=np.float32(1.0)
                     label2[0,i,j]=np.float32(i-x2/4.0)
                     label2[1,i,j]=np.float32(j-y2/4.0)
                     label2[2,i,j]=np.float32(i-x1/4.0)
                     label2[3,i,j]=np.float32(j-y1/4.0) 

             for i in range(centerx-distance-2,centerx+distance+2):
                 for j in range(centery-distance-2,centery+distance+2):
                     if label1[0,i,j]>=np.float32(0.5):
                        label3[0,i,j]=np.float32(1.0)
                     else:
                        label3[0,i,j]=np.float32(-1.0)
                         

         return[data,label1,label2,label3] 
         
     def GetRandomPatchlabel(self,boxs,img,box1):
     
         data=np.array(np.zeros((3,240,240)),dtype=np.float32)
         label1=np.array(np.zeros((1,60,60)),dtype=np.float32)
         label2=np.array(np.zeros((4,60,60)),dtype=np.float32)
         label3=np.array(np.zeros((1,60,60)),dtype=np.float32)  
         
         boximg=img[box1[1]:box1[3],box1[0]:box1[2],:]
         boximg=cv2.resize(boximg,(240,240),interpolation=cv2.INTER_CUBIC) 
         data=np.transpose(boximg,(2,1,0)) 
         
         posboxs=list()
         num=0
         maxarea=0
         for box in boxs:
             if self.containorcross(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])==True:
                [x1,y1,x2,y2]=self.caloverlap(box1,[box[0],box[1],box[0]+box[2],box[1]+box[3]])
                area1=float(abs((x2-x1)*(y2-y1)))
                area3=float(box[2]*box[3])
                if area1/area3>=0.7:
                   num+=1
                   posboxs.append([x1,y1,x2,y2])
                   if area1>maxarea:
                      maxarea=area1
                      
         if num>0:
             w=box1[2]-box1[0]
             h=box1[3]-box1[1]
             ratex=float(240)/w
             ratey=float(240)/h
             
             posnum=0 
             #compute the labels
             for box in posboxs:
           
                area1=float(abs((box[2]-box[0])*(box[3]-box[1])))
                if area1/maxarea<0.7:
                   continue
                #relative coordinates 
                x1=int((box[0]-box1[0])*ratex)
                y1=int((box[1]-box1[1])*ratey)
                x2=int(x1+(box[2]-box[0])*ratex)
                y2=int(y1+(box[3]-box[1])*ratey)
              
                centerx=int(((x2-x1)/2+x1)/4)
                centery=int(((y2-y1)/2+y1)/4)
             
                distance=int(min((x2-x1)/4,(y2-y1)/4)*0.15)
                for i in range(centerx-distance,centerx+distance):
                    for j in range(centery-distance,centery+distance): 
                        label1[0,i,j]=np.float32(1.0)
                        label2[0,i,j]=np.float32(i-x2/4.0)
                        label2[1,i,j]=np.float32(j-y2/4.0)
                        label2[2,i,j]=np.float32(i-x1/4.0)
                        label2[3,i,j]=np.float32(j-y1/4.0)  
                        posnum+=1
                
                stx=centerx-distance-2
                endx=centerx+distance-2
                sty=centery-distance-2
                endy=centery+distance+2
                if stx<0:
                   stx=0
                if endx>60:
                   endx=60
                if sty<0:
                   sty=0
                if endy>60:
                   endy=60         
                if stx<60 or endx>=0 or sty<60 or endy>=0:
                    for i in range(centerx-distance-2,centerx+distance+2):
                       for j in range(centery-distance-2,centery+distance+2):
                          if label1[0,i,j]>=np.float32(0.5):
                             label3[0,i,j]=np.float32(1.0)
                          else:
                             label3[0,i,j]=np.float32(-1.0)            
       
         return [data,label1,label2,label3]                    
             
            

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
         
         while(num<self.__class__.imgbatch):
         
              img=cv2.imread(self.param_str+self.files[0]+'/'+imagename[cur])
              length=len(imagename[cur])
              pos=h5py.File(self.param_str+self.files[0]+'/bb'+imagename[cur][3:length-4]+'.h5','r')
              boxs=np.array(pos['label'])
              cur=cur+1
              img=img/float(255)-0.5
              
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
                 
                
              for i in range(count):
                 box=boxs[i]
                 x1=box[0]
        	 y1=box[1]
        	 x2=x1+box[2]
        	 y2=y1+box[3]     
                 box1=self.cropimage(box,img)
                 [top[0].data[num],top[1].data[num],top[2].data[num],top[3].data[num]]=self.GettingdataAndlabel(boxs,img,box1,[x1,y1,x2,y2])
                 num=num+1
                 if num>=self.__class__.imgbatch:
                    break
                    
              if num>=self.__class__.imgbatch:
                    break      

              w=img.shape[1]
              h=img.shape[0]
              for i in range(count1):
                  #attention there curw and curh according to the w and h, if w and h changed , the rate also be changed
                  curw=np.random.randint(int(0.2*w),int(0.5*w))
                  curh=np.random.randint(int(0.2*h),int(0.6*h)) 
                  stx=np.random.randint(0,w-curw)
                  sty=np.random.randint(0,h-curh)
                  endx=stx+curw
                  endy=sty+curh
                  [top[0].data[num],top[1].data[num],top[2].data[num],top[3].data[num]]=self.GetRandomPatchlabel(boxs,img,[stx,sty,endx,endy]) 
                  num+=1
                  if num>=self.__class__.imgbatch:
                    break
	 print 'The train ------forward is over'

     def backward(self,top,propagate_Down,bottom):
          pass
