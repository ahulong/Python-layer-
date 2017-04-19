import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import h5py 
import sys
sys.path.insert(0, '/mnt/data1/zdb/work/caffe/python')
import caffe
from numpy import dtype
import os
import time
from multiprocessing import Pool
import math

#from bbox import bbox_overlaps
import cython_bbox


#def inter_over_union(a,b):
#    #a, b = [left, top, width, height]
#    a=np.float32(a)
#    b=np.float32(b)
#    inter_left = np.max([a[0],b[0]])
#    inter_top = np.max([a[1],b[1]])
#    inter_width = np.min([a[0]+a[2],b[0] + b[2]]) - inter_left;
#    inter_height = np.min([a[1]+a[3], b[1]+b[3]]) - inter_top;
#    if inter_width<0 or inter_height<0:
#        inter_area = 0
#    else:
#        inter_area = inter_width * inter_height;
#
#    union_area = a[2]*a[3] + b[2]*b[3] - inter_area;
#    return inter_area / union_area;
#
#
#def group_iou(bbs,sz):
#    feature_map_height = sz[0]
#    feature_map_width = sz[1]
#    sliding_window_height = sz[2]
#    sliding_window_width = sz[3]
#    sliding_window_stride = sz[4]
#    
#    num_bb = len(bbs)
#    iou = np.zeros((feature_map_height * feature_map_width, num_bb), dtype=np.float32)
#    for y_index in range(feature_map_height):
#        for x_index in range(feature_map_width):                    
#            h=sliding_window_height
#            w=sliding_window_width
#            x=x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
#            y=y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2
#              
#            i = y_index*feature_map_width + x_index
#            for j in range(num_bb):                        
#                iou[i,j] = inter_over_union([x,y,w,h], bbs[j])
#    return iou
#


def get_feature_size(w):
    w = int((w-1) / 2.0) + 1
    w = int(math.ceil((w-1)/2.0))+1
    w = int(math.ceil((w-1)/2.0))+1
    w = int(math.ceil((w-1)/2.0))+1
    w = 2 * (w-1) + 3 - 2*1
    #w = int(math.ceil((w-1)/2.0))+1
    return w
class FasterRCNN_Data(caffe.Layer):       
    def setup(self,bottom,top):   
        self._name_to_top_map={'data':0,'label':1,'sampling_param':2}
        #self.resize_width = 320
        #self.resize_height = 240
        self.resize_width = 400
        self.resize_height = 400
        self.batch_size = 64
        #self.sliding_window_width =  [10,10,10, 20,20,20, 30,30,30, 40,40,40, 50,50,50]
        #self.sliding_window_height = [10,15,20, 20,30,40, 30,45,60, 40,60,80, 50,75,100] 

        self.sliding_window_width =  []#[40,40,80, 60,20,20, 30,30,30, 40,40,40, 50,50,50]
        self.sliding_window_height = []#[40,80,40, 60,30,40, 30,45,60, 40,60,80, 50,75,100] 
        for area in 20**2 * 2**np.arange(5):
            for ratio in [0.5, 0.75, 1, 1.5, 2]: # x/y
                x = math.sqrt(area * ratio)
                y = x / ratio
                self.sliding_window_width.append(int(round(x)))
                self.sliding_window_height.append(int(round(y)))
        self.sliding_window_stride = 8   
        self.iou_positive_thres = 0.5
        self.iou_negative_thres = 0.4
        self.iter = 0
        self.index = 0
        self.filelist = []
        for line in open('/home/zdb/ssd/zdb/data/deepv/detection_python_layer/random_list_%s' % self.param_str):
            self.filelist.append(line.strip())
        self.total = len(self.filelist)
      
        top[0].reshape(1, 3, self.resize_height, self.resize_width)
        feature_map_height = self.resize_height / self.sliding_window_stride
        feature_map_width = self.resize_width / self.sliding_window_stride
        top[1].reshape(1, 5*len(self.sliding_window_width), feature_map_height, feature_map_width)
        top[2].reshape(self.batch_size, 7)
        self.pass_set = set()

        
    def reshape(self,bottom,top):
        pass
        #ass
                
    def forward(self,bottom,top):
        #load image
        #img_dir = '%s' % self.param_str        

        #while True:
            #dat_index = np.random.randint(387)
            #img_index = np.random.randint(512)
            #img_fn = '%s/dat-%06d/img-%06d.jpg' % (img_dir, dat_index, img_index)
            #img_fn = 
            #if os.path.exists(img_fn):
            #    break
        
#         print img_fn

        while 1:
            cal_time = 0
            import time
            if cal_time:
                start = time.time()
            img_fn = self.filelist[self.index % self.total]
            self.index += 1
            #print 'pass_set size:', len(self.pass_set)
            sys.stdout.flush()
            if (self.index-1)%self.total in self.pass_set:
                #print 'index', self.index
                #sys.stdout.flush()
                continue
            #self.index %= 100
            img = misc.imread(img_fn)
            img_height = np.shape(img)[0]
            img_width = np.shape(img)[1] 

            min_target = 500
            max_target = 1500
            ratio = min(img_height, img_width) * 1.0 / min_target
            if max(img_height, img_width) * 1.0 / ratio > max_target:
                ratio = max(img_height, img_width) * 1.0 / max_target


            self.resize_width = int(img_width / ratio)
            self.resize_height = int(img_height / ratio)

            # get output size
            top[0].reshape(1, 3, self.resize_height, self.resize_width)


            feature_map_height = get_feature_size(self.resize_height)# / self.sliding_window_stride
            feature_map_width = get_feature_size(self.resize_width)# / self.sliding_window_stride

            #print 'image size', self.resize_height, self.resize_width
            #print 'feature size', feature_map_height, feature_map_width


            top[1].reshape(1, 5*len(self.sliding_window_width), feature_map_height, feature_map_width)
            top[2].reshape(self.batch_size, 7)


            #print 'h', self.resize_height, 'w', self.resize_width
            img = misc.imresize(img,(self.resize_height, self.resize_width))
            #minv = np.min(img)
            #maxv = np.max(img)
            minv = 0
            maxv = 255
            #if minv == maxv:
            #    norm_img = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.float32)
            #else:
            #    norm_img = (np.float32(img) - minv) / (maxv - minv) - 0.5
            norm_img = ((np.float32(img)) - 128.0)/255.0
            
            top[0].data[0,:,:,:]=np.transpose(norm_img, (2,0,1))
            
            #load tag
            label_fn = img_fn.replace('JPEGImages', 'label_txt').replace('jpg', 'txt')
            bbs = []
            for line in open(label_fn):
                bbs.append(map(int, line.strip().split()[1:]))
                
                
            #tag_dir = '%s' % self.param_str
            #tag_fn = '%s/dat-%06d/bb-%06d.h5' % (tag_dir, dat_index, img_index)
            #with h5py.File(tag_fn,'r') as h5f:
            #    bbs = h5f['label'][:]
            
            #print 'bbs', bbs
            for i in range(len(bbs)):
                bbs[i][0] = np.float32(bbs[i][0])*self.resize_width/img_width
                bbs[i][2] = np.float32(bbs[i][2])*self.resize_width/img_width - bbs[i][0]
                bbs[i][1] = np.float32(bbs[i][1])*self.resize_height/img_height
                bbs[i][3] = np.float32(bbs[i][3])*self.resize_height/img_height - bbs[i][1]
            #print 'bbs resized', bbs
                
            #compute all ious  
            #feature_map_height = self.resize_height / self.sliding_window_stride
            #feature_map_width = self.resize_width / self.sliding_window_stride
                           
            #NxK N: boxes, K query boxes

            #K
            npbbs = np.array(bbs)
            npbbs[:,2] += npbbs[:,0]
            npbbs[:,3] += npbbs[:,1]
            #print 'npbbs', npbbs
            #N
            #print feature_map_width, feature_map_height, len(self.sliding_window_height)
            np_ancher = np.zeros((feature_map_width*feature_map_height*len(self.sliding_window_height),4))
            #print 'np_ancher.shape', np_ancher.shape
            ntot = feature_map_width*feature_map_height
            #xx, yy = np.meshgrid(np.arange(feature_map_height), np.arange(feature_map_width))
            xx, yy = np.meshgrid(np.arange(feature_map_width), np.arange(feature_map_height))
            #print 'xx', xx.shape
            #print 'yy', yy.shape
            assert(xx.shape[1] == feature_map_width)
            for size_index in range(len(self.sliding_window_height)):
                index = np.zeros((feature_map_height,feature_map_width,4))
                w = self.sliding_window_width[size_index]
                h = self.sliding_window_height[size_index]
                stride = self.sliding_window_stride

                index[:,:,0] = xx * stride + stride/2-1 - w/2
                index[:,:,1] = yy * stride + stride/2-1 - h/2
                #print index.shape
                #print npones.shape 
                #index[:,:,2] = w + index[:,:,0]
                #index[:,:,3] = h + index[:,:,1]
                index[:,:,2] = w * (index[:,:,0] > 0) * (index[:,:,0] + w <= self.resize_width) + index[:,:,0]
                index[:,:,3] = h * (index[:,:,1] > 0) * (index[:,:,1] + h <= self.resize_height) + index[:,:,1]
                #for y_index in range(feature_map_height):
                #    for x_index in range(feature_map_width):                    
                #        h=sliding_window_height
                #        w=sliding_window_width
                #        x=x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
                #        y=y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2
                index = index.reshape(-1, 4)
                #print 'index', index[:10,:]
                #start = 
                #print 'index.shape', index.shape
                #print 'np_ancher.shape', np_ancher.shape
                #print 'ntot', ntot
                #print 'size_index', size_index
                np_ancher[ntot*size_index:ntot*(1+size_index), :] = index

            iou = cython_bbox.bbox_overlaps(np_ancher, npbbs)
            #print iou[0,:]

            #pool = Pool(processes=4)   
            #group_iou_result = list()         
            #for size_index in range(len(self.sliding_window_height)):
            #    sz = [feature_map_height, feature_map_width, 
            #          self.sliding_window_height[size_index], 
            #          self.sliding_window_width[size_index], 
            #          self.sliding_window_stride]
            #    group_iou_result.append(pool.apply_async(group_iou,(bbs,sz)))                 
            #pool.close()
            #pool.join()
            #iou = np.zeros( (len(self.sliding_window_height) * feature_map_height * feature_map_width, len(bbs)), dtype=np.float32)
            #for i in range(len(self.sliding_window_height)):
            #    iou[i*feature_map_height * feature_map_width : (i+1)*feature_map_height * feature_map_width,:] = group_iou_result[i].get()
            
            if cal_time:
                end = time.time()
                print 'cal pos anchor part1 time:' , end - start
                start = time.time()
            #anchor box and gt box assignment
            pos_anchor=list()
            anchor_fired_bbs = list()        
            neg_anchor=list()
            bbs_fire_list = np.zeros(len(bbs),dtype=np.int8)

            xx, yy = np.meshgrid(np.arange(feature_map_width), np.arange(feature_map_height))
            for size_index in range(len(self.sliding_window_height)):
                index = np.zeros((feature_map_height,feature_map_width,7))
                w = self.sliding_window_width[size_index]
                h = self.sliding_window_height[size_index]
                stride = self.sliding_window_stride
                index[:,:,0] = xx * stride + stride/2-1 - w/2
                index[:,:,1] = yy * stride + stride/2-1 - h/2
                #index[:,:,2] = w #+ index[:,:,0]
                #index[:,:,3] = h #+ index[:,:,1]
                # out of the image, make w, h == 0, so iou will be 0, and never be picked out
                index[:,:,2] = w * (index[:,:,0] > 0) * (index[:,:,0] + w <= self.resize_width)
                index[:,:,3] = h * (index[:,:,1] > 0) * (index[:,:,1] + h <= self.resize_height)
                index[:,:,4] = xx 
                index[:,:,5] = yy
                index[:,:,6] = size_index
                index = index.reshape(-1, 7)
                anchor_index = size_index*feature_map_height*feature_map_width + yy*feature_map_width + xx
                tmp_iou = iou[anchor_index.flatten(), :]
                #print 'tmp_iou.shape', tmp_iou.shape
                max_iou = np.max(tmp_iou, 1)
                #print 'max_iou.shape', max_iou.shape
                neg_id = np.where(max_iou <= self.iou_negative_thres)[0]
                pos_id = np.where(max_iou > self.iou_positive_thres)[0]
                #print 'pos_id', pos_id
                neg_anchor.extend(list(index[neg_id,:]))
                max_id = np.argmax(tmp_iou, 1)
                #print 'max_id', max_id
                #print 'check_pos_id', max_id[pos_id]
                max_id = max_id[pos_id]

                
                #print 'size_index', size_index, max_id
                for idx, tmp in enumerate(max_id):
                    #print idx, pos_id[idx]
                    #print list(index[pos_id[idx], :])
                    pos_anchor.append(list(index[pos_id[idx], :]))
                    #print 'tmp', tmp
                    anchor_fired_bbs.append(tmp)
                    bbs_fire_list[tmp] = 1



                #for y_index in range(feature_map_height):
                #    for x_index in range(feature_map_width):
                #        h=self.sliding_window_height[size_index]
                #        w=self.sliding_window_width[size_index]
                #        x=x_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - w/2
                #        y=y_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - h/2
                #        
                #        anchor_box = [x,y,w,h, x_index, y_index, size_index]                      
                #        anchor_index = size_index*feature_map_height*feature_map_width + y_index*feature_map_width + x_index

                #        max_iou = np.max(iou[anchor_index, :])

                #        if max_iou <= self.iou_negative_thres:
                #            neg_anchor.append(anchor_box)
                #        elif max_iou >= self.iou_positive_thres:
                #            max_id = np.argmax(iou[anchor_index, :])
                #            pos_anchor.append(anchor_box)
                #            #bb_ind = fired_bb[np.random.randint(len(fired_bb))]
                #            #bb_ind = fired_bb[max_id]
                #            bb_ind = max_id
                #            anchor_fired_bbs.append(bb_ind)
                #            bbs_fire_list[bb_ind] = 1
                #                    
            
            #for j in range(len(bbs)):
            #    if bbs_fire_list[j] > 0:
            #        continue #this gt bb has been assigned an anchor box
#           #      print 'bbs[%d] is un-assigned' % j
            #    max_iou_anchor_ind = np.argmax(iou[:,j])
            #    size_index = max_iou_anchor_ind / (feature_map_height*feature_map_width)            
            #    y_index = (max_iou_anchor_ind % (feature_map_height*feature_map_width) ) / feature_map_width
            #    x_index = max_iou_anchor_ind % feature_map_width
            #    h=self.sliding_window_height[size_index]
            #    w=self.sliding_window_width[size_index]
            #    x=x_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - w/2
            #    y=y_index*self.sliding_window_stride + self.sliding_window_stride/2-1 - h/2
            #    anchor_box = [x,y,w,h, x_index, y_index, size_index]
            #    pos_anchor.append(anchor_box)
            #    anchor_fired_bbs.append(j)


            if cal_time:
                end = time.time()
                print 'cal pos anchor time:' , end - start
                start = time.time()
                                        
            assert(len(pos_anchor) == len(anchor_fired_bbs))


            pos_anchor = np.array(pos_anchor)
            anchor_fired_bbs = np.array(anchor_fired_bbs)
            neg_anchor = np.array(neg_anchor)
            
            sampling_param = np.zeros([self.batch_size, 7], dtype=np.float32)
            tags = np.zeros([1, 5*len(self.sliding_window_width),feature_map_height,feature_map_width],dtype=np.float32)
            #print 'tags.shape init', tags.shape
            
            rnd_perm = np.random.permutation(len(pos_anchor))        
            pos_anchor = pos_anchor[rnd_perm]
            anchor_fired_bbs = anchor_fired_bbs[rnd_perm]

            #start = time.time()
            #neg_anchor = np.random.permutation(neg_anchor)
            neg_index = np.random.choice(len(neg_anchor),self.batch_size)
            #end = time.time()
            #print 'choice time', end - start
            neg_anchor = neg_anchor[neg_index]
            #print 'neg_anchor', len(neg_anchor)
             
            if cal_time and 0:
                end = time.time()
                print 'permutation time: ', end - start
                start = time.time()
            pos_num_in_batch = min([self.batch_size/2,len(pos_anchor)])
            
            for i in range(pos_num_in_batch):
                x = pos_anchor[i][0]
                y = pos_anchor[i][1]
                w = pos_anchor[i][2]
                h = pos_anchor[i][3]
                x_index = pos_anchor[i][4]
                y_index = pos_anchor[i][5]
                size_index = pos_anchor[i][6]
                tags[0,0+5*size_index,y_index,x_index]=1.0
                gt = bbs[anchor_fired_bbs[i]]
                tags[0,1+5*size_index,y_index,x_index]=(gt[0] + 0.5*gt[2] - x - 0.5*w) / w
                tags[0,2+5*size_index,y_index,x_index]=(gt[1] + 0.5*gt[3] - y - 0.5*h) / h
                tags[0,3+5*size_index,y_index,x_index]=np.log(np.float32(gt[2])/w)
                tags[0,4+5*size_index,y_index,x_index]=np.log(np.float32(gt[3])/h)
                sampling_param[i,:] = pos_anchor[i]
            
            
            
            for i in range(pos_num_in_batch,self.batch_size):
                sampling_param[i,:] = neg_anchor[i - pos_num_in_batch]
                      
            #print 'tags.shape', tags.shape
            top[1].data[...]=tags    
            top[2].data[...]=sampling_param   
            self.iter += 1      
                
            
            if cal_time and 0:
                end = time.time()
                print 'data layer cost: ', end - start
            if self.iter % 10 == 0:    
    #             print 'pos = %d, bbs = %d' % (pos_num_in_batch, len(bbs))
                plt.clf()
                ax = plt.gca()
                plt.imshow(img)
                for i in range(pos_num_in_batch, self.batch_size):
                    j = i - pos_num_in_batch
                    bb=neg_anchor[j][0:4]
                    x_index=neg_anchor[j][4]
                    y_index=neg_anchor[j][5]
                    size_index=neg_anchor[j][6]
                    ax.add_patch(Rectangle((bb[0],bb[1]), bb[2], bb[3],facecolor='none',edgecolor='green'))
                for i in range(pos_num_in_batch):
                    bb=pos_anchor[i][0:4]
                    x_index=pos_anchor[i][4]
                    y_index=pos_anchor[i][5]
                    size_index=pos_anchor[i][6]
                    ax.add_patch(Rectangle((bb[0],bb[1]), bb[2], bb[3],facecolor='none',edgecolor='blue'))        
                for bb in bbs:
                    ax.add_patch(Rectangle((bb[0],bb[1]), bb[2], bb[3],facecolor='none',edgecolor='red'))
                plt.savefig('./models/fasterRCNN/rpn_stride8/assign.jpg',dpi=100)
    #             plt.show()
            if len(pos_anchor) < 5: #self.batch_size * 0.15:
                self.pass_set.add((self.index - 1) % self.total)
                continue
            else:
                break
          
           

    def backward(self,top,propagate_down,bottom):
        pass

