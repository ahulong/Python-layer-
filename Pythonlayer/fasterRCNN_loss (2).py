import sys
sys.path.insert(0, '/home/zdb/work/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import time
import sys

def get_feature_size(w):
    w = int((w-1) / 4.0) + 1
    w = int(math.ceil((w-1)/2.0))+1
    w = int(math.ceil((w-1)/2.0))+1
    w = int(math.ceil((w-1)/2.0))+1
    w = 2*w - 1
    return w


def transform_bb(bb0,w,h):
    bb = np.array([0,0,0,0])
    bb[2]=np.exp(bb0[2])*w
    bb[3]=np.exp(bb0[3])*h
    bb[0]=bb0[0]*w+w/2 - 0.5 * bb[2]
    bb[1]=bb0[1]*h+h/2 - 0.5 * bb[3]
    return bb

def non_max_suppression_slow(boxes,probs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0]+boxes[:,2]-1
    y2 = boxes[:,1]+boxes[:,3]-1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / (0.01 + area[j])

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick], probs[pick], pick


class FasterRCNN_Loss(caffe.Layer):
    def setup(self, bottom, top):
        print 'setup'
        self.reg_loss_weight = 30.0  
        self.iter = 0


    def reshape(self, bottom, top):
        #print 'reshape'
        self._name_to_bottom_map={'conv_cls':0,'conv_reg':1,'label':2,'sampling_param':3,'data':4}
        top[0].reshape(1)

    def forward(self, bottom, top):
        cal_time = 0
        if cal_time:
            start = time.time()
        sampling_param = bottom[3].data
        tags = bottom[2].data
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        data = bottom[4].data
        
        batch_size = len(sampling_param)
        cls_loss = 0.0
        correct_count = 0
        probs = np.zeros(batch_size,dtype=np.float32)     
        for i in range(batch_size):
            x_index = sampling_param[i][4]
            y_index = sampling_param[i][5]
            size_index = sampling_param[i][6]
            
            cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
            cls_pred = cls_conv[0,2*size_index : 2*size_index+2 ,y_index,x_index]#real number
            cls_pred_max = np.max(cls_pred)
            cls_loss += cls_pred[int(cls_label)] - cls_pred_max - np.log(np.sum(np.exp(cls_pred - cls_pred_max))) 
            
            probs[i] = np.exp(cls_pred[1]-cls_pred_max) / np.sum(np.exp(cls_pred - cls_pred_max))
            if int(cls_pred[1] > cls_pred[0]) == int(cls_label):
                correct_count += 1
                 
        cls_loss = - cls_loss/batch_size
        cls_acc = np.float32(correct_count) /batch_size
        
        reg_loss = 0
        pos_count = 0
        for i in range(batch_size):
            x_index = sampling_param[i][4]
            y_index = sampling_param[i][5]
            size_index = sampling_param[i][6]
            
            cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
            if cls_label < 0.5:
                continue
            
            reg_label = tags[0,5*size_index+1 : 5*size_index+5 ,y_index,x_index]
            reg_pred = reg_conv[0, 4*size_index : 4*size_index+4 ,y_index,x_index]

            #if np.min(reg_label) < 0 or np.max(reg_lab)
            pos_count += 1
            reg_loss += np.sum((reg_label - reg_pred)**2)
        
        if pos_count > 0:
            reg_loss = reg_loss/2.0/pos_count            
        
        top[0].data[...] = cls_loss + self.reg_loss_weight * reg_loss
        self.iter += 1
        
        if cal_time:
            end = time.time()
            print 'loss: ff cost',  end - start
        
        if self.iter % 20 == 0:
            print '%06d: cls_loss=%.3f, acc=%.3f, reg_loss=%.3f, pos=%d' % (self.iter-1, cls_loss, cls_acc, reg_loss, pos_count)        
            sys.stdout.flush()
        if self.iter % 100 == 0:
            img0 = bottom[4].data[0,:,:,:]
            img = (np.transpose(img0, (1,2,0)) + 128.0)/ 255.0
            #print 'img', img
            
            plt.clf()
            ax = plt.gca()
            plt.imshow(img)        
            final_width = get_feature_size(data.shape[3])
            final_height= get_feature_size(data.shape[2])
            #print 'final size', final_height, final_width

            #resize_height = data.shape[2]

            sliding_window_width =  []
            sliding_window_height = []
            for area in 20**2 * 2**np.arange(5):
                for ratio in [0.5, 0.75, 1, 1.5, 2]: # x/y
                    x = math.sqrt(area * ratio)
                    y = x / ratio
                    sliding_window_width.append(int(round(x)))
                    sliding_window_height.append(int(round(y)))

            #sliding_window_width =  [10,10,10, 20,20,20, 30,30,30, 40,40,40, 50,50,50]
            #sliding_window_height = [10,15,20, 20,30,40, 30,45,60, 40,60,80, 50,75,100]
            sliding_window_stride = 8
            cand_bbs = list()
            cand_probs = list()

            #print 'len(sliding_window_height)', len(sliding_window_height)
            sys.stdout.flush()
            zdb_prob = np.zeros((len(sliding_window_height), final_width, final_height))
            for size_index in range(len(sliding_window_height)):
                #for y_index in range(resize_height/sliding_window_stride):
                #    for x_index in range(resize_width/sliding_window_stride):

                for y_index in range(final_height):
                    for x_index in range(final_width):
                        h = sliding_window_height[size_index]
                        w = sliding_window_width[size_index]
                        x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
                        y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2
                          
                        #print 'indexes', y_index, x_index
                        prob = np.exp(cls_conv[0, 2*size_index+1, y_index, x_index])/np.sum(
                            np.exp(cls_conv[0, 2*size_index : 2*size_index+2, y_index, x_index]))
                        zdb_prob[size_index, x_index, y_index] = prob
                        #print prob
                        
                       
                        if prob>0.80:   
                            bb = np.zeros(4,dtype=np.float32)          
                            for i in range(4):
                                bb[i] = reg_conv[0, 4*size_index+i, y_index, x_index]
                            bb=transform_bb(bb ,w,h) + np.array([x,y,0,0])
                            cand_bbs.append(bb)
                            cand_probs.append(prob)
                            #ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3],facecolor='none',edgecolor=(1,1-prob,1-prob)))          
                            #plt.text(bb[0],bb[1],'%.3f' % prob,color='b')
                          
                #zdb_prob.save
            import cv2
            #print 'zdb_prob.shape', zdb_prob.shape, final_height
            cv2.imwrite('./models/fasterRCNN/rpn_stride8/conv_prob_output.jpg', zdb_prob.reshape(-1, final_height * 5))
                

#             print 'bb num = %d' % len(cand_bbs)
            
            origin_cand_bbs = np.array(cand_bbs)
            origin_cand_probs = np.array(cand_probs)        
            
            cand_bbs=np.array(cand_bbs)
            cand_probs = np.array(cand_probs)        
            ind = np.argsort(cand_probs)
            ind = ind[::-1]
            
            bb_num_show = np.min([500, len(cand_bbs)])
            cand_bbs = cand_bbs[ind[:bb_num_show]]
            cand_probs = cand_probs[ind[:bb_num_show]]
            for i in range(bb_num_show):
                bb = cand_bbs[i]
                prob = cand_probs[i]
                #ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3],facecolor='none',edgecolor=(1,1-prob,1-prob)))          
                #plt.text(bb[0],bb[1],'%.3f' % prob,color='b')
            
            
            if len(origin_cand_bbs):
                nms_bbs, nms_probs, nms_ids = non_max_suppression_slow(origin_cand_bbs,origin_cand_probs,0.3)
            else:
                nms_bbs = []
                num_probs = []
                nms_ids = []
            for idx, bb in enumerate(nms_bbs):
                ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3],facecolor='none',edgecolor=(0,1,0)))        
                plt.text(bb[0],bb[1],'%.3f' % nms_probs[idx],color='b')
            plt.text(5,5,'%06d' % self.iter,color='r')
            
            plt.savefig('./models/fasterRCNN/rpn_stride8/snapshot.jpg',dpi=100)

            #import cv2
            #cv2.imshow('debug', zdb_prob.reshape(-1, final_height*5))
            #cv2.waitKey(0)
            
#             plt.show()
        
        
    def backward(self, top, propagate_down, bottom):
        
        #fid = open('bp.log','w+')
        cal_time = 0
        if cal_time:
            start = time.time()
        sampling_param = bottom[3].data
        tags = bottom[2].data
        reg_conv = bottom[1].data
        cls_conv = bottom[0].data
        data = bottom[4].data
        
        batch_size = len(sampling_param)


        final_width = get_feature_size(data.shape[3])
        final_height= get_feature_size(data.shape[2])

        sliding_window_width =  []
        sliding_window_height = []
        for area in 20**2 * 2**np.arange(5):
            for ratio in [0.5, 0.75, 1, 1.5, 2]: # x/y
                x = math.sqrt(area * ratio)
                y = x / ratio
                sliding_window_width.append(int(round(x)))
                sliding_window_height.append(int(round(y)))

        sliding_window_stride = 8
        cand_bbs = list()
        cand_probs = list()
        cand_params = list()

        dd = time.time()
        zdb_prob = np.zeros((len(sliding_window_height), final_width, final_height))

        for size_index in range(len(sliding_window_height)):
            prob = np.exp(cls_conv[0, 2*size_index+1, :, :])/np.sum(
                   np.exp(cls_conv[0, 2*size_index : 2*size_index+2, :, :]), axis=0)
            cls_label = tags[0, 5*size_index, :, :]
            idx = np.where((prob > 0.6) * (cls_label.astype(np.int32) == 0))
            #print 'idx', idx
            for y_index, x_index in zip(idx[0], idx[1]):
                bb = np.zeros(4,dtype=np.float32)          
                h = sliding_window_height[size_index]
                w = sliding_window_width[size_index]
                x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
                y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2
                for i in range(4):
                    bb[i] = reg_conv[0, 4*size_index+i, y_index, x_index]
                bb=transform_bb(bb ,w,h) + np.array([x,y,0,0])
                cand_bbs.append(bb)
                cand_probs.append(prob[y_index, x_index])
                cand_params.append([x_index, y_index, size_index])



        #for size_index in range(len(sliding_window_height)):
        #    for y_index in range(final_height):
        #        for x_index in range(final_width):
        #            h = sliding_window_height[size_index]
        #            w = sliding_window_width[size_index]
        #            x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
        #            y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2

        #            if x < 0 or y < 0 or x > data.shape[3] or y > data.shape[2]:
        #                continue
        #              
        #            #print 'indexes', y_index, x_index
        #            prob = np.exp(cls_conv[0, 2*size_index+1, y_index, x_index])/np.sum(
        #                np.exp(cls_conv[0, 2*size_index : 2*size_index+2, y_index, x_index]))
        #            zdb_prob[size_index, x_index, y_index] = prob
        #            cls_label = tags[0,5*size_index,y_index,x_index]
        #           
        #            # false positive
        #            # do not consider false negative here!
        #            if prob>0.7 and int(cls_label) == 0:   
        #                bb = np.zeros(4,dtype=np.float32)          
        #                for i in range(4):
        #                    bb[i] = reg_conv[0, 4*size_index+i, y_index, x_index]
        #                bb=transform_bb(bb ,w,h) + np.array([x,y,0,0])
        #                cand_bbs.append(bb)
        #                cand_probs.append(prob)
        #                cand_params.append([x_index, y_index, size_index])
        #
        origin_cand_bbs = np.array(cand_bbs)
        origin_cand_probs = np.array(cand_probs)        


        #print 'before nms:', time.time() - dd
        #print 'origin_cand_bbs len:', len(origin_cand_bbs)
        #print 'origin_cand_bbs', origin_cand_bbs
        #print 'origin_card_probs', origin_cand_probs
        #print 'len(origin_cand_bbs)', len(origin_cand_bbs)
        if len(origin_cand_bbs):
            nms_bbs, nms_probs, nms_ids = non_max_suppression_slow(origin_cand_bbs,origin_cand_probs,0.3)
        else:
            nms_bbs = []
            nms_probs = []
            nms_ids = []
        #print 'nms time:', time.time() - dd
        # very little in nms time

        # pass one in case that some car is not labeled! and maximum 32 negatives
        nms_ids = nms_ids[1:32]

        if propagate_down[0]:
            cls_diff = np.zeros_like(cls_conv, dtype=np.float32)        
            for i in range(batch_size):
                x_index = sampling_param[i][4]
                y_index = sampling_param[i][5]
                size_index = sampling_param[i][6]
            
                cls_label = tags[0,5*size_index,y_index,x_index]    
                cls_pred = cls_conv[0,2*size_index : 2*size_index+2 ,y_index,x_index]#real number                        
                
                if cls_pred[1] - cls_pred[0] < 20:
                    cls_diff[0,  2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[0])) - np.float32(cls_label<0.5)
                else:
                    cls_diff[0,  2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label<0.5)
                
                if cls_pred[0] - cls_pred[1] < 20:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[1])) - np.float32(cls_label>0.5)
                else:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label>0.5)
                
            for i in nms_ids:
                x_index = cand_params[i][0]
                y_index = cand_params[i][1]
                size_index = cand_params[i][2]
            
                cls_label = tags[0,5*size_index,y_index,x_index]    
                cls_pred = cls_conv[0,2*size_index : 2*size_index+2 ,y_index,x_index]#real number                        
                
                if cls_pred[1] - cls_pred[0] < 20:
                    cls_diff[0,  2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[0])) - np.float32(cls_label<0.5)
                else:
                    cls_diff[0,  2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label<0.5)
                
                if cls_pred[0] - cls_pred[1] < 20:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 1.0 / np.sum(np.exp(cls_pred - cls_pred[1])) - np.float32(cls_label>0.5)
                else:
                    cls_diff[0,1+2*size_index, y_index, x_index] += 0.0 - np.float32(cls_label>0.5)
                

            bottom[0].diff[...] = cls_diff/(batch_size + len(nms_ids))
        # nms_ids should not be considered in this part!
        if propagate_down[1]:
            reg_diff = np.zeros_like(reg_conv, dtype=np.float32)
            pos_count = 0
            for i in range(batch_size):
                x_index = sampling_param[i][4]
                y_index = sampling_param[i][5]
                size_index = sampling_param[i][6]
            
                cls_label = tags[0,5*size_index,y_index,x_index]#1 or 0
                if cls_label < 0.5:
                    continue
                pos_count += 1
                
                reg_label = tags[0,5*size_index+1 : 5*size_index+5, y_index, x_index]
                reg_pred = reg_conv[0,4*size_index : 4*size_index+4, y_index, x_index]
                reg_diff[0,4*size_index : 4*size_index+4, y_index, x_index] += reg_pred - reg_label
            
            if pos_count > 0:
                reg_diff = reg_diff/pos_count
            
            
            bottom[1].diff[...] = self.reg_loss_weight * reg_diff
        
        if cal_time:
            end = time.time()
            print 'loss bp: cost %f\n' % (end - start)
            #fid.write('loss bp: cost %f\n' % (end - start))
        #fid.close()
