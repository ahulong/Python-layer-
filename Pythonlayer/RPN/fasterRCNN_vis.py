
# coding: utf-8

# In[40]:

# from xml.etree import ElementTree  
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
import h5py
import os
import time
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/panzheng/workspace/caffe_150922/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# In[10]:

# def plot_rect(bb,color):
#     ax = plt.gca()
#     ax.add_patch( Rectangle( (bb[0],bb[1]), bb[2], bb[3], facecolor='none',edgecolor=color))

def transform_bb(bb,w,h):
    bb[2]=np.exp(bb[2])*w
    bb[3]=np.exp(bb[3])*h
    bb[0]=bb[0]*w+w/2 - 0.5 * bb[2]
    bb[1]=bb[1]*h+h/2 - 0.5 * bb[3]
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
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return (boxes[pick],pick)



# In[ ]:

#only rpn

#benchmark testing
caffe.set_mode_gpu()
netroot = '/home/panzheng/workspace/caffe_150922/models/fasterRCNN/rpn_8layer'
net_rpn = caffe.Net('%s/rpn_8layer_deploy.prototxt' % netroot,
                '%s/rpn_8layer_iter_50000.caffemodel' % netroot,
                caffe.TEST)

# imgfile_root = '/media/panzheng/Data/libraD/dataset/benchmark/d4'
# savefile_root = '/media/panzheng/Data/libraD/result/d4_8layer'

# imgfile_root = '/media/panzheng/Data/HumanDetection/imgwithbox_v1/dat-000090'
# savefile_root = '/media/panzheng/Data/libraD/result/rpn_stride4_iter_heightxxx00'

# imgfile_root = '/media/panzheng/Data/libraD/dataset/benchmark/5p_10-20m/'
# savefile_root = '/media/panzheng/Data/libraD/result/5p_10-20m_8layer'

# imgfile_root = '/media/panzheng/Data/libraD/dataset/monocular'
# savefile_root = '/media/panzheng/Data/libraD/result/monocular_16ayer_lg'

# imgfile_root = '/media/panzheng/Data/HumanDetection/benchmark/no_baseline/demoroom_queue/img'
# savefile_root = '/media/panzheng/Data/libraD/result/demoroom_queue_8layer'

imgfile_root = '/media/panzheng/Data/libraD/dataset/surveillance/videoxxx'
savefile_root = '/media/panzheng/Data/libraD/result/surveillance/videoxxx'

output_width = 1280
output_height = 720
resize_width = widthxxx
resize_height = heightxxx
sliding_window_width =  [10,10,10, 30,30,30, 50,50,50]
sliding_window_height = [10,15,20, 30,45,60, 50,75,100] 
sliding_window_stride = 8   
rpn_thres = 0.7

if not os.path.exists(savefile_root):
    os.mkdir(savefile_root)
    print "%s craeated" % savefile_root

start_t = time.time()
count = 0
for i in range(0,3600):
    count += 1
#     img_fn = '%s/%d_left.jpg' % (imgfile_root, i) 
    img_fn = '%s/img-%06d.jpg' % (imgfile_root, i) 
    save_fn = '%s/res-%06d.jpg' % (savefile_root, i) 
    if not os.path.exists(img_fn):
        continue
#     print img_fn
    
    img = cv2.imread(img_fn)
    img = cv2.resize(img, (output_width,output_height))
    img_h = np.shape(img)[0]
    img_w = np.shape(img)[1]
    norm_img = cv2.resize(img,(resize_width, resize_height))   
    norm_img = np.float32(np.transpose(norm_img,(2,0,1)))
    norm_img = norm_img[(2,1,0),:,:]
    norm_img = (norm_img - np.min(norm_img))/(np.max(norm_img) - np.min(norm_img)) - 0.5
    
    net_rpn.blobs['data'].data[...] = norm_img
    out = net_rpn.forward()

    cls_res = out['conv8_cls'][0]
    reg_res = out['conv8_reg'][0]

    cand_bbs = list()
    cand_probs = list()
    for size_index in range(len(sliding_window_height)):
        h = sliding_window_height[size_index]
        w = sliding_window_width[size_index]
        for y_index in range(resize_height/sliding_window_stride):
            y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2
            for x_index in range(resize_width/sliding_window_stride):
                x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2
                prob = np.exp(cls_res[2*size_index+1, y_index, x_index])/np.sum(
                    np.exp(cls_res[2*size_index : 2*size_index+2, y_index, x_index]))
                
                if prob > rpn_thres:
                    #bb=[x,y,w,h] #anchor box
                    #predicted box
                    bb=transform_bb(reg_res[4*size_index : 4*size_index+4, y_index, x_index],w,h)+np.array([x,y,0,0])
                    cand_bbs.append(bb)
                    cand_probs.append(prob)
                    #plot_rect([bb[0]+0.5*bb[2], bb[1]+0.5*bb[3], 1, 1],(1,(1-prob)/(1-rpn_thres),(1-prob)/(1-rpn_thres)))          
                    #plot_rect(bb,(1,(1-prob)/(1-thres),(1-prob)/(1-thres)))          
                    #plt.text(bb[0],bb[1],'%.3f' % prob,color='b')
    
    if len(cand_bbs)>0:
        cand_probs = np.array(cand_probs)
        cand_bbs=np.array(cand_bbs)  
        (cand_bbs,idx_pick) = non_max_suppression_slow(cand_bbs,cand_probs,0.4)
        cand_probs = cand_probs[idx_pick]
        cand_bbs[:,0] = cand_bbs[:,0]*img_w/resize_width
        cand_bbs[:,2] = cand_bbs[:,2]*img_w/resize_width
        cand_bbs[:,1] = cand_bbs[:,1]*img_h/resize_height
        cand_bbs[:,3] = cand_bbs[:,3]*img_h/resize_height
        cand_bbs = np.int32(cand_bbs)        
        for i in range(len(cand_bbs)):
            if cand_probs[i]>0.9:
                cv2.rectangle(img,(cand_bbs[i,0],cand_bbs[i,1]),
                              (cand_bbs[i,0]+cand_bbs[i,2],cand_bbs[i,1]+cand_bbs[i,3]),
                              (255,0,0),2)
                cv2.putText(img,'%.3f' % cand_probs[i],
                            (cand_bbs[i,0],cand_bbs[i,1])
                            ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    cv2.imwrite(save_fn,img)
    
    if np.mod(count,10)==0:
        end_t = time.time()
        print '%d images have been processed. %f fps.' % (count,(count/(end_t - start_t)))
print 'Done!'
