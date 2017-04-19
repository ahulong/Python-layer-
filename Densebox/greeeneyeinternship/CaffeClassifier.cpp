#include"CaffeClassifier.h"
int output_width = 1280, output_height = 720, resize_width = 480, resize_height = 300;

CaffeClassifier::CaffeClassifier(const string& model_file,
                       const string& trained_file,
                       const bool use_GPU,
                       const int batch_size) {
   if (use_GPU) {
       Caffe::set_mode(Caffe::GPU);
       Caffe::SetDevice(0);
       useGPU_ = true;
   }
   else {
       Caffe::set_mode(Caffe::CPU);
       useGPU_ = false;
   }

  /* Set batchsize */
  batch_size_ = batch_size;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  
}



// predict single image forward function
vector<Blob<float>* > CaffeClassifier::PredictBatch(vector< cv::Mat > imgs) {

  Blob<float>* input_layer = net_->input_blobs()[0];  
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);
  
  float* input_data = input_layer->mutable_cpu_data();
  int cnt = 0;
  for(int i = 0; i < imgs.size(); i++) {
    Mat sample;
    Mat img = imgs[i];

    if (img.channels() == 3 && num_channels_ == 1)
        cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cvtColor(img, sample, CV_RGBA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;
    
   
    if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
         resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
    }
    double img_min, img_max;
    //Point minLoc; 
    //Point maxLoc;
    minMaxLoc(img, &img_min, &img_max);//, &minLoc, &maxLoc);
    for(int k = 0; k < sample.channels(); k++) {
        for(int i = 0; i < sample.rows; i++) {
            for(int j = 0; j < sample.cols; j++) {
               input_data[cnt] = (float(sample.at<uchar>(i,j*3+k))-img_min)/float(img_max-img_min)-0.5;//normalization the input image
               cnt += 1;
            }
        }
    }
  }
  
  /* Forward dimension change to all layers. */
  net_->Reshape();
  
  
  struct timeval start;
  gettimeofday(&start, NULL);

  net_->ForwardPrefilled();
  if(useGPU_) {
    cudaDeviceSynchronize();
  }

  struct timeval end;
  gettimeofday(&end, NULL);
 // cout << "pure model predict time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;

  /* Copy the output layer to a std::vector */
  vector<Blob<float>* > outputs;
  for(int i = 0; i < net_->num_outputs(); i++) {
    Blob<float>* output_layer = net_->output_blobs()[i];
    outputs.push_back(output_layer);
  }
  return outputs;
}



void CaffeClassifier::nms(vector<struct Bbox>& p, float threshold) {

   sort(p.begin(), p.end(), mycmp);
   int cnt = 0;
   for(int i = 0; i < p.size(); i++) {
   
     if(p[i].deleted) continue;
     cnt += 1;
     for(int j = i+1; j < p.size(); j++) {
     
       if(!p[j].deleted) {
         cv::Rect intersect = p[i].rect & p[j].rect;
         float iou = intersect.area() * 1.0/p[j].rect.area(); /// (p[i].rect.area() + p[j].rect.area() - intersect.area());
         if (iou > threshold) {
           p[j].deleted = true;
         }
       }
     }
   }
   //cout << "[after nms] left is " << cnt << endl;
}


vector<struct Bbox> CaffeClassifier::get_final_bbox(vector<Mat> images, vector<Blob<float>* >& outputs, int sliding_window_width[],int sliding_window_height[] , float global_confidence,float enlarge_ratioh, float enlarge_ratiow, int num, long long *time1, long long *time2) {
    
    Blob<float>* cls = outputs[0];
    Blob<float>* reg = outputs[1];
     
    cls->Reshape(cls->num(), cls->channels(), cls->height(), cls->width());
    reg->Reshape(reg->num(), reg->channels(), reg->height(), reg->width());

    assert(cls->num() == reg->num());

    assert(cls->channels() == 18);
    assert(reg->channels() == 36);

    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());
    
    vector<struct Bbox> vbbox;
   
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();
    int sliding_window_stride = 8 ;
    int w,h;
    struct timeval start1;
    gettimeofday(&start1, NULL); 
  
      
    
    for(int i = 0; i < cls->num(); i++) {  // = batchsize 
         { 
            int skip = cls->height() * cls->width();
            for (int j=0; j<num; j++)
            {
              h = sliding_window_height[j];
              w = sliding_window_width[j] ; 
              for (int y_index=0; y_index<int(resize_height/sliding_window_stride);y_index++)
              {
                int y = y_index*sliding_window_stride + sliding_window_stride/2-1 - h/2;
                for (int x_index=0; x_index<int(resize_width/sliding_window_stride); x_index++)
                {
                   int x = x_index*sliding_window_stride + sliding_window_stride/2-1 - w/2;
                   
                   float x0=cls_cpu[i*skip*4+2*j*skip+y_index*cls->width()+x_index];
                   float x1=cls_cpu[i*skip*4+(2*j+1)*skip+y_index*cls->width()+x_index];

                   float max_01=max(x1,x0);
                   x0 -= max_01;
                   x1 -= max_01;
                   float prob= exp(x1)/(exp(x1)+exp(x0));
                   float rect[4] = {};
                   if(prob > global_confidence){
                             
                        rect[2]=exp(reg_cpu[i*36*skip+(j*4+2)*skip+y_index*reg->width()+x_index])*w;
                        rect[3]=exp(reg_cpu[i*36*skip+(j*4+3)*skip+y_index*reg->width()+x_index])*h;
               
                        rect[0]=reg_cpu[i*36*skip+j*4*skip+y_index*reg->width()+x_index];   
                        rect[1]=reg_cpu[i*36*skip+(j*4+1)*skip+y_index*reg->width()+x_index];
                   
                        rect[0]=rect[0]*w+w/2-rect[2]/2+x;
                        rect[1]=rect[1]*h+h/2-rect[3]/2+y; 
                        struct Bbox bbox;
                        bbox.confidence = prob;
                        bbox.rect = Rect(rect[0], rect[1], rect[2], rect[3]);
                        assert(images.size() == 1);
                        bbox.rect &= Rect(0,0,images[0].cols, images[0].rows);
                        bbox.deleted = false;
                        vbbox.push_back(bbox);
                    }
                }
              }
            } 
            
        }
    }
    struct timeval end1;
    gettimeofday(&end1, NULL);
   // cout<<end1.tv_sec<<" "<< start1.tv_sec<<" " << end1.tv_usec <<" "<< start1.tv_usec<<endl;
    (*time1)+=(1000000*(end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec); 
   cout<<"time1"<<*time1<<endl;
   // cout<<"time1"<<*time1+1<<endl;
    struct timeval start2;
    gettimeofday(&start2, NULL);   
    if (vbbox.size()!=0){
    
      sort(vbbox.begin(), vbbox.end(), mycmp);
      nms(vbbox, 0.4);
    }
    struct timeval end2;
    gettimeofday(&end2, NULL);  
    (*time2)+=(1000000*(end2.tv_sec - start2.tv_sec) + end2.tv_usec - start2.tv_usec); 
    //cout << "[debug nms passed!]" << endl;
    vector<struct Bbox> final_vbbox;
    
    for(int i = 0; i < vbbox.size(); i++) {
    
        if(!vbbox[i].deleted && vbbox[i].confidence > global_confidence) {
            struct Bbox box = vbbox[i];
            float x = box.rect.x * enlarge_ratiow;
            float y = box.rect.y * enlarge_ratioh;
            float w = box.rect.width *enlarge_ratiow;
            float h = box.rect.height* enlarge_ratioh;
            //cout<<"hw"<<enlarge_ratiow<<" "<<enlarge_ratioh;
            box.rect.x = x;
            box.rect.y = y;
            box.rect.width = w;
            box.rect.height = h;
            if (vbbox[i].confidence > global_confidence){
                final_vbbox.push_back(box);
            }
        }
    }
    return final_vbbox;
}

//reading  images from one file
void CaffeClassifier::createDocList(vector<string> &doc_list, const string path){
   
    //cout<<"enter"<<endl;
    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(path.c_str());
    
  if (dpdf != NULL){
     while (epdf = readdir(dpdf)){
      string name=string(epdf->d_name);
      // for jpg format image
      if (name[name.length()-1]!='g')
         continue;
      doc_list.push_back(path+string(name));
   }
    closedir(dpdf);
}else{
      cout<<"the path is empty"<<endl;
}   

}

/***read image from one image file***/
void Imagefileprocess()
{
      
    Mat image;
    string model_file   = "/home/apollo/Workspace/user/caffe/RPN/rpn_16layer_lg_deploy.prototxt"; //  rcnn_googlenet.prototxt
    string trained_file = "/home/apollo/Workspace/user/caffe/RPN/rpn_16layer_lg_iter_600000.caffemodel"; //
    CaffeClassifier classifier_single(model_file, trained_file, true, 1);
    vector<Mat> images;
    float global_confidence = 0.9;
    string imgfile_root = "/home/apollo/Workspace/user/caffe/vid1/";
    string savefile_root = "/home/apollo/Workspace/user/caffe/vid1result/";
    struct stat fileStat;
    assert ((stat(dir.c_str(imgfile_root), &file-Stat) == 0) && S_ISDIR(fileStat.st_mode));
    assert ((stat(dir.c_str(savefile_root), &fileStat) == 0) && S_ISDIR(fileStat.st_mode));
    vector<string> imagelist;
    

    float enlarge_ratioh= output_height*1.0/resize_height  ,enlarge_ratiow=output_width*1.0/resize_width;
    int sliding_window_width[9] =  {10,10,10, 30,30,30, 50,50,50};
    int sliding_window_height[9] = {10,15,20, 30,45,60, 50,75,100}; 
    int sliding_window_stride = 8;
    float rpn_thres = 0.4;
    
    classifier_single.createDocList(imagelist,imgfile_root);
    vector<string>::iterator iter;
    sort(imagelist.begin(),imagelist.end());
    struct timeval start;
    gettimeofday(&start, NULL);
    long long timepre=0;
    long long timecalf=0;
    long long timecalnms=0;
    for(iter=imagelist.begin();iter!=imagelist.end();iter++)
    {
      image = imread(*iter,1);
      //namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
      //mshow("Display Image", image);
      //waitKey(0);
      //cout<<float(image.at<uchar>(2,2))<<endl;
      //cout<<*iter<<endl;
      
     
    
     // cout << "filename " << *iter << endl;
      if (image.empty()) {
            cout << "Wrong Image" << endl;
            continue;
        }
        
       Mat img;
       img = image.clone();
     
       resize(img,img,Size(output_width,output_height));
             
       Mat norm_img;
       resize(img, norm_img,Size(resize_width, resize_height)); 
       cvtColor(norm_img, norm_img, CV_RGBA2BGR);  
       images.push_back(norm_img);
         
       struct timeval start1;
       gettimeofday(&start1, NULL); 
       vector<Blob<float>* > outputs = classifier_single.PredictBatch(images);
       struct timeval end1;
       gettimeofday(&end1, NULL);
       timepre+=(1000000*(end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec); 
       
       vector<struct Bbox> result =classifier_single.get_final_bbox(images, outputs,sliding_window_width,sliding_window_height,rpn_thres, enlarge_ratioh, enlarge_ratiow,9, &timecalf,&timecalnms);

       for(int bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
                if (result[bbox_id].confidence>0.4){
                    rectangle(image, result[bbox_id].rect, Scalar(0,255,0),3);
                    string info="head";
                    putText(image, info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y), CV_FONT_HERSHEY_COMPLEX, 0.5,  Scalar(0,0,255));
                }
            }
            
    
       // struct timeval end;
        //gettimeofday(&end, NULL);
        //cout << "total time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 <<"ms" <<endl;
      // string imgname=*iter;
      // imwrite(savefile_root+imgname.substr((*iter).find_last_of('/')+1), image);
       // cout<<"after save"<<endl;
       // imshow("debug.jpg", image);
       // waitKey(1000/(26*7)); 
        images.pop_back();
    }
    struct timeval end;
    gettimeofday(&end, NULL);
    cout << "total time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 <<"ms" <<endl;
    cout<<"timepre"<<timepre/1000<<" "<<"timecalf"<<timecalf/1000<<" "<<"timecalnms"<<timecalnms/1000<<endl;
}


int main(int argc, char** argv )
{
  google::InitGoogleLogging(argv[0]);
  Imagefileprocess();
    return 0;
}
