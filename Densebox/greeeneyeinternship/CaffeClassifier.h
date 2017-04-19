#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include<vector>
#include<string>
#include<iostream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <ctime>
#include <cassert>
#include <algorithm>
#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
using namespace std;
using namespace cv;
using namespace caffe;

typedef std::pair<int, float> Prediction;
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

struct Bbox {
    float confidence;
    Rect rect;
    bool deleted;

};
bool mycmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

class CaffeClassifier {
 public:
  CaffeClassifier(const string& model_file,
             const string& trained_file,
             const bool use_GPU,
             const int batch_size);

  vector<Blob<float>* > PredictBatch(vector<Mat> imgs);
  void nms(vector<struct Bbox>& p, float threshold);
  vector<struct Bbox> get_final_bbox(vector<Mat> images, vector<Blob<float>* >& outputs, int sliding_window_width[],int sliding_window_height[] , float global_confidence,float enlarge_ratioh, float enlarge_ratiow, int num, long long *time1, long long *time2);
  void createDocList(vector<string> &doc_list, const string path);
  void SetMean(float a, float b, float c);
  
 private:

  void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

  void PreprocessBatch(const vector<Mat> imgs, vector<vector<Mat> >* input_batch);
 
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  cv::Mat mean_;
  bool useGPU_;
};
