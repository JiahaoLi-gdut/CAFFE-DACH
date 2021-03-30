#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif // USE_OPENCV

#include <vector>
#include <ostream>

#include "caffe/layers/image_storage_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ImageStorageLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ImageStorageParameter image_storage_param = this->layer_param_.image_storage_param();
  encoder_  = image_storage_param.encoder();
  saveroot_ = image_storage_param.saveroot();
}

template <typename Dtype>
void ImageStorageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
#ifdef USE_OPENCV
  const int outer_numb = bottom[0]->shape(0);
  const int inner_numb = bottom[0]->count() / outer_numb;
  ostringstream os;
  vector<Dtype> buffer(inner_numb);
  const Dtype* data_start = bottom[0]->cpu_data();
  for (int i = 0; i < outer_numb; ++i) {
    const Dtype* data_end = data_start + inner_numb;
    std::copy(data_start, data_end, &buffer[0]);
    data_start = data_end;
    cv::Mat cvmat(buffer, false);
    cvmat = cvmat.reshape(3, bottom[0]->shape(2));
    os << saveroot_ << "/frame_" << i << "." << encoder_;
    imwrite(os.str(), cvmat);
  }
#endif // USE_OPENCV
}

template <typename Dtype>
void ImageStorageLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_CLASS(ImageStorageLayer);
REGISTER_LAYER_CLASS(ImageStorage);
} // namespace caffe