#include <vector>

#include "caffe/layers/bottom2param_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Bottom2ParamLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size()) {
    CHECK_EQ(bottom.size(), this->blobs_.size()) << "Incorrect number of param blobs.";
    for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
      if (bottom[bottom_index]->shape() != this->blobs_[bottom_index]->shape()) {
        Blob<Dtype> param_shaped_blob(bottom[bottom_index]->shape());
          LOG(FATAL) << "Incorrect param shape: expected shape "
            << param_shaped_blob.shape_string() << "; instead, shape was "
            << this->blobs_[bottom_index]->shape_string();
      }
    }
  } else {
    this->blobs_.resize(bottom.size());
    for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
      vector<int> param_shape = bottom[bottom_index]->shape();
      this->blobs_[bottom_index].reset(new Blob<Dtype>(param_shape));
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template <typename Dtype>
void Bottom2ParamLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
    const int count = bottom[bottom_index]->count();
    const Dtype* bottom_data = bottom[bottom_index]->cpu_data();
    Dtype* params_data = this->blobs_[bottom_index]->mutable_cpu_data();
    caffe_copy(count, bottom_data, params_data);
  }
}

template <typename Dtype>
void Bottom2ParamLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
    if (propagate_down[bottom_index]) {
      const int count = bottom[bottom_index]->count();
      const Dtype* params_diff = this->blobs_[bottom_index]->cpu_diff();
      Dtype* bottom_diff = bottom[bottom_index]->mutable_cpu_diff();
      caffe_copy(count, params_diff, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Bottom2ParamLayer);
#endif
INSTANTIATE_CLASS(Bottom2ParamLayer);
REGISTER_LAYER_CLASS(Bottom2Param);
} // namespace caffe