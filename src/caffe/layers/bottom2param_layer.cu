#include <vector>

#include "caffe/layers/bottom2param_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Bottom2ParamLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
    const int count = bottom[bottom_index]->count();
    const Dtype* bottom_data = bottom[bottom_index]->gpu_data();
    Dtype* params_data = this->blobs_[bottom_index]->mutable_gpu_data();
    caffe_copy(count, bottom_data, params_data);
  }
}

template <typename Dtype>
void Bottom2ParamLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int bottom_index = 0; bottom_index < bottom.size(); ++bottom_index) {
    if (propagate_down[bottom_index]) {
      const int count = bottom[bottom_index]->count();
      const Dtype* params_diff = this->blobs_[bottom_index]->gpu_diff();
      Dtype* bottom_diff = bottom[bottom_index]->mutable_gpu_diff();
      caffe_copy(count, params_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Bottom2ParamLayer);
} // namespace caffe