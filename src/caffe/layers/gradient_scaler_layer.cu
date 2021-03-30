#include <vector>
#include "caffe/layers/gradient_scaler_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[0]) {
  	caffe_gpu_scale(top[0]->count(), -coeff_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientScalerLayer);
} // namespace caffe