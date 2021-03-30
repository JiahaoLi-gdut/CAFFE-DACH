#include <vector>
#include "caffe/layers/exp_scaler_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExpScalerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* top_diff = top[0]->mutable_gpu_diff();
  caffe_gpu_sign(count, bottom_data, top_diff);
  caffe_gpu_abs(count, bottom_data, top_data);
  if (log_base_ != Dtype(1)) {
    caffe_gpu_scal(count, log_base_, top_data);
  }
  caffe_gpu_exp(count, top_data, top_data);
  caffe_gpu_add_scalar(count, Dtype(-1), top_data);
  caffe_gpu_scal(count, scale_, top_data);
  caffe_gpu_mul(count, top_diff, top_data, top_data);
}

template <typename Dtype>
void ExpScalerLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_abs(count, bottom_data, bottom_diff);
  if (log_base_ != Dtype(1)) {
    caffe_gpu_scal(count, log_base_, bottom_diff);
    caffe_gpu_exp(count, bottom_diff, bottom_diff);
    caffe_gpu_scal(count, scale_ * log_base_, bottom_diff);
  }
  else {
    caffe_gpu_exp(count, bottom_diff, bottom_diff);
    caffe_gpu_scal(count, scale_, bottom_diff);
  }
  caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpScalerLayer);
} // namespace caffe