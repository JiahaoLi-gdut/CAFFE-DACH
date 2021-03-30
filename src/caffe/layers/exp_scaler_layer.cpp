#include <vector>
#include "caffe/layers/exp_scaler_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExpScalerLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const Dtype base = this->layer_param_.exp_scaler_param().base();
  if (base != Dtype(-1)) CHECK_GT(base, 0) << "base must be strictly positive.";
  log_base_ = (base == Dtype(-1)) ? Dtype(1) : log(base);
  CHECK(!isnan(log_base_)) << "NaN result: log(base) = log(" << base << ") = " << log_base_;
  CHECK(!isinf(log_base_)) << "Inf result: log(base) = log(" << base << ") = " << log_base_;
  scale_ = this->layer_param_.exp_scaler_param().scale();
  CHECK_NE(scale_, 0) << "scale cannot be zero.";
}

template <typename Dtype>
void ExpScalerLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_diff = top[0]->mutable_cpu_diff();
  caffe_cpu_sign(count, bottom_data, top_diff);
  caffe_abs(count, bottom_data, top_data);
  if (log_base_ != Dtype(1)) {
    caffe_scal(count, log_base_, top_data);
  }
  caffe_exp(count, top_data, top_data);
  caffe_add_scalar(count, Dtype(-1), top_data);
  caffe_scal(count, scale_, top_data);
  caffe_mul(count, top_diff, top_data, top_data);
}

template <typename Dtype>
void ExpScalerLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_abs(count, bottom_data, bottom_diff);
  if (log_base_ != Dtype(1)) {
    caffe_scal(count, log_base_, bottom_diff);
    caffe_exp(count, bottom_diff, bottom_diff);
    caffe_scal(count, scale_ * log_base_, bottom_diff);
  }
  else {
    caffe_exp(count, bottom_diff, bottom_diff);
    caffe_scal(count, scale_, bottom_diff);
  }
  caffe_mul(count, top_diff, bottom_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(ExpScalerLayer);
#endif
INSTANTIATE_CLASS(ExpScalerLayer);
REGISTER_LAYER_CLASS(ExpScaler);
} // namespace caffe