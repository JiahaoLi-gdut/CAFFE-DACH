#include <algorithm>
#include <vector>
#include "caffe/layers/upsampling_layer.hpp"

namespace caffe {

template <typename Dtype>
void upsampling_forward_cpu(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const Dtype* x, Dtype* y) {
  const int c_y = c_x * c_k;
  const int h_y = h_x * h_k;
  const int w_y = w_x * w_k;
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int x_w = (y_i % w_y) / w_k;
    const int x_h = (y_i / w_y % h_y) / h_k;
    const int x_c = (y_i / w_y / h_y % c_y) / c_k;
    const int x_n =  y_i / w_y / h_y / c_y;
    const int x_i = ((x_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
    y[y_i] = x[x_i];
  }
}

template <typename Dtype>
void upsampling_backward_cpu(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const Dtype* y, Dtype* x) {
  const int c_y = c_x * c_k;
  const int h_y = h_x * h_k;
  const int w_y = w_x * w_k;
  const int m_x = n * c_x * h_x * w_x;
  for (int x_i = 0; x_i < m_x; ++x_i) {
    const int x_w = x_i % w_x;
    const int x_h = x_i / w_x % h_x;
    const int x_c = x_i / w_x / h_x % c_x;
    const int x_n = x_i / w_x / h_x / c_x;
    const int b_w = x_w * w_k;
    const int b_h = x_h * h_k;
    const int b_c = x_c * c_k;
    const int e_w = b_w + w_k;
    const int e_h = b_h + h_k;
    const int e_c = b_c + c_k;
    x[x_i] = 0;
    for (int y_c = b_c; y_c < e_c; ++y_c) {
      for (int y_h = b_h; y_h < e_h; ++y_h) {
        for (int y_w = b_w; y_w < e_w; ++y_w) {
          const int y_i = ((x_n * c_y + y_c) * h_y + y_h) * w_y + y_w;
          x[x_i] += y[y_i];
        }
      }
    }
  }
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  UpsamplingParameter upsampling_param = this->layer_param_.upsampling_param();
  kernel_c_ = upsampling_param.kernel_c();
  kernel_h_ = upsampling_param.kernel_h();
  kernel_w_ = upsampling_param.kernel_w();
  CHECK_GT(kernel_c_, 0) << "kernel_c cannot be zero.";
  CHECK_GT(kernel_h_, 0) << "kernel_h cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "kernel_w cannot be zero.";
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int c_y = bottom[0]->shape(1) * kernel_c_;
  const int h_y = bottom[0]->shape(2) * kernel_h_;
  const int w_y = bottom[0]->shape(3) * kernel_w_;
  top[0]->Reshape(bottom[0]->shape(0), c_y, h_y, w_y);
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  Dtype* topper_datum = top[0]->mutable_cpu_data();
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  upsampling_forward_cpu(
    bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    kernel_c_, kernel_h_, kernel_w_, bottom_datum, topper_datum);
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  Dtype* bottom_diffs = bottom[0]->mutable_cpu_diff();
  const Dtype* topper_diffs = top[0]->cpu_diff();
  upsampling_backward_cpu(
    bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    kernel_c_, kernel_h_, kernel_w_, topper_diffs, bottom_diffs);
}

#ifdef CPU_ONLY
STUB_GPU(UpsamplingLayer);
#endif
INSTANTIATE_CLASS(UpsamplingLayer);
} // namespace caffe