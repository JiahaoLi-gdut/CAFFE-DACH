#include <algorithm>
#include <vector>
#include "caffe/layers/upsampling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void upsampling_forward_gpu(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const Dtype* x, Dtype* y) {
  const int c_y = c_x * c_k;
  const int h_y = h_x * h_k;
  const int w_y = w_x * w_k;
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int x_w = (y_i % w_y) / w_k;
    const int x_h = (y_i / w_y % h_y) / h_k;
    const int x_c = (y_i / w_y / h_y % c_y) / c_k;
    const int x_n =  y_i / w_y / h_y / c_y;
    const int x_i = ((x_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
    y[y_i] = x[x_i];
  }
}

template <typename Dtype>
__global__ void upsampling_backward_gpu(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const Dtype* y, Dtype* x) {
  const int c_y = c_x * c_k;
  const int h_y = h_x * h_k;
  const int w_y = w_x * w_k;
  const int m_x = n * c_x * h_x * w_x;
  CUDA_KERNEL_LOOP(x_i, m_x) {
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
void UpsamplingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  Dtype* topper_datum = top[0]->mutable_gpu_data();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  upsampling_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
  	bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    kernel_c_, kernel_h_, kernel_w_, bottom_datum, topper_datum);
}

template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  const Dtype* topper_diffs = top[0]->gpu_diff();
  upsampling_backward_gpu<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
  	bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    kernel_c_, kernel_h_, kernel_w_, topper_diffs, bottom_diffs);
}

INSTANTIATE_LAYER_GPU_FUNCS(UpsamplingLayer);
} // namespace caffe