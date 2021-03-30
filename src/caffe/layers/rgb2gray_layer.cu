#include <vector>

#include "caffe/layers/rgb2gray_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  int num_axes = bottom[0]->num_axes();
  if (num_axes >= 2) {
    int num_channel = bottom[0]->shape(1);
    if (num_channel == 3) {
      int outer_num = bottom[0]->shape(0);
      int inner_num = bottom[0]->count(1);
      int pixel_num = bottom[0]->count(2);
      Dtype* top_data = top[0]->mutable_gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      caffe_gpu_set(outer_num * pixel_num, Dtype(0), top_data);
      for (int i = 0; i < outer_num; ++i) {
        caffe_gpu_axpy(pixel_num, Dtype(0.299), bottom_data + i * inner_num, top_data + i * pixel_num);
        caffe_gpu_axpy(pixel_num, Dtype(0.587), bottom_data + i * inner_num + pixel_num, top_data + i * pixel_num);
        caffe_gpu_axpy(pixel_num, Dtype(0.114), bottom_data + i * inner_num + 2 * pixel_num, top_data + i * pixel_num);
      }
      return;
    }
  }
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  int num_axes = bottom[0]->num_axes();
  if (num_axes >= 2) {
    int num_channel = bottom[0]->shape(1);
    if (num_channel == 3) {
      int outer_num = bottom[0]->count(0, 1);
      int inner_num = bottom[0]->count(1);
      int pixel_num = bottom[0]->count(2);
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      caffe_gpu_set(outer_num * inner_num, Dtype(0), bottom_diff);
      for (int i = 0; i < outer_num; ++i) {
        caffe_gpu_axpy(pixel_num, Dtype(0.299), top_diff + i * pixel_num, bottom_diff + i * inner_num);
        caffe_gpu_axpy(pixel_num, Dtype(0.587), top_diff + i * pixel_num, bottom_diff + i * inner_num + pixel_num);
        caffe_gpu_axpy(pixel_num, Dtype(0.114), top_diff + i * pixel_num, bottom_diff + i * inner_num + 2 * pixel_num);
      }
      return;
    }
  }
  bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_LAYER_GPU_FUNCS(Rgb2GrayLayer);
} // namespace caffe