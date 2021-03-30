#include <vector>

#include "caffe/layers/rgb2gray_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_axes = bottom[0]->num_axes();
  if (num_axes >= 2) {
    int num_channel = bottom[0]->shape(1);
    if (num_channel == 3) {
      vector<int> blob_shape_info(num_axes);
      blob_shape_info[0] = bottom[0]->shape(0);
      blob_shape_info[1] = 1;
      if (2 < num_axes) blob_shape_info[2] = bottom[0]->shape(2);
      if (3 < num_axes) blob_shape_info[3] = bottom[0]->shape(3);
      top[0]->Reshape(blob_shape_info);
      return;
    }
  }
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num_axes = bottom[0]->num_axes();
  if (num_axes >= 2) {
    int num_channel = bottom[0]->shape(1);
    if (num_channel == 3) {
      vector<int> blob_shape_info(num_axes);
      blob_shape_info[0] = bottom[0]->shape(0);
      blob_shape_info[1] = 1;
      if (2 < num_axes) blob_shape_info[2] = bottom[0]->shape(2);
      if (3 < num_axes) blob_shape_info[3] = bottom[0]->shape(3);
      top[0]->Reshape(blob_shape_info);
      return;
    }
  }
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::Forward_cpu(
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
      Dtype* top_data = top[0]->mutable_cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      caffe_set(outer_num * pixel_num, Dtype(0), top_data);
      for (int i = 0; i < outer_num; ++i) {
        caffe_axpy(pixel_num, Dtype(0.299), bottom_data + i * inner_num, top_data + i * pixel_num);
        caffe_axpy(pixel_num, Dtype(0.587), bottom_data + i * inner_num + pixel_num, top_data + i * pixel_num);
        caffe_axpy(pixel_num, Dtype(0.114), bottom_data + i * inner_num + 2 * pixel_num, top_data + i * pixel_num);
      }
      return;
    }
  }
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void Rgb2GrayLayer<Dtype>::Backward_cpu(
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
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_set(outer_num * inner_num, Dtype(0), bottom_diff);
      for (int i = 0; i < outer_num; ++i) {
        caffe_axpy(pixel_num, Dtype(0.299), top_diff + i * pixel_num, bottom_diff + i * inner_num);
        caffe_axpy(pixel_num, Dtype(0.587), top_diff + i * pixel_num, bottom_diff + i * inner_num + pixel_num);
        caffe_axpy(pixel_num, Dtype(0.114), top_diff + i * pixel_num, bottom_diff + i * inner_num + 2 * pixel_num);
      }
      return;
    }
  }
  bottom[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(Rgb2GrayLayer);
#endif
INSTANTIATE_CLASS(Rgb2GrayLayer);
REGISTER_LAYER_CLASS(Rgb2Gray);
} // namespace caffe