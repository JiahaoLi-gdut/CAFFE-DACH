#include <algorithm>
#include <vector>
#include "caffe/layers/precision_recall_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void argmax_gpu_for_channel(
    int outer_num, int channel, int inner_num,
    const Dtype* pred_data, Dtype* midd_data) {
  int dist_c = outer_num * inner_num;
  CUDA_KERNEL_LOOP(d, dist_c) {
    int p = 0;
    for (int c = 0; c < channel; ++c) {
      if (pred_data[(d / inner_num * channel + c) * inner_num + d % inner_num]
        > pred_data[(d / inner_num * channel + p) * inner_num + d % inner_num]) {
        p = c;
      }
    }
    midd_data[d] = p;
  }
}

template <typename Dtype>
__global__ void precision_recall_gpu_for_outer(
    int outer_num, int channel, int inner_num,
    const Dtype* midd_data, const Dtype* true_data,
    Dtype* precision_data, Dtype* recall_data) {
  int dist_o = channel * inner_num;
  CUDA_KERNEL_LOOP(d, dist_o) {
    int true_pos = 0, false_pos = 0;
    int false_neg = 0/*, true_neg = 0*/;
    for (int o = 0; o < outer_num; ++o) {
      int comp_index = o * inner_num + d % inner_num;
      true_pos  += (midd_data[comp_index] == d / inner_num && true_data[comp_index] == d / inner_num);
      false_pos += (midd_data[comp_index] == d / inner_num && true_data[comp_index] != d / inner_num);
      false_neg += (midd_data[comp_index] != d / inner_num && true_data[comp_index] == d / inner_num);
      //true_neg  += (midd_data[comp_index] != d / inner_num && true_data[comp_index] != d / inner_num);
    }
    precision_data[d] = recall_data[d] = 1.0;
    if (true_pos + false_pos) precision_data[d] = (Dtype)true_pos / (true_pos + false_pos);
    if (true_pos + false_neg)    recall_data[d] = (Dtype)true_pos / (true_pos + false_neg);
  }
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  Dtype* precision_data = top[0]->mutable_gpu_data();
  Dtype* recall_data = top[1]->mutable_gpu_data();
  Dtype* midd_data = midd_.mutable_gpu_data();
  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* true_data = bottom[1]->gpu_data();
  argmax_gpu_for_channel<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_num_, channel_, inner_num_, pred_data, midd_data);
  precision_recall_gpu_for_outer<Dtype><<<CAFFE_GET_BLOCKS(channel_ * inner_num_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_num_, channel_, inner_num_, midd_data, true_data, precision_data, recall_data);
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PrecisionRecallLayer);
} // namespace caffe