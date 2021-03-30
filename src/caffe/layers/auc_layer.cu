#include <algorithm>
#include <vector>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "caffe/layers/auc_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void auc_gpu_for_outer(
    int outer_num, int channel, int inner_num,
    const Dtype* pred_data, const Dtype* true_data,
    Dtype* sorted_pred_key, int* sorted_pred_val, Dtype* auc_data) {
  int dim = channel * inner_num;
  CUDA_KERNEL_LOOP(d, dim) {
    Dtype* cur_sorted_pred_key = sorted_pred_key + d * outer_num;
    int*   cur_sorted_pred_val = sorted_pred_val + d * outer_num;
    for (int o = 0; o < outer_num; ++o) {
      cur_sorted_pred_key[o] = pred_data[o * dim + d];
      cur_sorted_pred_val[o] = o;
    }
    thrust::sort_by_key(thrust::device, cur_sorted_pred_key, cur_sorted_pred_key + outer_num, cur_sorted_pred_val);
    Dtype total_auc = 0.; // auc value.
    int total_pos = 0; // total positive samples.
    int total_neg = 0; // total negative samples.
    int count_pos = 0; // count positive samples in one block with same prediction prob. 
    int count_sam = 0; // count samples in one block with same prediction prob.
    int ranks_sum = 0; // sum rank of sample in one block with same prediction prob.
    Dtype last_pred = pred_data[cur_sorted_pred_val[0] * dim + d]; // last prediction prob.
    for (int o = 0; o < outer_num; ++o) {
      int label_hit = static_cast<int>(true_data[cur_sorted_pred_val[o] * inner_num + d % inner_num]) == d / inner_num;
      label_hit ? ++total_pos : ++total_neg;
      if (last_pred != pred_data[cur_sorted_pred_val[o] * dim + d]) {
        total_auc += Dtype(count_pos * ranks_sum) / count_sam;
        count_sam = 1;
        ranks_sum = o + 1;
        count_pos = label_hit;
        last_pred = pred_data[cur_sorted_pred_val[o] * dim + d];
      }
      else {
        count_sam += 1;
        ranks_sum += o + 1;
        count_pos += label_hit;
      }
    }
    total_auc += Dtype(count_pos * ranks_sum) / count_sam;
    total_auc -= Dtype(total_pos * (total_pos + 1)) / 2;
    total_auc /= Dtype(total_pos * total_neg);
    auc_data[d] = total_auc;
  }
}

template <typename Dtype>
void AUCLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  void* sorted_pred;
  Dtype* auc_data = top[0]->mutable_gpu_data();
  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* true_data = bottom[1]->gpu_data();
  CUDA_CHECK(cudaMalloc(&sorted_pred, outer_num_ * channel_ * inner_num_ * (sizeof(Dtype) + sizeof(int))));
  Dtype* sorted_pred_key = (Dtype*)sorted_pred;
  int*   sorted_pred_val = (int*)(sorted_pred_key + outer_num_ * channel_ * inner_num_);
  auc_gpu_for_outer<Dtype><<<CAFFE_GET_BLOCKS(channel_ * inner_num_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_num_, channel_, inner_num_, pred_data, true_data, sorted_pred_key, sorted_pred_val, auc_data);
  CUDA_CHECK(cudaFree(sorted_pred));
}

template <typename Dtype>
void AUCLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AUCLayer);
} // namespace caffe