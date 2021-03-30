#include <vector>

#include "caffe/layers/param2top_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Param2TopLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  for (int i = 0; i < top.size(); ++i) {
    const int filler_id = (fillers_.size() > 1) ? i : 0;
    const int refill_id = (refills_.size() > 1) ? i : 0;
    if (refills_[refill_id]) {
      fillers_[filler_id]->Fill(this->blobs_[i].get());
      if (refills_[refill_id] == 1) {
        refills_[refill_id] = 0;
      }
    }
    const int count = top[i]->count();
    const Dtype* params_data = this->blobs_[i]->gpu_data();
    Dtype* topper_data = top[i]->mutable_gpu_data();
    caffe_copy(count, params_data, topper_data);
  }
}

template <typename Dtype>
void Param2TopLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < top.size(); ++i) {
    if (this->param_propagate_down_[i]) {
      const int count = top[i]->count();
      const Dtype* topper_diff = top[i]->gpu_diff();
      Dtype* params_diff = this->blobs_[i]->mutable_gpu_diff();
      caffe_copy(count, topper_diff, params_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Param2TopLayer);
} // namespace caffe