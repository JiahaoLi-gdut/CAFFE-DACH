#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "caffe/layers/auc_layer.hpp"

namespace caffe {

template <typename Dtype>
void AUCLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auc_axis_  = bottom[0]->CanonicalAxisIndex(this->layer_param_.auc_param().axis());
  outer_num_ = bottom[0]->count(0, auc_axis_);
  inner_num_ = bottom[0]->count(auc_axis_ + 1);
  channel_   = bottom[0]->shape(auc_axis_);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.erase(top_shape.begin(), top_shape.begin() + auc_axis_);
  top[0]->Reshape(top_shape);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
    << "Number of labels must match number of predictions; "
    << "e.g., if auc axis == 1 and prediction shape is (N, C, H, W), "
    << "label count (number of labels) must be N*H*W, "
    << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
void AUCLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  Dtype* auc_data = top[0]->mutable_cpu_data();
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* true_data = bottom[1]->cpu_data();
  int dim = channel_ * inner_num_;
  vector<std::pair<Dtype, int> > sorted_pred(outer_num_);
  for (int d = 0; d < dim; ++d) {
    for (int o = 0; o < outer_num_; ++o) {
      sorted_pred[o] = std::make_pair(pred_data[o * dim + d], o);
    }
    std::sort(sorted_pred.begin(), sorted_pred.end());
    Dtype total_auc = 0.; // auc value.
    int total_pos = 0; // total positive samples.
    int total_neg = 0; // total negative samples.
    int count_pos = 0; // count positive samples in one block with same prediction prob. 
    int count_sam = 0; // count samples in one block with same prediction prob.
    int ranks_sum = 0; // sum rank of sample in one block with same prediction prob.
    Dtype last_pred = pred_data[sorted_pred[0].second * dim + d]; // last prediction prob.
    for (int o = 0; o < outer_num_; ++o) {
      int label_hit = static_cast<int>(true_data[sorted_pred[o].second * inner_num_ + d % inner_num_]) == d / inner_num_;
      label_hit ? ++total_pos : ++total_neg;
      if (last_pred != pred_data[sorted_pred[o].second * dim + d]) {
        total_auc += Dtype(count_pos * ranks_sum) / count_sam;
        count_sam = 1;
        ranks_sum = o + 1;
        count_pos = label_hit;
        last_pred = pred_data[sorted_pred[o].second * dim + d];
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
void AUCLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AUCLayer);
#endif
INSTANTIATE_CLASS(AUCLayer);
REGISTER_LAYER_CLASS(AUC);
} // namespace caffe