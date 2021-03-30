#include <vector>

#include "caffe/layers/precision_recall_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.precision_recall_param().axis());
  outer_num_ = bottom[0]->count(0, axis_);
  inner_num_ = bottom[0]->count(axis_ + 1);
  channel_   = bottom[0]->shape(axis_);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.erase(top_shape.begin(), top_shape.begin() + axis_);
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
  vector<int> midd_shape = bottom[0]->shape();
  midd_shape[axis_] = 1;
  midd_.Reshape(midd_shape);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
    << "Number of labels must match number of predictions; "
    << "e.g., if axis == 1 and prediction shape is (N, C, H, W), "
    << "label count (number of labels) must be N*H*W, "
    << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  Dtype* precision_data = top[0]->mutable_cpu_data();
  Dtype* recall_data = top[1]->mutable_cpu_data();
  Dtype* midd_data = midd_.mutable_cpu_data();
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* true_data = bottom[1]->cpu_data();
  int dist_c = outer_num_ * inner_num_;
  for (int p = 0, d = 0; d < dist_c; ++d) {
    for (int c = p = 0; c < channel_; ++c) {
      if (pred_data[(d / inner_num_ * channel_ + c) * inner_num_ + d % inner_num_]
        > pred_data[(d / inner_num_ * channel_ + p) * inner_num_ + d % inner_num_]) {
        p = c;
      }
    }
    midd_data[d] = p;
  }
  int dist_o = channel_ * inner_num_;
  for (int d = 0; d < dist_o; ++d) {
    int true_pos = 0, false_pos = 0;
    int false_neg = 0/*, true_neg = 0*/;
    for (int o = 0; o < outer_num_; ++o) {
      int comp_index = o * inner_num_ + d % inner_num_;
      true_pos  += (midd_data[comp_index] == d / inner_num_ && true_data[comp_index] == d / inner_num_);
      false_pos += (midd_data[comp_index] == d / inner_num_ && true_data[comp_index] != d / inner_num_);
      false_neg += (midd_data[comp_index] != d / inner_num_ && true_data[comp_index] == d / inner_num_);
      //true_neg  += (midd_data[comp_index] != d / inner_num_ && true_data[comp_index] != d / inner_num_);
    }
    precision_data[d] = recall_data[d] = 1.0;
    if (true_pos + false_pos) precision_data[d] = (Dtype)true_pos / (true_pos + false_pos);
    if (true_pos + false_neg)    recall_data[d] = (Dtype)true_pos / (true_pos + false_neg);
  }
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PrecisionRecallLayer);
#endif
INSTANTIATE_CLASS(PrecisionRecallLayer);
REGISTER_LAYER_CLASS(PrecisionRecall);
} // namespace caffe