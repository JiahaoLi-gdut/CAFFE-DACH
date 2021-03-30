#include <vector>
#include "caffe/layers/gradient_alignment_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  lambda_ = this->layer_param_.gradient_alignment_param().lambda();
}

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  top[0]->ShareData(*bottom[0]);
  top[1]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[0]) {
    int count = top[0]->count();
    const Dtype* top0_diff = top[0]->cpu_diff();
    const Dtype* top1_diff = top[1]->cpu_diff();
    Dtype top0_dot = caffe_cpu_dot(count, top0_diff, top0_diff);
    Dtype top1_dot = caffe_cpu_dot(count, top1_diff, top1_diff);
    bottom[0]->ShareDiff(*top[0]);
    Dtype coeff = std::max(top0_dot / top1_dot, Dtype(1.0));
    caffe_axpy(count, lambda_ * coeff, top1_diff, bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientAlignmentLayer);
#endif
INSTANTIATE_CLASS(GradientAlignmentLayer);
REGISTER_LAYER_CLASS(GradientAlignment);
} // namespace caffe