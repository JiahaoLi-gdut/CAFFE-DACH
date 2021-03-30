#include <vector>
#include "caffe/layers/gradient_alignment_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  top[0]->ShareData(*bottom[0]);
  top[1]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientAlignmentLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[0]) {
    int count = top[0]->count();
    const Dtype* top0_diff = top[0]->gpu_diff();
    const Dtype* top1_diff = top[1]->gpu_diff();
    Dtype top0_dot, top1_dot;
    caffe_gpu_dot(count, top0_diff, top0_diff, &top0_dot);
    caffe_gpu_dot(count, top1_diff, top1_diff, &top1_dot);
    bottom[0]->ShareDiff(*top[0]);
    Dtype coeff = max(top0_dot / top1_dot, Dtype(1.0));
    caffe_gpu_axpy(count, lambda_ * coeff, top1_diff, bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientAlignmentLayer);
} // namespace caffe