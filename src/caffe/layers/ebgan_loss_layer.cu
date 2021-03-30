#include <algorithm>
#include <vector>
#include "caffe/layers/ebgan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EBGANLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    if (preforward_flag) {
      // the loss m-D(G(z)) for discriminator.
      const Dtype* bottom_data = bottom[0]->gpu_data(); // D(G(z))
      Dtype* middle_data = middle_blob_.mutable_gpu_data();
      Dtype* middle_diff = middle_blob_.mutable_gpu_diff();
      Dtype* top_data = top[0]->mutable_gpu_data();
      caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(positive_margin_), middle_data); // m
      caffe_gpu_axpy(outer_numb_ * inner_numb_, Dtype(-1), bottom_data, middle_data); // m-D(G(z))
      caffe_gpu_sign(outer_numb_ * inner_numb_, middle_data, middle_diff); // sign(m-D(G(z))) with values {-1,1}
      caffe_gpu_add_scalar(outer_numb_ * inner_numb_, Dtype(1), middle_diff); // sign(m-D(G(z))) with values {0,2}
      caffe_gpu_scal(outer_numb_ * inner_numb_, Dtype(0.5), middle_diff); // sign(m-D(G(z))) with values {0,1}
      caffe_gpu_mul(outer_numb_ * inner_numb_, middle_diff, middle_data, middle_data); // max(0, m-D(G(z)))
      caffe_gpu_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
    else {
      // the loss D(x) for discriminator.
      const Dtype* bottom_data = bottom[0]->gpu_data(); // D(x)
      const Dtype* middle_data = middle_blob_.mutable_gpu_data(); // max(0, m-D(G(z)))
      Dtype* top_data = top[0]->mutable_gpu_data();
      caffe_gpu_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
        caffe_gpu_axpy(inner_numb_, Dtype(1), bottom_data + i * inner_numb_, top_data);
      }
    }
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // the loss D(G(z)) for discriminator.
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    caffe_gpu_set(inner_numb_, Dtype(0), top_data);
    for (int i = 0; i < outer_numb_; ++i) {
      caffe_gpu_axpy(inner_numb_, Dtype(1), bottom_data + i * inner_numb_, top_data);
    }
  }
}

template <typename Dtype>
void EBGANLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    // descending stochastic gradient.
    const Dtype* middle_diff = middle_blob_.gpu_diff(); // sign(m-D(G(z))) with values {0,1}
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(1), bottom_diff); // 1.
    caffe_gpu_axpy(outer_numb_ * inner_numb_, Dtype(-1), middle_diff, bottom_diff);
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // descending stochastic gradient.
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(1), bottom_diff); // 1.
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EBGANLossLayer);
} // namespace caffe