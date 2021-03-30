#include <algorithm>
#include <vector>
#include "caffe/layers/vanilla_gan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    if (preforward_flag) {
      // the loss log(1-D(G(z))) for discriminator.
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* middle_data = middle_blob_.mutable_gpu_data();
      Dtype* middle_diff = middle_blob_.mutable_gpu_diff();
      Dtype* top_data = top[0]->mutable_gpu_data();
      caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(1), middle_data);
      caffe_gpu_axpy(outer_numb_ * inner_numb_, Dtype(-1), bottom_data, middle_data); // 1-D(G(z))
      caffe_gpu_powx(outer_numb_ * inner_numb_, middle_data, Dtype(-1), middle_diff); // 1/(1-D(G(z)))
      caffe_gpu_log(outer_numb_ * inner_numb_, middle_data, middle_data); // log(1-D(G(z)))
      caffe_gpu_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
    else {
      // the loss log(D(x)) for discriminator.
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* middle_data = middle_blob_.mutable_gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      caffe_gpu_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
      caffe_gpu_log(outer_numb_ * inner_numb_, bottom_data, middle_data); // log(D(x))
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // the loss log(D(G(z))) for discriminator.
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* middle_data = middle_blob_.mutable_gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    caffe_gpu_log(outer_numb_ * inner_numb_, bottom_data, middle_data); // log(D(G(z)))
    caffe_gpu_set(inner_numb_, Dtype(0), top_data);
    for (int i = 0; i < outer_numb_; ++i) {
      caffe_gpu_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
    }
  }
}

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    // ascending stochastic gradient, so add minus sign before the gradient to be descended.
    const Dtype* bottom_data = bottom[0]->gpu_data();  // D(x)
    const Dtype* middle_diff = middle_blob_.gpu_diff(); // 1/(1-D(G(z)))
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_powx(outer_numb_ * inner_numb_, bottom_data, Dtype(-1), bottom_diff);  // 1/D(x)
    caffe_gpu_sub(outer_numb_ * inner_numb_, middle_diff, bottom_diff, bottom_diff); // -[1/D(x)-1/(1-D(G(z)))] = 1/(1-D(G(z)))-1/D(x)
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // ascending stochastic gradient, so add minus sign before the gradient to be descended.
    const Dtype* bottom_data = bottom[0]->gpu_data(); // D(G(z))
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_powx(outer_numb_ * inner_numb_, bottom_data, Dtype(-1), bottom_diff); // 1/D(G(z))
    caffe_gpu_scal(outer_numb_ * inner_numb_, Dtype(-1), bottom_diff); // -1/D(G(z))
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(VanillaGANLossLayer);
} // namespace caffe