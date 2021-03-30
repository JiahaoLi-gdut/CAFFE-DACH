#include <algorithm>
#include <vector>

#include "caffe/layers/vanilla_gan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class SolverIterChangedHandlerForVanillaGANLossLayer : public Listener {
public:
  SolverIterChangedHandlerForVanillaGANLossLayer(
      VanillaGANLossLayer<Dtype>* vanilla_gan_loss_layer)
      : vanilla_gan_loss_layer_(vanilla_gan_loss_layer) {
  }
  void handle(const void* message) {
    vanilla_gan_loss_layer_->SolverIterChangedHandle(message);
  }
private:
  VanillaGANLossLayer<Dtype>* vanilla_gan_loss_layer_;
};

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  VanillaGANLossParameter vanilla_gan_loss_param = this->layer_param_.vanilla_gan_loss_param();
  bottom_blob_axis_ = bottom[0]->CanonicalAxisIndex(vanilla_gan_loss_param.bottom_blob_axis());
  generator_interval_ = vanilla_gan_loss_param.generator_interval();
  generator_postpone_ = vanilla_gan_loss_param.generator_postpone();
  generator_duration_ = vanilla_gan_loss_param.generator_duration();
  discriminator_interval_ = vanilla_gan_loss_param.discriminator_interval();
  discriminator_postpone_ = vanilla_gan_loss_param.discriminator_postpone();
  discriminator_duration_ = vanilla_gan_loss_param.discriminator_duration();
  outer_numb_ = bottom[0]->count(0, bottom_blob_axis_);
  inner_numb_ = bottom[0]->count() / outer_numb_;
  SyncMessenger::AddListener("Any", "Any", "Solver", "Any", "SOLVER_ITER_CHANGED", 1, 0, 1,
      new SolverIterChangedHandlerForVanillaGANLossLayer<Dtype>(this));
}

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  for (int i = bottom_blob_axis_; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  middle_blob_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::SolverIterChangedHandle(const void* message) {
  solver_iter_ = *(static_cast<const int*>(message));
}

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    if (preforward_flag) {
      // the loss log(1-D(G(z))) for discriminator.
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* middle_data = middle_blob_.mutable_cpu_data();
      Dtype* middle_diff = middle_blob_.mutable_cpu_diff();
      Dtype* top_data = top[0]->mutable_cpu_data();
      caffe_set(outer_numb_ * inner_numb_, Dtype(1), middle_data);
      caffe_axpy(outer_numb_ * inner_numb_, Dtype(-1), bottom_data, middle_data); // 1-D(G(z))
      caffe_powx(outer_numb_ * inner_numb_, middle_data, Dtype(-1), middle_diff); // 1/(1-D(G(z)))
      caffe_log(outer_numb_ * inner_numb_, middle_data, middle_data); // log(1-D(G(z)))
      caffe_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
    else {
      // the loss log(D(x)) for discriminator.
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* middle_data = middle_blob_.mutable_cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      caffe_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
      caffe_log(outer_numb_ * inner_numb_, bottom_data, middle_data); // log(D(x))
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // the loss log(D(G(z))) for discriminator.
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* middle_data = middle_blob_.mutable_cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_log(outer_numb_ * inner_numb_, bottom_data, middle_data); // log(D(G(z)))
    caffe_set(inner_numb_, Dtype(0), top_data);
    for (int i = 0; i < outer_numb_; ++i) {
      caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
    }
  }
}

template <typename Dtype>
void VanillaGANLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    // ascending stochastic gradient, so add minus sign before the gradient to be descended.
    const Dtype* bottom_data = bottom[0]->cpu_data();  // D(x)
    const Dtype* middle_diff = middle_blob_.cpu_diff(); // 1/(1-D(G(z)))
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_powx(outer_numb_ * inner_numb_, bottom_data, Dtype(-1), bottom_diff);  // 1/D(x)
    caffe_sub(outer_numb_ * inner_numb_, middle_diff, bottom_diff, bottom_diff); // -[1/D(x)-1/(1-D(G(z)))] = 1/(1-D(G(z)))-1/D(x)
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // ascending stochastic gradient, so add minus sign before the gradient to be descended.
    const Dtype* bottom_data = bottom[0]->cpu_data(); // D(G(z))
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_powx(outer_numb_ * inner_numb_, bottom_data, Dtype(-1), bottom_diff); // 1/D(G(z))
    caffe_scal(outer_numb_ * inner_numb_, Dtype(-1), bottom_diff); // -1/D(G(z))
  }
}

#ifdef CPU_ONLY
STUB_GPU(VanillaGANLossLayer);
#endif
INSTANTIATE_CLASS(VanillaGANLossLayer);
REGISTER_LAYER_CLASS(VanillaGANLoss);
} // namespace caffe