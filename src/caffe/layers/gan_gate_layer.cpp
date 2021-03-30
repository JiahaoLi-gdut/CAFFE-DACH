#include <algorithm>
#include <vector>
#include "caffe/layers/gan_gate_layer.hpp"

namespace caffe {

template <typename Dtype>
class SolverIterChangedHandlerForGANGateLayer : public Listener {
public:
  SolverIterChangedHandlerForGANGateLayer(
      GANGateLayer<Dtype>* gan_gate_layer)
      : gan_gate_layer_(gan_gate_layer) {
  }
  void handle(const void* message) {
    gan_gate_layer_->SolverIterChangedHandle(message);
  }
private:
  GANGateLayer<Dtype>* gan_gate_layer_;
};

template <typename Dtype>
void GANGateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  GANGateParameter gan_gate_param = this->layer_param_.gan_gate_param();
  generator_interval_ = gan_gate_param.generator_interval();
  generator_postpone_ = gan_gate_param.generator_postpone();
  generator_duration_ = gan_gate_param.generator_duration();
  discriminator_interval_ = gan_gate_param.discriminator_interval();
  discriminator_postpone_ = gan_gate_param.discriminator_postpone();
  discriminator_duration_ = gan_gate_param.discriminator_duration();
  SyncMessenger::AddListener("Any", "Any", "Solver", "Any", "SOLVER_ITER_CHANGED", 1, 0, 1,
      new SolverIterChangedHandlerForGANGateLayer<Dtype>(this));
}

template <typename Dtype>
void GANGateLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GANGateLayer<Dtype>::SolverIterChangedHandle(const void* message) {
  solver_iter_ = *(static_cast<const int*>(message));
}

template <typename Dtype>
void GANGateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    top[0]->ShareData(*bottom[1]);
    top[0]->ShareDiff(*bottom[1]);
  }
}

INSTANTIATE_CLASS(GANGateLayer);
REGISTER_LAYER_CLASS(GANGate);
} // namespace caffe