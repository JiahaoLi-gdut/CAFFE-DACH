#include <algorithm>
#include <vector>

#include "caffe/layers/ebgan_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class SolverIterChangedHandlerForEBGANLossLayer : public Listener {
public:
  SolverIterChangedHandlerForEBGANLossLayer(
      EBGANLossLayer<Dtype>* ebgan_loss_layer)
      : ebgan_loss_layer_(ebgan_loss_layer) {
  }
  void handle(const void* message) {
    ebgan_loss_layer_->SolverIterChangedHandle(message);
  }
private:
  EBGANLossLayer<Dtype>* ebgan_loss_layer_;
};

template <typename Dtype>
void EBGANLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  EBGANLossParameter ebgan_loss_param = this->layer_param_.ebgan_loss_param();
  bottom_blob_axis_ = bottom[0]->CanonicalAxisIndex(ebgan_loss_param.bottom_blob_axis());
  generator_interval_ = ebgan_loss_param.generator_interval();
  generator_postpone_ = ebgan_loss_param.generator_postpone();
  generator_duration_ = ebgan_loss_param.generator_duration();
  discriminator_interval_ = ebgan_loss_param.discriminator_interval();
  discriminator_postpone_ = ebgan_loss_param.discriminator_postpone();
  discriminator_duration_ = ebgan_loss_param.discriminator_duration();
  positive_margin_ = ebgan_loss_param.positive_margin();
  outer_numb_ = bottom[0]->count(0, bottom_blob_axis_);
  inner_numb_ = bottom[0]->count() / outer_numb_;
  SyncMessenger::AddListener("Any", "Any", "Solver", "Any", "SOLVER_ITER_CHANGED", 1, 0, 1,
      new SolverIterChangedHandlerForEBGANLossLayer<Dtype>(this));
}

template <typename Dtype>
void EBGANLossLayer<Dtype>::Reshape(
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
void EBGANLossLayer<Dtype>::SolverIterChangedHandle(const void* message) {
  solver_iter_ = *(static_cast<const int*>(message));
}

template <typename Dtype>
void EBGANLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    if (preforward_flag) {
      // the loss m-D(G(z)) for discriminator.
      const Dtype* bottom_data = bottom[0]->cpu_data(); // D(G(z))
      Dtype* middle_data = middle_blob_.mutable_cpu_data();
      Dtype* middle_diff = middle_blob_.mutable_cpu_diff();
      Dtype* top_data = top[0]->mutable_cpu_data();
      caffe_set(outer_numb_ * inner_numb_, Dtype(positive_margin_), middle_data); // m
      caffe_axpy(outer_numb_ * inner_numb_, Dtype(-1), bottom_data, middle_data); // m-D(G(z))
      caffe_cpu_sign(outer_numb_ * inner_numb_, middle_data, middle_diff); // sign(m-D(G(z))) with values {-1,1}
      caffe_add_scalar(outer_numb_ * inner_numb_, Dtype(1), middle_diff); // sign(m-D(G(z))) with values {0,2}
      caffe_scal(outer_numb_ * inner_numb_, Dtype(0.5), middle_diff); // sign(m-D(G(z))) with values {0,1}
      caffe_mul(outer_numb_ * inner_numb_, middle_diff, middle_data, middle_data); // max(0, m-D(G(z)))
      caffe_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
      }
    }
    else {
      // the loss D(x) for discriminator.
      const Dtype* bottom_data = bottom[0]->cpu_data(); // D(x)
      const Dtype* middle_data = middle_blob_.mutable_cpu_data(); // max(0, m-D(G(z)))
      Dtype* top_data = top[0]->mutable_cpu_data();
      caffe_set(inner_numb_, Dtype(0), top_data);
      for (int i = 0; i < outer_numb_; ++i) {
        caffe_axpy(inner_numb_, Dtype(1), middle_data + i * inner_numb_, top_data);
        caffe_axpy(inner_numb_, Dtype(1), bottom_data + i * inner_numb_, top_data);
      }
    }
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // the loss D(G(z)) for discriminator.
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(inner_numb_, Dtype(0), top_data);
    for (int i = 0; i < outer_numb_; ++i) {
      caffe_axpy(inner_numb_, Dtype(1), bottom_data + i * inner_numb_, top_data);
    }
  }
}

template <typename Dtype>
void EBGANLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  if (discriminator_interval_ &&
      solver_iter_ % discriminator_interval_ >= discriminator_postpone_ &&
      solver_iter_ % discriminator_interval_ <  discriminator_postpone_ + discriminator_duration_) {
    // descending stochastic gradient.
    const Dtype* middle_diff = middle_blob_.cpu_diff(); // sign(m-D(G(z))) with values {0,1}
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(outer_numb_ * inner_numb_, Dtype(1), bottom_diff); // 1.
    caffe_axpy(outer_numb_ * inner_numb_, Dtype(-1), middle_diff, bottom_diff);
  }
  if (generator_interval_ &&
      solver_iter_ % generator_interval_ >= generator_postpone_ &&
      solver_iter_ % generator_interval_ <  generator_postpone_ + generator_duration_) {
    // descending stochastic gradient.
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(outer_numb_ * inner_numb_, Dtype(1), bottom_diff); // 1.
  }
}

#ifdef CPU_ONLY
STUB_GPU(EBGANLossLayer);
#endif
INSTANTIATE_CLASS(EBGANLossLayer);
REGISTER_LAYER_CLASS(EBGANLoss);
} // namespace caffe