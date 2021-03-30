#include <vector>
#include "caffe/layers/gradient_scaler_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/messenger.hpp"

namespace caffe {

template <typename Dtype>
class AdaptationCoefficientHandler: public Listener {
 public:
  AdaptationCoefficientHandler(GradientScalerLayer<Dtype>* gradient_scaler_layer)
    : gradient_scaler_layer_(gradient_scaler_layer) {
  }
  void handle(const void* message) {
    gradient_scaler_layer_->AdaptationCoefficientHandle(message);
  }
private:
  GradientScalerLayer<Dtype>* gradient_scaler_layer_;
};

template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  lower_bound_ = this->layer_param_.gradient_scaler_param().lower_bound();
  upper_bound_ = this->layer_param_.gradient_scaler_param().upper_bound();
  alpha_ = this->layer_param_.gradient_scaler_param().alpha();
  max_iter_ = this->layer_param_.gradient_scaler_param().max_iter();
  coeff_ = Dtype(1); // Default adaptation coefficient.
  DCHECK(lower_bound_ <= upper_bound_);
  DCHECK(alpha_ >= 0.f);
  DCHECK(max_iter_ >= 1.f);
  SyncMessenger::AddListener(
    "Any", "Any", "Solver", "Any", "SOLVER_ITER_CHANGED", 1, 0, 1,
    new AdaptationCoefficientHandler<Dtype>(this)
  );
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::AdaptationCoefficientHandle(const void* message) {
  int iter = *(static_cast<const int*>(message));
  Dtype height = upper_bound_ - lower_bound_;
  Dtype progress = std::min(Dtype(1), Dtype(iter) / max_iter_);
  coeff_ = Dtype(2) * height / (Dtype(1) + exp(-alpha_ * progress)) - height + lower_bound_;
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[0]) {
    caffe_cpu_scale(bottom[0]->count(), -coeff_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif
INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);
} // namespace caffe