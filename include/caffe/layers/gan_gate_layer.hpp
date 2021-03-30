#ifndef CAFFE_GAN_GATE_LAYER_HPP_
#define CAFFE_GAN_GATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GANGateLayer : public Layer<Dtype> {
public:
  explicit GANGateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GANGate"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  void SolverIterChangedHandle(const void* message);

protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag) {}

  int generator_interval_;
  int generator_postpone_;
  int generator_duration_;
  int discriminator_interval_;
  int discriminator_postpone_;
  int discriminator_duration_;
  int solver_iter_;
};
} // namespace caffe
#endif // CAFFE_GAN_GATE_LAYER_HPP_