#ifndef CAFFE_EBGAN_LOSS_LAYER_HPP_
#define CAFFE_EBGAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class EBGANLossLayer : public LossLayer<Dtype> {
public:
  explicit EBGANLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  virtual inline const char* type() const { return "EBGANLoss"; }
  void SolverIterChangedHandle(const void* message);

protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag);
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag);

  int bottom_blob_axis_;
  int generator_interval_;
  int generator_postpone_;
  int generator_duration_;
  int discriminator_interval_;
  int discriminator_postpone_;
  int discriminator_duration_;
  int positive_margin_;
  int outer_numb_;
  int inner_numb_;
  int solver_iter_;
  Blob<Dtype> middle_blob_;
};
} // namespace caffe
#endif // CAFFE_EBGAN_LOSS_LAYER_HPP_