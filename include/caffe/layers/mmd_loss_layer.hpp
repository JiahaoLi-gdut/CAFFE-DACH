#ifndef CAFFE_MMD_LOSS_LAYER_HPP_
#define CAFFE_MMD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MMDLossLayer : public LossLayer<Dtype> {
public:
  explicit MMDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MMDLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

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

  // external parameter
  int   kernel_numb_;
  Dtype kernel_mult_;
  bool  absfix_flag_;
  Dtype loss_weight_;
  // internal parameter
  Dtype buffer_gamma_;
  Blob<int> random_blob_;
  Blob<Dtype> buffer_blob_;
};
} // namespace caffe
#endif // CAFFE_MMD_LAYER_HPP_