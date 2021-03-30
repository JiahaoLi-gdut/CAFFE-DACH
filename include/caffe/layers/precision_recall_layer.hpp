#ifndef CAFFE_PRECISION_RECALL_LAYER_HPP_
#define CAFFE_PRECISION_RECALL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PrecisionRecallLayer : public Layer<Dtype> {
public:
  explicit PrecisionRecallLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PrecisionRecall"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
  /// @copydoc PrecisionRecallLayer
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

  int axis_;
  int outer_num_;
  int inner_num_;
  int channel_;
  Blob<Dtype> midd_;
};
} // namespace caffe
#endif // CAFFE_PRECISION_RECALL_LAYER_HPP_