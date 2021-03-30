#ifndef CAFFE_AUC_LAYER_HPP_
#define CAFFE_AUC_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * This layer must be used for binary classification with Softmax layer.
 */
template <typename Dtype>
class AUCLayer : public Layer<Dtype> {
public:
  explicit AUCLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AUC"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  /// @copydoc AUCLayer
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

  int auc_axis_;
  int outer_num_;
  int inner_num_;
  int channel_;
};

} // namespace caffe

#endif // CAFFE_AUC_LAYER_HPP_
