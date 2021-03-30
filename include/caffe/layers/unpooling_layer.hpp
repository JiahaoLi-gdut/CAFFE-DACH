#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class UnpoolingLayer : public Layer<Dtype> {
public:
  explicit UnpoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Unpooling"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {
    return (this->layer_param_.unpooling_param().unpool() ==
            UnpoolingParameter_UnpoolMethod_MAX) ? 2 : 1;
  }

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

  int cutext_c_, cutext_h_, cutext_w_;
  int kernel_c_, kernel_h_, kernel_w_;
  int stride_c_, stride_h_, stride_w_;
  int pad_c_, pad_h_, pad_w_;
  bool ceil_c_, ceil_h_, ceil_w_;
};
} // namespace caffe
#endif // CAFFE_UNPOOLING_LAYER_HPP_