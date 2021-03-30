#ifndef CAFFE_BOTTOM2PARAM_LAYER_HPP_
#define CAFFE_BOTTOM2PARAM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Bottom2ParamLayer : public Layer<Dtype> {
public:
  explicit Bottom2ParamLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Bottom2Param"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

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
};
}  // namespace caffe
#endif  // CAFFE_BOTTOM2PARAM_LAYER_HPP_