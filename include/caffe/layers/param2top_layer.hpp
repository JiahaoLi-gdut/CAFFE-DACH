#ifndef CAFFE_PARAM2TOP_LAYER_HPP_
#define CAFFE_PARAM2TOP_LAYER_HPP_

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Param2TopLayer : public Layer<Dtype> {
public:
  explicit Param2TopLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Param2Top"; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

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

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<unsigned> refills_;
};
}  // namespace caffe
#endif  // CAFFE_PARAM2TOP_LAYER_HPP_