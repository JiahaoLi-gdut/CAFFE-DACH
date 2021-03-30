#ifndef CAFFE_CLASS_LATCH_LAYER_HPP_
#define CAFFE_CLASS_LATCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ClassLatchLayer : public Layer<Dtype> {
public:
  explicit ClassLatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param), solver_iter_(0), preforward_tag_(false) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClassLatch"; }
  virtual inline int MinBottomBlobs() const { return 2; }
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
      const bool prebackward_flag) {}
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag) {}

  int  solver_iter_;
  bool preforward_tag_;
  bool latch_mode_;
  bool latch_flag_;
  vector<int> label_axis_;
  vector<int> label_nmax_;
  vector<int> latch_nmin_;
  vector<int> latch_numb_;
  vector<shared_ptr<Blob<Dtype> > > storer_blob_;
};
} // namespace caffe
#endif // CAFFE_CLASS_LATCH_LAYER_HPP_