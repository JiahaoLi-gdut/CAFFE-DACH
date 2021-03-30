#ifndef CAFFE_MMDX_LOSS_LAYER_HPP_
#define CAFFE_MMDX_LOSS_LAYER_HPP_

#include <map>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MMDXLossLayer : public LossLayer<Dtype> {
 public:
  explicit MMDXLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MMDXLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }

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
  map<int, vector<int> > source_vmap_;
  map<int, vector<int> > target_vmap_;
  map<int, shared_ptr<Blob<int> > > source_bmap_;
  map<int, shared_ptr<Blob<int> > > target_bmap_;
  map<int, map<int, shared_ptr<Blob<int> > > > random_mmap_;
  map<int, map<int, shared_ptr<Blob<Dtype> > > > buffer_mmap_;
  map<int, map<int, Dtype> > buffer_gmap_;
};
} // namespace caffe
#endif // CAFFE_MMDX_LOSS_LAYER_HPP_