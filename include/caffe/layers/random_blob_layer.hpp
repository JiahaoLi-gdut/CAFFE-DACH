#ifndef CAFFE_RANDOM_BLOB_LAYER_HPP_
#define CAFFE_RANDOM_BLOB_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RandomBlobLayer : public Layer<Dtype> {
 public:
  explicit RandomBlobLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandomBlob"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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

  string type_;
  Dtype min_; // the min value in uniform filler for blob.
  Dtype max_; // the max value in uniform filler for blob.
  Dtype mean_; // the mean value in Gaussian filler for blob.
  Dtype sigma_; // the sigma value in Gaussian filler for blob.
  int num_; // effective when the bottom blob not exists.
  int channel_; // effective when the bottom blob not exists.
  int height_; // effective when the bottom blob not exists.
  int width_; // effective when the bottom blob not exists.
};
} // namespace caffe
#endif // CAFFE_RANDOM_BLOB_LAYER_HPP_