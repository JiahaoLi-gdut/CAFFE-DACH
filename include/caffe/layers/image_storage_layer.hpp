#ifndef CAFFE_IMAGE_STORAGE_LAYER_HPP_
#define CAFFE_IMAGE_STORAGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
template <typename Dtype>
class ImageStorageLayer : public Layer<Dtype> {
public:
  explicit ImageStorageLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "ImageStorage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag);

private:
  string encoder_;
  string saveroot_;
};
} // namespace caffe
#endif // CAFFE_IMAGE_STORAGE_LAYER_HPP_