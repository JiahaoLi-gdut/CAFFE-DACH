#include <vector>
#include "caffe/layers/random_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomBlobLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  RandomBlobParameter random_blob_param = this->layer_param_.random_blob_param();
  type_  = random_blob_param.type();
  min_   = random_blob_param.min();
  max_   = random_blob_param.max();
  mean_  = random_blob_param.mean();
  sigma_ = random_blob_param.sigma();
  if (bottom.size()) {
    num_     = bottom[0]->shape(0);
    channel_ = bottom[0]->shape(1);
    height_  = bottom[0]->shape(2);
    width_   = bottom[0]->shape(3);
  }
  else {
    num_     = random_blob_param.num();
    channel_ = random_blob_param.channel();
    height_  = random_blob_param.height();
    width_   = random_blob_param.width();
  }
  CHECK(type_ == "uniform" || type_ == "gaussian")
    << "Random Blob implemented only for 'uniform', 'gaussian'";
}

template <typename Dtype>
void RandomBlobLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, channel_, height_, width_);
}

template <typename Dtype>
void RandomBlobLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (type_ == "uniform") {
    caffe_rng_uniform(top[0]->count(), min_, max_, top[0]->mutable_cpu_data());
  }
  else if (type_ == "gaussian") {
    caffe_rng_gaussian(top[0]->count(), mean_, sigma_, top[0]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RandomBlobLayer, Forward);
#endif
INSTANTIATE_CLASS(RandomBlobLayer);
REGISTER_LAYER_CLASS(RandomBlob);
} // namespace caffe