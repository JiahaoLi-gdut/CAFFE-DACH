#include <vector>
#include "caffe/layers/random_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomBlobLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (type_ == "uniform") {
    caffe_gpu_rng_uniform(top[0]->count(), min_, max_, top[0]->mutable_gpu_data());
  }
  else if (type_ == "gaussian") {
    caffe_gpu_rng_gaussian(top[0]->count(), mean_, sigma_, top[0]->mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(RandomBlobLayer);
} // namespace caffe