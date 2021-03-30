#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/param2top_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Param2TopLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Param2TopParameter& param = this->layer_param_.param2top_param();
  const int num_top = top.size();
  const int num_data_filler = param.data_filler_size();
  const int num_data_refill = param.data_refill_size();
  CHECK(num_data_filler == 0 || num_data_filler == 1 || num_data_filler == num_top)
      << "Number of data fillers must be 0, 1 or equal to the number of tops: "
      << num_top << "; you specified " << num_data_filler << " data fillers.";
  CHECK(num_data_refill == 0 || num_data_refill == 1 || num_data_refill == num_top)
      << "Number of data refills must be 0, 1 or equal to the number of tops: "
      << num_top << "; you specified " << num_data_refill << " data refills.";
  const bool legacy_dims = param.num_size() || param.channels_size() ||
                           param.height_size() || param.width_size();
  if (legacy_dims) {
    CHECK_EQ(0, param.shape_size())
        << "Both shape and legacy fields were specified";
    // Using deprecated 4D output dim specifiers.
    CHECK(param.num_size() == 1 || param.num_size() == num_top)
        << "Must specify 'num' once, or once per top blob "
        << "(" << num_top << "); specified " << param.num_size() << ".";
    CHECK(param.channels_size() == 1 || param.channels_size() == num_top)
        << "Must specify 'channels' once, or once per top blob "
        << "(" << num_top << "); specified " << param.channels_size() << ".";
    CHECK(param.height_size() == 1 || param.height_size() == num_top)
        << "Must specify 'height' once, or once per top blob "
        << "(" << num_top << "); specified " << param.height_size() << ".";
    CHECK(param.width_size() == 1 || param.width_size() == num_top)
        << "Must specify 'width' once, or once per top blob "
        << "(" << num_top << "); specified " << param.width_size() << ".";
  } else {
    CHECK(param.shape_size() == 1 || param.shape_size() == num_top)
        << "Must specify 'shape' once, or once per top blob "
        << "(" << num_top << "); specified " << param.shape_size() << ".";
  }
  // fillers_[i] tells Forward i how to fill the data.
  fillers_.clear();
  if (num_data_filler <= 1) {
    FillerParameter filler_param;
    if (num_data_filler == 0) {
      filler_param.set_type("constant");
      filler_param.set_value(0);
    } else {
      filler_param.CopyFrom(param.data_filler(0));
    }
    fillers_.resize(1);
    fillers_[0].reset(GetFiller<Dtype>(filler_param));
  } else {
    fillers_.resize(num_top);
    for (int i = 0; i < num_top; ++i) {
      fillers_[i].reset(GetFiller<Dtype>(param.data_filler(i)));
    }
  }
  // refills_[i] tells Forward i whether or not to actually refill top Blob i.
  // If refills_[i] is false, Forward does nothing for Blob i.
  refills_.clear();
  if (num_data_refill <= 1) {
    refills_.resize(1);
    if (num_data_filler == 0) {
      refills_[0] = 0;
    } else {
      refills_[0] = param.data_refill(0);
    }
  } else {
    refills_.resize(num_top);
    for (int i = 0; i < num_top; ++i) {
      refills_[i] = param.data_refill(i);
    }
  }
  // reshape param blobs and top blobs
  this->blobs_.resize(num_top);
  for (int i = 0; i < num_top; ++i) {
    if (legacy_dims) {
      const int num = (param.num_size() == 1) ? param.num(0) : param.num(i);
      const int channels = (param.channels_size() == 1) ? param.channels(0) : param.channels(i);
      const int height = (param.height_size() == 1) ? param.height(0) : param.height(i);
      const int width = (param.width_size() == 1) ? param.width(0) : param.width(i);
      top[i]->Reshape(num, channels, height, width);
    } else {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
    this->blobs_[i].reset(new Blob<Dtype>());
    this->blobs_[i]->ReshapeLike(*top[i]);
  }
  // Run Forward once
  this->Forward(bottom, top, false);
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void Param2TopLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < top.size(); ++i) {
    const int filler_id = (fillers_.size() > 1) ? i : 0;
    const int refill_id = (refills_.size() > 1) ? i : 0;
    if (refills_[refill_id]) {
      fillers_[filler_id]->Fill(this->blobs_[i].get());
      if (refills_[refill_id] == 1) {
        refills_[refill_id] = 0;
      }
    }
    const int count = top[i]->count();
    const Dtype* params_data = this->blobs_[i]->cpu_data();
    Dtype* topper_data = top[i]->mutable_cpu_data();
    caffe_copy(count, params_data, topper_data);
  }
}

template <typename Dtype>
void Param2TopLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  for (int i = 0; i < top.size(); ++i) {
    if (this->param_propagate_down_[i]) {
      const int count = top[i]->count();
      const Dtype* topper_diff = top[i]->cpu_diff();
      Dtype* params_diff = this->blobs_[i]->mutable_cpu_diff();
      caffe_copy(count, topper_diff, params_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Param2TopLayer);
#endif
INSTANTIATE_CLASS(Param2TopLayer);
REGISTER_LAYER_CLASS(Param2Top);
} // namespace caffe