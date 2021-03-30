#include "caffe/layers/msb_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// n_s: source sample size
// d_s: source sample dimension
// n_t: target sample size
// d_t: target sample dimension
// x_s: source sample
// x_t: target sample
// y_s: source bias
// y_t: target bias
// function sums Mean Square Bias (merge y_s and y_t)
template <typename Dtype>
void mean_square_bias_call_cpu(
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t) {
  const int d_x = d_s < d_t ? d_s : d_t;
  const int m_s = n_s * d_x;
  const int m_t = n_t * d_x;
  const int m_x = m_s + m_t;
  for (int x_i = 0; x_i < m_x; ++x_i) {
    const int x_w = x_i % d_x;
    const int x_h = x_i / d_x;
    const int d_z = x_h < n_s ? d_t : d_s;
    const int n_z = x_h < n_s ? n_t : n_s;
    const Dtype* const z = x_h < n_s
        ? (x_t + x_w) : (x_s + x_w);
    const Dtype* const x = x_h < n_s
        ? (x_s + x_w + d_s * x_h)
        : (x_t + x_w + d_t * (x_h - n_s));
    Dtype* const y = x_i < m_s
        ? (y_s + x_i) : (y_t + x_i - m_s);
    *y = 0;
    for (int z_i = 0; z_i < n_z; ++z_i) {
      *y += *x - *(z + d_z * z_i);
    }
    *y /= n_z;
    *y *= *y;
  }
}

template <typename Dtype>
void mean_square_bias_main_cpu(
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t,
    Dtype* const msr, Dtype* const msb) {
  Dtype b_s, b_t, b_x;
  const int d_x = d_s < d_t ? d_s : d_t;
  const int m_s = n_s * d_x;
  const int m_t = n_t * d_x;
  mean_square_bias_call_cpu(
    n_s, d_s, n_t, d_t,
    x_s, y_s, x_t, y_t
  );
  b_s = caffe_cpu_lossness_sum(m_s, y_s);
  b_t = caffe_cpu_lossness_sum(m_t, y_t);
  b_x = b_s + b_t;
  b_s /= b_x;
  b_t /= b_x;
  b_s *= (n_s + n_t) / (2 * n_s);
  b_t *= (n_s + n_t) / (2 * n_t);
  *msr = (n_s + n_t) / b_x;
  *msb = b_s + b_t;
}

///////////////////////////////////////////////////////////////////

// n_s: source sample size
// d_s: source sample dimension
// n_t: target sample size
// d_t: target sample dimension
// x_s: source sample
// x_t: target sample
// y_s: source bias
// y_t: target bias
// function sums Mean Deltas Diff
template <typename Dtype>
void mean_deltas_diff_call_cpu(
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype r_b, const Dtype l_w,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t) {
  const int d_x = d_s < d_t ? d_s : d_t;
  const int n_x = n_s + n_t;
  const int m_x = n_x * d_x;
  for (int x_i = 0; x_i < m_x; ++x_i) {
    const int x_w = x_i % d_x;
    const int x_h = x_i / d_x;
    const Dtype* const x = x_h < n_s
        ? (x_s + x_w + d_s * x_h)
        : (x_t + x_w + d_t * (x_h - n_s));
    Dtype* const y = x_h < n_s
        ? (y_s + x_w + d_s * x_h)
        : (y_t + x_w + d_t * (x_h - n_s));
    for (int x_j = 0; x_j < n_x; ++x_j) {
      const Dtype r_z = x_h < n_s
          ? (x_j < n_s ? n_s : -n_t / 2)
          : (x_j < n_s ? -n_s / 2 : n_t);
      const Dtype* const z = x_j < n_s
          ? (x_s + x_w + d_s * x_j)
          : (x_t + x_w + d_t * (x_j - n_s));
      *y = *x + *z / r_z;
    }
    *y /= x_h < n_s ? n_s : n_t;
    *y *= r_b * l_w;
  }
}

template <typename Dtype>
void mean_deltas_diff_main_cpu(
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype r_b, const Dtype l_w,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t) {
  caffe_set(d_s * n_s, Dtype(0), y_s);
  caffe_set(d_t * n_t, Dtype(0), y_t);
  mean_deltas_diff_call_cpu(
    n_s, d_s, n_t, d_t, r_b,
    l_w, x_s, y_s, x_t, y_t
  );
}

///////////////////////////////////////////////////////////////////

template <typename Dtype>
void MSBLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //MSBLossParameter msb_loss_param = this->layer_param_.msb_loss_param();
  loss_weight_ = this->layer_param_.loss_weight(0);
}

template <typename Dtype>
void MSBLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(vector<int>(0));
}

template <typename Dtype>
void MSBLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const int source_count = bottom[0]->count();
  const int target_count = bottom[1]->count();
  const int source_dnumb = bottom[0]->shape(0);
  const int target_dnumb = bottom[1]->shape(0);
  const int source_ddims = source_count / source_dnumb;
  const int target_ddims = target_count / target_dnumb;
  const Dtype* source_datum = bottom[0]->cpu_data();
  const Dtype* target_datum = bottom[1]->cpu_data();
  Dtype* source_diffs = bottom[0]->mutable_cpu_diff(); // used for buffer
  Dtype* target_diffs = bottom[1]->mutable_cpu_diff(); // used for buffer
  Dtype* topper_datum = top[0]->mutable_cpu_data();
  mean_square_bias_main_cpu(
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    source_datum, source_diffs,
    target_datum, target_diffs,
    &buffer_ratio_, topper_datum
  );
}

template <typename Dtype>
void MSBLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  const int source_count = bottom[0]->count();
  const int target_count = bottom[1]->count();
  const int source_dnumb = bottom[0]->shape(0);
  const int target_dnumb = bottom[1]->shape(0);
  const int source_ddims = source_count / source_dnumb;
  const int target_ddims = target_count / target_dnumb;
  const Dtype* source_datum = bottom[0]->cpu_data();
  const Dtype* target_datum = bottom[1]->cpu_data();
  Dtype* source_diffs = bottom[0]->mutable_cpu_diff();
  Dtype* target_diffs = bottom[1]->mutable_cpu_diff();
  mean_deltas_diff_main_cpu(
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    buffer_ratio_, loss_weight_,
    source_datum, source_diffs,
    target_datum, target_diffs
  );
}

#ifdef CPU_ONLY
STUB_GPU(MSBLossLayer);
#endif
INSTANTIATE_CLASS(MSBLossLayer);
REGISTER_LAYER_CLASS(MSBLoss);
} // namespace caffe