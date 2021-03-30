#include "caffe/layers/mmd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// n_y: number of y
// d_y: dimension of y
// n_z: number of z
// d_z: dimension of z
// n_s: source sample size
// d_s: source sample dimension
// n_t: target sample size
// d_t: target sample dimension
// x_s: source sample
// x_t: target sample
// y: sum output storage
// z: sum output storage
// function sums Squared Spread Distance (merge y and z)
template <typename Dtype>
void pairwise_ssd_summation_call_cpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype* const x_s, Dtype* const y,
    const Dtype* const x_t, Dtype* const z) {
  const int d_x = d_s < d_t ? d_s : d_t;
  const int n_x = n_s + n_t;
  const int m_x = n_x * n_x * d_x;
  const int m_y = n_y * d_y;
  const int m_z = n_z * d_z;
  const int m_o = m_y + m_z;
  const int q_m = m_x / m_o;
  const int r_m = m_x % m_o;
  const int m_t = m_x < m_o ? m_x : m_o;
  for (int t_i = 0; t_i < m_t; ++t_i) {
    Dtype* const o = t_i < m_y ? (y + t_i) : (z + t_i - m_y);
    *o = 0;
    const int m_f = q_m + (t_i < r_m);
    for (int f_i = 0; f_i < m_f; ++f_i) {
      const int x_i = t_i + f_i * m_o;
      const int x_w = x_i % d_x;
      const int x_h = x_i / d_x % n_x;
      const int x_c = x_i / d_x / n_x;
      const Dtype x_0 = x_h < n_s
          ? x_s[x_w + d_s * x_h]
          : x_t[x_w + d_t * (x_h - n_s)];
      const Dtype x_1 = x_c < n_s
          ? x_s[x_w + d_s * x_c]
          : x_t[x_w + d_t * (x_c - n_s)];
      *o += (x_0 - x_1) * (x_0 - x_1);
    }
  }
}

template <typename Dtype>
Dtype pairwise_ssd_summation_main_cpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const Dtype* const x_s, Dtype* const y,
    const Dtype* const x_t, Dtype* const z) {
  Dtype tmp, sum = 0;
  const int d_x = d_s < d_t ? d_s : d_t;
  const int n_x = n_s + n_t;
  const int m_x = n_x * n_x * d_x;
  const int m_y = n_y * d_y;
  const int m_z = n_z * d_z;
  const int m_o = m_y + m_z;
  const int m_t = m_x < m_o ? m_x : m_o;
  const int u_y = m_y < m_t ? m_y : m_t;
  const int u_z = (m_y < m_t) * (m_t - m_y);
  pairwise_ssd_summation_call_cpu(
    n_y, d_y, n_z, d_z,
    n_s, d_s, n_t, d_t,
    x_s, y, x_t, z
  );
  if (u_y > 0) {
    tmp = caffe_cpu_lossness_sum(u_y, y);
    sum += tmp;
  }
  if (u_z > 0) {
    tmp = caffe_cpu_lossness_sum(u_z, z);
    sum += tmp;
  }
  return sum;
}

///////////////////////////////////////////////////////////////////

// n_y: number of y
// d_y: dimension of y
// n_z: number of z
// d_z: dimension of z
// n_s: source sample size
// n_t: target sample size
// d_s: source sample dimension
// d_t: target sample dimension
// n_r: number of random group
// d_r: random integer number per group
// n_k: number of kernel
// g_b: based gamma
// g_m: multiply gamma
// x_s: source sample
// x_t: target sample
// r_p: ramdom position
// y: sum output storage
// z: sum output storage
// function sums Maximum Mean Discrepancy.
template <typename Dtype>
void pairwise_mmd_summation_call_cpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int n_r, const int d_r,
    const int n_k, const Dtype g_b,
    const Dtype g_m, const int* const r_p,
    const Dtype* const x_s, Dtype* const y,
    const Dtype* const x_t, Dtype* const z) {
  const int m_x = n_r * d_r * n_k;
  const int m_y = n_y * d_y;
  const int m_z = n_z * d_z;
  const int m_o = m_y + m_z;
  const int q_m = m_x / m_o;
  const int r_m = m_x % m_o;
  const int m_t = m_x < m_o ? m_x : m_o;
  for (int t_i = 0; t_i < m_t; ++t_i) {
    Dtype* const o = t_i < m_y ? (y + t_i) : (z + t_i - m_y);
    *o = 0;
    const int m_f = q_m + (t_i < r_m);
    for (int f_i = 0; f_i < m_f; ++f_i) {
      const int x_i = t_i + f_i * m_o;
      const int x_w = x_i % n_k;
      const int x_h = x_i / n_k % d_r;
      const int x_c = x_i / n_k / d_r;
      const int x_u = (x_h + 1) % d_r;
      const int d_0 = x_h / 2 % 2 ? d_t : d_s;
      const int d_1 = x_u / 2 % 2 ? d_t : d_s;
      const Dtype* const x_0 = x_h / 2 % 2
          ? (x_t + d_t * (r_p[x_c * d_r + x_h] % n_t))
          : (x_s + d_s * (r_p[x_c * d_r + x_h] % n_s));
      const Dtype* const x_1 = x_u / 2 % 2
          ? (x_t + d_t * (r_p[x_c * d_r + x_u] % n_t))
          : (x_s + d_s * (r_p[x_c * d_r + x_u] % n_s));
      const Dtype* const x_2 = x_0 != x_1 ? x_1 : (x_u / 2 % 2
          ? (x_t + d_t * ((r_p[x_c * d_r + x_u] + 1) % n_t))
          : (x_s + d_s * ((r_p[x_c * d_r + x_u] + 1) % n_s)));
      Dtype sum = 0;
      for (int x_d = 0; x_d < d_0 && x_d < d_1; ++x_d) {
        const Dtype dif = x_0[x_d] - x_2[x_d];
        sum += dif * dif;
      }
      *o += exp(-g_b * pow(g_m, x_w) * sum) * (1 - x_h % 2 * 2);
    }
  }
}

template <typename Dtype>
Dtype pairwise_mmd_summation_main_cpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int n_r, const int d_r,
    const int n_k, const Dtype g_b,
    const Dtype g_m, const int* const r_p,
    const Dtype* const x_s, Dtype* const y,
    const Dtype* const x_t, Dtype* const z) {
  Dtype tmp, sum = 0;
  const int m_x = n_r * d_r * n_k;
  const int m_y = n_y * d_y;
  const int m_z = n_z * d_z;
  const int m_o = m_y + m_z;
  const int m_t = m_x < m_o ? m_x : m_o;
  const int u_y = m_y < m_t ? m_y : m_t;
  const int u_z = (m_y < m_t) * (m_t - m_y);
  pairwise_mmd_summation_call_cpu(
    n_y, d_y, n_z, d_z,
    n_s, d_s, n_t, d_t,
    n_r, d_r, n_k, g_b,
    g_m, r_p, x_s, y, x_t, z
  );
  if (u_y > 0) {
    tmp = caffe_cpu_lossness_sum(u_y, y);
    sum += tmp;
  }
  if (u_z > 0) {
    tmp = caffe_cpu_lossness_sum(u_z, z);
    sum += tmp;
  }
  return sum;
}

///////////////////////////////////////////////////////////////////

template <typename Dtype>
void pairwise_sub_cpu(
    const int d_s, const int d_t,
    const Dtype* const x_s, const Dtype* const x_t,
    Dtype* const y) {
  const int d_x = d_s < d_t ? d_s : d_t;
  for (int x_d = 0; x_d < d_x; ++x_d) {
    y[x_d] = x_s[x_d] - x_t[x_d];
  }
}

template <typename Dtype>
void pairwise_mmd_derivative_cpu(
    const int d_s, const int d_t,
    const int n_k, const Dtype g_b,
    const Dtype g_m, const Dtype l_w,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t,
          Dtype* const buf) {
  Dtype mul = 0;
  const int d_x = d_s < d_t ? d_s : d_t;
  pairwise_sub_cpu(d_s, d_t, x_s, x_t, buf);
  Dtype dot = caffe_cpu_dot(d_x, buf, buf);
  Dtype k_g = g_b / pow(g_m, (Dtype)(n_k / 2));
  for(int k_i = 0; k_i < n_k; ++k_i) {
    mul -= 2 * k_g * exp(-k_g * dot);
    k_g *= g_m;
  }
  caffe_scal(d_x, mul / l_w, buf);
  caffe_add(d_x, y_s, buf, y_s);
  caffe_sub(d_x, y_t, buf, y_t);
}

///////////////////////////////////////////////////////////////////

template <typename Dtype>
void MMDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  MMDLossParameter mmd_loss_param = this->layer_param_.mmd_loss_param();
  kernel_numb_ = mmd_loss_param.kernel_numb();
  kernel_mult_ = mmd_loss_param.kernel_mult();
  absfix_flag_ = mmd_loss_param.absfix_flag();
  loss_weight_ = this->layer_param_.loss_weight(0);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int source_count = bottom[0]->count();
  const int target_count = bottom[1]->count();
  const int source_dnumb = bottom[0]->shape(0);
  const int target_dnumb = bottom[1]->shape(0);
  const int source_ddims = source_count / source_dnumb;
  const int target_ddims = target_count / target_dnumb;
  const int sample_dnumb = source_dnumb < target_dnumb ? target_dnumb : source_dnumb;
  const int sample_ddims = source_ddims < target_ddims ? target_ddims : source_ddims;
  vector<int> random_shape(2, 4);
  vector<int> buffer_shape(1, sample_ddims);
  random_shape[0] = sample_dnumb;
  random_blob_.Reshape(random_shape);
  buffer_blob_.Reshape(buffer_shape);
  int* random_datum = random_blob_.mutable_cpu_data();
  const int random_count = random_blob_.count();
  for (int random_index = 0; random_index < random_count; ++random_index) {
    *random_datum = rand();
    ++random_datum;
  }
  // reshape top blob.
  top[0]->Reshape(vector<int>(0));
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const int source_count = bottom[0]->count();
  const int target_count = bottom[1]->count();
  const int source_dnumb = bottom[0]->shape(0);
  const int target_dnumb = bottom[1]->shape(0);
  const int domain_dnumb = source_dnumb + target_dnumb;
  const int source_ddims = source_count / source_dnumb;
  const int target_ddims = target_count / target_dnumb;
  const int sample_dnumb = source_dnumb < target_dnumb ? target_dnumb : source_dnumb;
  const int sample_ddims = source_ddims < target_ddims ? source_ddims : target_ddims;
  const Dtype* source_datum = bottom[0]->cpu_data();
  const Dtype* target_datum = bottom[1]->cpu_data();
  const int* random_datum = random_blob_.cpu_data();
  Dtype* source_diffs = bottom[0]->mutable_cpu_diff(); // used for buffer
  Dtype* target_diffs = bottom[1]->mutable_cpu_diff(); // used for buffer
  Dtype* topper_datum = top[0]->mutable_cpu_data();
  //calculate bandwidth
  CHECK(domain_dnumb * domain_dnumb * sample_ddims >= 0)
    << "Integer value overflow!";
  const Dtype buffer_absum = pairwise_ssd_summation_main_cpu(
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    source_datum, source_diffs,
    target_datum, target_diffs
  );
  buffer_gamma_ = domain_dnumb * (domain_dnumb - 1) / buffer_absum;
  buffer_gamma_ *= !absfix_flag_ || 0 < buffer_gamma_;
  //calculate each kernel of data and loss
  const Dtype kernel_gamma = buffer_gamma_ / pow(kernel_mult_, (Dtype)(kernel_numb_ / 2));
  *topper_datum = pairwise_mmd_summation_main_cpu(
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    source_dnumb, source_ddims,
    target_dnumb, target_ddims,
    sample_dnumb, 4,
    kernel_numb_, kernel_gamma,
    kernel_mult_, random_datum,
    source_datum, source_diffs,
    target_datum, target_diffs
  );
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_cpu(
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
  const int* random_datum = random_blob_.cpu_data();
  const Dtype* source_datum = bottom[0]->cpu_data();
  const Dtype* target_datum = bottom[1]->cpu_data();
  Dtype* source_diffs = bottom[0]->mutable_cpu_diff();
  Dtype* target_diffs = bottom[1]->mutable_cpu_diff();
  Dtype* buffer_datum = buffer_blob_.mutable_cpu_data();
  caffe_set(source_count, Dtype(0), source_diffs);
  caffe_set(target_count, Dtype(0), target_diffs);
  const int sample_dnumb = source_dnumb < target_dnumb ? target_dnumb : source_dnumb;
  for(int sample_index = 0; sample_index < sample_dnumb; ++sample_index, random_datum += 4) {
    const int source_indx0 = random_datum[0] % source_dnumb;
    const int source_indx1 = random_datum[1] % source_dnumb;
    const int source_indx2 = (source_indx1 + (source_indx0 == source_indx1)) % source_dnumb;
    const int target_indx0 = random_datum[2] % target_dnumb;
    const int target_indx1 = random_datum[3] % target_dnumb;
    const int target_indx2 = (target_indx1 + (target_indx0 == target_indx1)) % target_dnumb;
    const Dtype* source_data1 = source_datum + source_indx0 * source_ddims;
    const Dtype* source_data2 = source_datum + source_indx2 * source_ddims;
    const Dtype* target_data1 = target_datum + target_indx0 * target_ddims;
    const Dtype* target_data2 = target_datum + target_indx2 * target_ddims;
    Dtype* source_diff1 = source_diffs + source_indx0 * source_ddims;
    Dtype* source_diff2 = source_diffs + source_indx2 * source_ddims;
    Dtype* target_diff1 = target_diffs + target_indx0 * target_ddims;
    Dtype* target_diff2 = target_diffs + target_indx2 * target_ddims;
    pairwise_mmd_derivative_cpu(
      source_ddims, source_ddims,
      kernel_numb_, buffer_gamma_,
      kernel_mult_, sample_dnumb / loss_weight_,
      source_data1, source_diff1,
      source_data2, source_diff2,
      buffer_datum
    );
    pairwise_mmd_derivative_cpu(
      source_ddims, target_ddims,
      kernel_numb_, buffer_gamma_,
      kernel_mult_, -sample_dnumb / loss_weight_,
      source_data2, source_diff2,
      target_data1, target_diff1,
      buffer_datum
    );
    pairwise_mmd_derivative_cpu(
      target_ddims, target_ddims,
      kernel_numb_, buffer_gamma_,
      kernel_mult_, sample_dnumb / loss_weight_,
      target_data1, target_diff1,
      target_data2, target_diff2,
      buffer_datum
    );
    pairwise_mmd_derivative_cpu(
      target_ddims, source_ddims,
      kernel_numb_, buffer_gamma_,
      kernel_mult_, -sample_dnumb / loss_weight_,
      target_data2, target_diff2,
      source_data1, source_diff1,
      buffer_datum
    );
  }
}

#ifdef CPU_ONLY
STUB_GPU(MMDLossLayer);
#endif
INSTANTIATE_CLASS(MMDLossLayer);
REGISTER_LAYER_CLASS(MMDLoss);
} // namespace caffe