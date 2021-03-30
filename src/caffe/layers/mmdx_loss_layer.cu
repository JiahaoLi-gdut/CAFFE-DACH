#include "caffe/layers/mmdx_loss_layer.hpp"
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
// p_s: position array to x_s
// p_t: position array to x_t
// x_s: source sample
// x_t: target sample
// y: sum output storage
// z: sum output storage
// function sums Squared Spread Distance
template <typename Dtype>
__global__ void pairwise_ssdx_summation_call_gpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int* const p_s, const int* const p_t,
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
  CUDA_KERNEL_LOOP(t_i, m_t) {
    Dtype* const o = t_i < m_y ? (y + t_i) : (z + t_i - m_y);
    *o = 0;
    const int m_f = q_m + (t_i < r_m);
    for (int f_i = 0; f_i < m_f; ++f_i) {
      const int x_i = t_i + f_i * m_o;
      const int x_w = x_i % d_x;
      const int x_h = x_i / d_x % n_x;
      const int x_c = x_i / d_x / n_x;
      const Dtype x_0 = x_h < n_s
          ? x_s[x_w + d_s * p_s[x_h]]
          : x_t[x_w + d_t * p_t[x_h - n_s]];
      const Dtype x_1 = x_c < n_s
          ? x_s[x_w + d_s * p_s[x_c]]
          : x_t[x_w + d_t * p_t[x_c - n_s]];
      *o += (x_0 - x_1) * (x_0 - x_1);
    }
  }
}

template <typename Dtype>
Dtype pairwise_ssdx_summation_main_gpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int* const p_s, const int* const p_t,
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
  pairwise_ssdx_summation_call_gpu<Dtype><<<CAFFE_GET_BLOCKS(m_t), CAFFE_CUDA_NUM_THREADS>>>(
    n_y, d_y, n_z, d_z,
    n_s, d_s, n_t, d_t,
    p_s, p_t, x_s, y, x_t, z
  );
  if (u_y > 0) {
    caffe_gpu_lossness_sum(u_y, y, &tmp);
    sum += tmp;
  }
  if (u_z > 0) {
    caffe_gpu_lossness_sum(u_z, z, &tmp);
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
// p_s: position array to x_s
// p_t: position array to x_t
// x_s: source sample
// x_t: target sample
// r_p: ramdom position
// y: sum output storage
// z: sum output storage
// function sums Maximum Mean Discrepancy.
template <typename Dtype>
__global__ void pairwise_mmdx_summation_call_gpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int n_r, const int d_r, 
    const int n_k, const Dtype g_b,
    const Dtype g_m, const int* const r_p,
    const int* const p_s, const int* const p_t,
    const Dtype* const x_s, Dtype* const y,
    const Dtype* const x_t, Dtype* const z) {
  const int m_x = n_r * d_r * n_k;
  const int m_y = n_y * d_y;
  const int m_z = n_z * d_z;
  const int m_o = m_y + m_z;
  const int q_m = m_x / m_o;
  const int r_m = m_x % m_o;
  const int m_t = m_x < m_o ? m_x : m_o;
  CUDA_KERNEL_LOOP(t_i, m_t) {
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
          ? (x_t + d_t * p_t[r_p[x_c * d_r + x_h] % n_t])
          : (x_s + d_s * p_s[r_p[x_c * d_r + x_h] % n_s]);
      const Dtype* const x_1 = x_u / 2 % 2
          ? (x_t + d_t * p_t[r_p[x_c * d_r + x_u] % n_t])
          : (x_s + d_s * p_s[r_p[x_c * d_r + x_u] % n_s]);
      const Dtype* const x_2 = x_0 != x_1 ? x_1 : (x_u / 2 % 2
          ? (x_t + d_t * p_t[(r_p[x_c * d_r + x_u] + 1) % n_t])
          : (x_s + d_s * p_s[(r_p[x_c * d_r + x_u] + 1) % n_s]));
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
Dtype pairwise_mmdx_summation_main_gpu(
    const int n_y, const int d_y,
    const int n_z, const int d_z,
    const int n_s, const int d_s,
    const int n_t, const int d_t,
    const int n_r, const int d_r, 
    const int n_k, const Dtype g_b,
    const Dtype g_m, const int* const r_p,
    const int* const p_s, const int* const p_t,
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
  pairwise_mmdx_summation_call_gpu<Dtype><<<CAFFE_GET_BLOCKS(m_t), CAFFE_CUDA_NUM_THREADS>>>(
    n_y, d_y, n_z, d_z,
    n_s, d_s, n_t, d_t,
    n_r, d_r, n_k, g_b,
    g_m, r_p, p_s, p_t,
    x_s, y, x_t, z
  );
  if (u_y > 0) {
    caffe_gpu_lossness_sum(u_y, y, &tmp);
    sum += tmp;
  }
  if (u_z > 0) {
    caffe_gpu_lossness_sum(u_z, z, &tmp);
    sum += tmp;
  }
  return sum;
}

///////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void pairwise_sub_gpu(
    const int d_s, const int d_t,
    const Dtype* const x_s, const Dtype* const x_t,
    Dtype* const y) {
  const int d_x = d_s < d_t ? d_s : d_t;
  CUDA_KERNEL_LOOP(x_d, d_x) {
    y[x_d] = x_s[x_d] - x_t[x_d];
  }
}

template <typename Dtype>
void pairwise_mmdx_derivative_gpu(
    const int d_s, const int d_t,
    const int n_k, const Dtype g_b,
    const Dtype g_m, const Dtype l_w,
    const Dtype* const x_s, Dtype* const y_s,
    const Dtype* const x_t, Dtype* const y_t,
          Dtype* const buf) {
  Dtype mul = 0;
  const int d_x = d_s < d_t ? d_s : d_t;
  pairwise_sub_gpu<Dtype><<<CAFFE_GET_BLOCKS(d_x), CAFFE_CUDA_NUM_THREADS>>>(
    d_s, d_t, x_s, x_t, buf
  );
  Dtype dot; caffe_gpu_dot(d_x, buf, buf, &dot);
  Dtype k_g = g_b / pow(g_m, (Dtype)(n_k / 2));
  for(int k_i = 0; k_i < n_k; ++k_i) {
    mul -= 2 * k_g * exp(-k_g * dot);
    k_g *= g_m;
  }
  caffe_gpu_scal(d_x, mul / l_w, buf);
  caffe_gpu_add(d_x, y_s, buf, y_s);
  caffe_gpu_sub(d_x, y_t, buf, y_t);
}

template <typename Dtype>
void MMDXLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const int source_count = bottom[0]->count();
  const int target_count = bottom[1]->count();
  const int source_dnumb = bottom[0]->shape(0);
  const int target_dnumb = bottom[1]->shape(0);
  const int source_ddims = source_count / source_dnumb;
  const int target_ddims = target_count / target_dnumb;
  const int sample_ddims = source_ddims < target_ddims ? source_ddims : target_ddims;
  const Dtype* source_datum = bottom[0]->gpu_data();
  const Dtype* target_datum = bottom[1]->gpu_data();
  Dtype* source_diffs = bottom[0]->mutable_gpu_diff();
  Dtype* target_diffs = bottom[1]->mutable_gpu_diff();
  Dtype* topper_datum = top[0]->mutable_cpu_data();
  typename map<int, map<int, shared_ptr<Blob<int> > > >::iterator random_miter = random_mmap_.begin();
  for (; random_miter != random_mmap_.end(); ++random_miter) {
    const int source_label = random_miter->first;
    map<int, shared_ptr<Blob<int> > >& random_nbmap = random_miter->second;
    shared_ptr<Blob<int> > source_pblob = source_bmap_[source_label];
    const int  source_pnumb = source_pblob->count();
    const int* source_pdata = source_pblob->gpu_data();
    typename map<int, shared_ptr<Blob<int> > >::iterator random_biter = random_nbmap.begin();
    for (; random_biter != random_nbmap.end(); ++random_biter) {
      const int target_label = random_biter->first;
      shared_ptr<Blob<int> > random_nblob = random_biter->second;
      shared_ptr<Blob<int> > target_pblob = target_bmap_[target_label];
      const int  target_pnumb = target_pblob->count();
      const int* target_pdata = target_pblob->gpu_data();
      const int* random_datum = random_nblob->gpu_data();
      const int domain_pnumb = source_pnumb + target_pnumb;
      const int sample_pnumb = source_pnumb < target_pnumb ? target_pnumb : source_pnumb;
      CHECK(domain_pnumb * domain_pnumb * sample_ddims >= 0)
          << "Integer value overflow!";
      //calculate bandwidth
      const Dtype buffer_absum = pairwise_ssdx_summation_main_gpu(
        source_dnumb, source_ddims,
        target_dnumb, target_ddims,
        source_pnumb, source_ddims,
        target_pnumb, target_ddims,
        source_pdata, target_pdata,
        source_datum, source_diffs,
        target_datum, target_diffs
      );
      Dtype buffer_gamma = domain_pnumb * (domain_pnumb - 1) / buffer_absum;
      buffer_gamma *= !absfix_flag_ || 0 < buffer_gamma;
      buffer_gmap_[source_label][target_label] = buffer_gamma;
      //calculate each kernel of data and loss
      const Dtype kernel_gamma = buffer_gamma / pow(kernel_mult_, (Dtype)(kernel_numb_ / 2));
      *topper_datum += pairwise_mmdx_summation_main_gpu(
        source_dnumb, source_ddims,
        target_dnumb, target_ddims,
        source_pnumb, source_ddims,
        target_pnumb, target_ddims,
        sample_pnumb, 4,
        kernel_numb_, kernel_gamma,
        kernel_mult_, random_datum,
        source_pdata, target_pdata,
        source_datum, source_diffs,
        target_datum, target_diffs
      );
    }
  }
}

template <typename Dtype>
void MMDXLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  const int source_ddims = bottom[0]->count() / bottom[0]->shape(0);
  const int target_ddims = bottom[1]->count() / bottom[1]->shape(0);
  const Dtype* source_datum = bottom[0]->gpu_data();
  const Dtype* target_datum = bottom[1]->gpu_data();
  Dtype* source_diffs = bottom[0]->mutable_gpu_diff();
  Dtype* target_diffs = bottom[1]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), source_diffs);
  caffe_gpu_set(bottom[1]->count(), Dtype(0), target_diffs);
  typename map<int, map<int, shared_ptr<Blob<int> > > >::iterator random_miter = random_mmap_.begin();
  for (; random_miter != random_mmap_.end(); ++random_miter) {
    const int source_label = random_miter->first;
    map<int, shared_ptr<Blob<int> > >& random_nbmap = random_miter->second;
    vector<int>& source_vect = source_vmap_[source_label];
    const int source_pnumb = source_vect.size();
    typename map<int, shared_ptr<Blob<int> > >::iterator random_biter = random_nbmap.begin();
    for (; random_biter != random_nbmap.end(); ++random_biter) {
      const int target_label = random_biter->first;
      shared_ptr<Blob<int> > random_nblob = random_biter->second;
      vector<int>& target_vect = target_vmap_[target_label];
      const int target_pnumb = target_vect.size();
      if(source_pnumb < 2 || target_pnumb < 2) continue;
      shared_ptr<Blob<Dtype> > buffer_dblob = buffer_mmap_[source_label][target_label];
      const Dtype buffer_gamma = buffer_gmap_[source_label][target_label];
      const int* random_datum = random_nblob->cpu_data();
      Dtype* buffer_datum = buffer_dblob->mutable_gpu_data();
      const int sample_pnumb = source_pnumb < target_pnumb ? target_pnumb : source_pnumb;
      for(int sample_index = 0; sample_index < sample_pnumb; ++sample_index, random_datum += 4) {
        const int source_indx0 = random_datum[0] % source_pnumb;
        const int source_indx1 = random_datum[1] % source_pnumb;
        const int source_indx2 = (source_indx1 + (source_indx0 == source_indx1)) % source_pnumb;
        const int target_indx0 = random_datum[2] % target_pnumb;
        const int target_indx1 = random_datum[3] % target_pnumb;
        const int target_indx2 = (target_indx1 + (target_indx0 == target_indx1)) % target_pnumb;
        const Dtype* source_data1 = source_datum + source_vect[source_indx0] * source_ddims;
        const Dtype* source_data2 = source_datum + source_vect[source_indx2] * source_ddims;
        const Dtype* target_data1 = target_datum + target_vect[target_indx0] * target_ddims;
        const Dtype* target_data2 = target_datum + target_vect[target_indx2] * target_ddims;
        Dtype* source_diff1 = source_diffs + source_vect[source_indx0] * source_ddims;
        Dtype* source_diff2 = source_diffs + source_vect[source_indx2] * source_ddims;
        Dtype* target_diff1 = target_diffs + target_vect[target_indx0] * target_ddims;
        Dtype* target_diff2 = target_diffs + target_vect[target_indx2] * target_ddims;
        pairwise_mmdx_derivative_gpu(
          source_ddims, source_ddims,
          kernel_numb_, buffer_gamma,
          kernel_mult_, sample_pnumb / loss_weight_,
          source_data1, source_diff1,
          source_data2, source_diff2,
          buffer_datum
        );
        pairwise_mmdx_derivative_gpu(
          source_ddims, target_ddims,
          kernel_numb_, buffer_gamma,
          kernel_mult_, -sample_pnumb / loss_weight_,
          source_data2, source_diff2,
          target_data1, target_diff1,
          buffer_datum
        );
        pairwise_mmdx_derivative_gpu(
          target_ddims, target_ddims,
          kernel_numb_, buffer_gamma,
          kernel_mult_, sample_pnumb / loss_weight_,
          target_data1, target_diff1,
          target_data2, target_diff2,
          buffer_datum
        );
        pairwise_mmdx_derivative_gpu(
          target_ddims, source_ddims,
          kernel_numb_, buffer_gamma,
          kernel_mult_, -sample_pnumb / loss_weight_,
          target_data2, target_diff2,
          source_data1, source_diff1,
          buffer_datum
        );
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDXLossLayer);
} // namespace caffe