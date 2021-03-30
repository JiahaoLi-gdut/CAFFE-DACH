#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/////////////////////////////////////////////////////////////////////////////////ave pooling

template <typename Dtype>
__global__ void ave_pooling_gpu(const int n,
    const int  c_x, const int  h_x, const int  w_x,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, Dtype* const y) {
  const int w_z = max(0, w_x + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_x + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_x + 2 * c_p - c_k) / c_s + 1;
  const int w_y = w_z + (w_c && max(0, w_x + 2 * w_p - w_k) % w_s && w_z * w_s < w_x + w_p);
  const int h_y = h_z + (h_c && max(0, h_x + 2 * h_p - h_k) % h_s && h_z * h_s < h_x + h_p);
  const int c_y = c_z + (c_c && max(0, c_x + 2 * c_p - c_k) % c_s && c_z * c_s < c_x + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = max(y_w * w_s - w_p, 0);
    const int b_h = max(y_h * h_s - h_p, 0);
    const int b_c = max(y_c * c_s - c_p, 0);
    const int e_w = min(y_w * w_s + w_k - w_p, w_x);
    const int e_h = min(y_h * h_s + h_k - h_p, h_x);
    const int e_c = min(y_c * c_s + c_k - c_p, c_x);
    const int w_g = e_w - b_w;
    const int h_g = e_h - b_h;
    const int c_g = e_c - b_c;
    const int x_i = y_n * c_x * h_x * w_x;
    y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j];
        }
      }
    }
    y[y_i] /= w_g * h_g * c_g;
  }
}

template <typename Dtype>
__global__ void ave_unpooling_gpu(const int n,
    const int  c_y, const int  h_y, const int  w_y,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, Dtype* const y) {
  const int w_z = max(0, w_y + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_y + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_y + 2 * c_p - c_k) / c_s + 1;
  const int w_x = w_z + (w_c && max(0, w_y + 2 * w_p - w_k) % w_s && w_z * w_s < w_y + w_p);
  const int h_x = h_z + (h_c && max(0, h_y + 2 * h_p - h_k) % h_s && h_z * h_s < h_y + h_p);
  const int c_x = c_z + (c_c && max(0, c_y + 2 * c_p - c_k) % c_s && c_z * c_s < c_y + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w + w_p < w_k) ? 0 : ((y_w + w_p - w_k) / w_s + 1);
    const int b_h = (y_h + h_p < h_k) ? 0 : ((y_h + h_p - h_k) / h_s + 1);
    const int b_c = (y_c + c_p < c_k) ? 0 : ((y_c + c_p - c_k) / c_s + 1);
    const int e_w = min((y_w + w_p) / w_s + 1, w_x);
    const int e_h = min((y_h + h_p) / h_s + 1, h_x);
    const int e_c = min((y_c + c_p) / c_s + 1, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int o_c = max(x_c * c_s - c_p, 0);
      const int t_c = min(x_c * c_s + c_k - c_p, c_y);
      const int c_g = t_c - o_c;
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int o_h = max(x_h * h_s - h_p, 0);
        const int t_h = min(x_h * h_s + h_k - h_p, h_y);
        const int h_g = t_h - o_h;
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int o_w = max(x_w * w_s - w_p, 0);
          const int t_w = min(x_w * w_s + w_k - w_p, w_y);
          const int w_g = t_w - o_w;
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j] / (w_g * h_g * c_g);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////max pooling

template <typename Dtype>
__global__ void max_pooling_gpu(const int n,
    const int  c_x, const int  h_x, const int  w_x,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, Dtype* const y, Dtype* const m) {
  const int w_z = max(0, w_x + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_x + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_x + 2 * c_p - c_k) / c_s + 1;
  const int w_y = w_z + (w_c && max(0, w_x + 2 * w_p - w_k) % w_s && w_z * w_s < w_x + w_p);
  const int h_y = h_z + (h_c && max(0, h_x + 2 * h_p - h_k) % h_s && h_z * h_s < h_x + h_p);
  const int c_y = c_z + (c_c && max(0, c_x + 2 * c_p - c_k) % c_s && c_z * c_s < c_x + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = max(y_w * w_s - w_p, 0);
    const int b_h = max(y_h * h_s - h_p, 0);
    const int b_c = max(y_c * c_s - c_p, 0);
    const int e_w = min(y_w * w_s + w_k - w_p, w_x);
    const int e_h = min(y_h * h_s + h_k - h_p, h_x);
    const int e_c = min(y_c * c_s + c_k - c_p, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    const int x_j = (b_c * h_x + b_h) * w_x + b_w;
    y[y_i] = x[x_i + x_j]; m[y_i] = x_j;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_k = (x_c * h_x + x_h) * w_x + x_w;
          if (y[y_i] < x[x_i + x_k]) {
            y[y_i] = x[x_i + x_k];
            m[y_i] = x_k;
          }
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void max_unpooling_gpu(const int n,
    const int  c_y, const int  h_y, const int  w_y,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, const Dtype* const m, Dtype* const y) {
  const int w_z = max(0, w_y + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_y + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_y + 2 * c_p - c_k) / c_s + 1;
  const int w_x = w_z + (w_c && max(0, w_y + 2 * w_p - w_k) % w_s && w_z * w_s < w_y + w_p);
  const int h_x = h_z + (h_c && max(0, h_y + 2 * h_p - h_k) % h_s && h_z * h_s < h_y + h_p);
  const int c_x = c_z + (c_c && max(0, c_y + 2 * c_p - c_k) % c_s && c_z * c_s < c_y + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w + w_p < w_k) ? 0 : ((y_w + w_p - w_k) / w_s + 1);
    const int b_h = (y_h + h_p < h_k) ? 0 : ((y_h + h_p - h_k) / h_s + 1);
    const int b_c = (y_c + c_p < c_k) ? 0 : ((y_c + c_p - c_k) / c_s + 1);
    const int e_w = min((y_w + w_p) / w_s + 1, w_x);
    const int e_h = min((y_h + h_p) / h_s + 1, h_x);
    const int e_c = min((y_c + c_p) / c_s + 1, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    const int y_j = (y_c * h_y + y_h) * w_y + y_w;
    y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j] * (m[x_i + x_j] == y_j);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////sto pooling

template <typename Dtype>
__global__ void sto_pooling_train_gpu(const int n,
    const int  c_x, const int  h_x, const int  w_x,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, const Dtype* const r, Dtype* const y, Dtype* const m) {
  const int w_z = max(0, w_x + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_x + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_x + 2 * c_p - c_k) / c_s + 1;
  const int w_y = w_z + (w_c && max(0, w_x + 2 * w_p - w_k) % w_s && w_z * w_s < w_x + w_p);
  const int h_y = h_z + (h_c && max(0, h_x + 2 * h_p - h_k) % h_s && h_z * h_s < h_x + h_p);
  const int c_y = c_z + (c_c && max(0, c_x + 2 * c_p - c_k) % c_s && c_z * c_s < c_x + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = max(y_w * w_s - w_p, 0);
    const int b_h = max(y_h * h_s - h_p, 0);
    const int b_c = max(y_c * c_s - c_p, 0);
    const int e_w = min(y_w * w_s + w_k - w_p, w_x);
    const int e_h = min(y_h * h_s + h_k - h_p, h_x);
    const int e_c = min(y_c * c_s + c_k - c_p, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    y[y_i] = m[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j];
        }
      }
    }
    y[y_i] *= r[y_i];
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          if ((m[y_i] += x[x_i + x_j]) - y[y_i] >= 0) {
            y[y_i] = x[x_i + x_j];
            m[y_i] = x_j;
            x_w = e_w - 1;
            x_h = e_h - 1;
            x_c = e_c - 1;
          }
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void sto_pooling_test_gpu(const int n,
    const int  c_x, const int  h_x, const int  w_x,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, Dtype* const y) {
  const int w_z = max(0, w_x + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_x + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_x + 2 * c_p - c_k) / c_s + 1;
  const int w_y = w_z + (w_c && max(0, w_x + 2 * w_p - w_k) % w_s && w_z * w_s < w_x + w_p);
  const int h_y = h_z + (h_c && max(0, h_x + 2 * h_p - h_k) % h_s && h_z * h_s < h_x + h_p);
  const int c_y = c_z + (c_c && max(0, c_x + 2 * c_p - c_k) % c_s && c_z * c_s < c_x + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = max(y_w * w_s - w_p, 0);
    const int b_h = max(y_h * h_s - h_p, 0);
    const int b_c = max(y_c * c_s - c_p, 0);
    const int e_w = min(y_w * w_s + w_k - w_p, w_x);
    const int e_h = min(y_h * h_s + h_k - h_p, h_x);
    const int e_c = min(y_c * c_s + c_k - c_p, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    Dtype sum = y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j] * x[x_i + x_j];
          sum += x[x_i + x_j];
        }
      }
    }
    y[y_i] = 0 < sum ? (y[y_i] / sum) : 0;
  }
}

template <typename Dtype>
__global__ void sto_unpooling_gpu(const int n,
    const int  c_y, const int  h_y, const int  w_y,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, const Dtype* const m, Dtype* const y) {
  const int w_z = max(0, w_y + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_y + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_y + 2 * c_p - c_k) / c_s + 1;
  const int w_x = w_z + (w_c && max(0, w_y + 2 * w_p - w_k) % w_s && w_z * w_s < w_y + w_p);
  const int h_x = h_z + (h_c && max(0, h_y + 2 * h_p - h_k) % h_s && h_z * h_s < h_y + h_p);
  const int c_x = c_z + (c_c && max(0, c_y + 2 * c_p - c_k) % c_s && c_z * c_s < c_y + c_p);
  const int m_y = n * c_y * h_y * w_y;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w + w_p < w_k) ? 0 : ((y_w + w_p - w_k) / w_s + 1);
    const int b_h = (y_h + h_p < h_k) ? 0 : ((y_h + h_p - h_k) / h_s + 1);
    const int b_c = (y_c + c_p < c_k) ? 0 : ((y_c + c_p - c_k) / c_s + 1);
    const int e_w = min((y_w + w_p) / w_s + 1, w_x);
    const int e_h = min((y_h + h_p) / h_s + 1, h_x);
    const int e_c = min((y_c + c_p) / c_s + 1, c_x);
    const int x_i = y_n * c_x * h_x * w_x;
    const int y_j = (y_c * h_y + y_h) * w_y + y_w;
    y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j] * (m[x_i + x_j] == y_j);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_AVE:
    ave_pooling_gpu<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->gpu_data(), top[0]->mutable_gpu_data()
    ); break;
  case PoolingParameter_PoolMethod_MAX:
    max_pooling_gpu<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), msk_rnd_.mutable_gpu_data()
    ); break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      caffe_gpu_rng_uniform(msk_rnd_.count(), Dtype(0), Dtype(1), msk_rnd_.mutable_gpu_diff());
      sto_pooling_train_gpu<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->shape(0),
        bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_,
        stride_c_, stride_h_, stride_w_,
        pad_c_, pad_h_, pad_w_,
        ceil_c_, ceil_h_, ceil_w_,
        bottom[0]->gpu_data(), msk_rnd_.gpu_diff(), top[0]->mutable_gpu_data(), msk_rnd_.mutable_gpu_data()
      );
    } else {
      sto_pooling_test_gpu<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->shape(0),
        bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_,
        stride_c_, stride_h_, stride_w_,
        pad_c_, pad_h_, pad_w_,
        ceil_c_, ceil_h_, ceil_w_,
        bottom[0]->gpu_data(), top[0]->mutable_gpu_data()
      );
    } break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_AVE:
    ave_unpooling_gpu<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff()
    ); break;
  case PoolingParameter_PoolMethod_MAX:
    max_unpooling_gpu<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->gpu_diff(), msk_rnd_.gpu_data(), bottom[0]->mutable_gpu_diff()
    ); break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    sto_unpooling_gpu<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->gpu_diff(), msk_rnd_.gpu_data(), bottom[0]->mutable_gpu_diff()
    ); break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);
} // namespace caffe