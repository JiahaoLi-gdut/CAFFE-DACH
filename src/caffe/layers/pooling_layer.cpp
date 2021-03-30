#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

/////////////////////////////////////////////////////////////////////////////////ave pooling

template <typename Dtype>
void ave_pooling_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void ave_unpooling_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void max_pooling_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void max_unpooling_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void sto_pooling_train_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void sto_pooling_test_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void sto_unpooling_cpu(const int n,
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
  for (int y_i = 0; y_i < m_y; ++y_i) {
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
void PoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  // Check bottom dimension
  CHECK_EQ(4, bottom[0]->num_axes())
    << "Input must have 4 axes, corresponding to (num, channels, height, width)";
  // Check ceil
  if (pool_param.has_ceil3d()) {
    CHECK(!pool_param.has_ceil2d() || pool_param.ceil2d() == pool_param.ceil3d())
      << "ceil2d and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_ceil_c() || pool_param.ceil_c() == pool_param.ceil3d())
      << "ceil_c and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_ceil_h() || pool_param.ceil_h() == pool_param.ceil3d())
      << "ceil_h and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_ceil_w() || pool_param.ceil_w() == pool_param.ceil3d())
      << "ceil_w and ceil3d can not be valued differently, "
      << "when they are specified by user";
    ceil_c_ = ceil_h_ = ceil_w_ = pool_param.ceil3d();
  } else if (pool_param.has_ceil2d()) {
    CHECK(!pool_param.has_ceil_h() || pool_param.ceil_h() == pool_param.ceil2d())
      << "ceil_h and ceil2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_ceil_w() || pool_param.ceil_w() == pool_param.ceil2d())
      << "ceil_w and ceil2d can not be valued differently, "
      << "when they are specified by user";
    ceil_c_ = pool_param.has_ceil_c() ? pool_param.ceil_c() : pool_param.ceil3d();
    ceil_h_ = ceil_w_ = pool_param.ceil2d();
  } else {
    ceil_c_ = pool_param.has_ceil_c() ? pool_param.ceil_c() : pool_param.ceil3d();
    ceil_h_ = pool_param.has_ceil_h() ? pool_param.ceil_h() : pool_param.ceil3d();
    ceil_w_ = pool_param.has_ceil_w() ? pool_param.ceil_w() : pool_param.ceil3d();
  }
  // Check pad
  if (pool_param.has_pad3d()) {
    CHECK(!pool_param.has_pad2d() || pool_param.pad2d() == pool_param.pad3d())
      << "pad2d and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_pad_c() || pool_param.pad_c() == pool_param.pad3d())
      << "pad_c and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_pad_h() || pool_param.pad_h() == pool_param.pad3d())
      << "pad_h and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_pad_w() || pool_param.pad_w() == pool_param.pad3d())
      << "pad_w and pad3d can not be valued differently, "
      << "when they are specified by user";
    pad_c_ = pad_h_ = pad_w_ = pool_param.pad3d();
  } else if (pool_param.has_pad2d()) {
    CHECK(!pool_param.has_pad_h() || pool_param.pad_h() == pool_param.pad2d())
      << "pad_h and pad2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_pad_w() || pool_param.pad_w() == pool_param.pad2d())
      << "pad_w and pad2d can not be valued differently, "
      << "when they are specified by user";
    pad_c_ = pool_param.has_pad_c() ? pool_param.pad_c() : pool_param.pad3d();
    pad_h_ = pad_w_ = pool_param.pad2d();
  } else {
    pad_c_ = pool_param.has_pad_c() ? pool_param.pad_c() : pool_param.pad3d();
    pad_h_ = pool_param.has_pad_h() ? pool_param.pad_h() : pool_param.pad3d();
    pad_w_ = pool_param.has_pad_w() ? pool_param.pad_w() : pool_param.pad3d();
  }
  // Check stride
  if (pool_param.has_stride3d()) {
    CHECK(!pool_param.has_stride2d() || pool_param.stride2d() == pool_param.stride3d())
      << "stride2d and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_stride_c() || pool_param.stride_c() == pool_param.stride3d())
      << "stride_c and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_stride_h() || pool_param.stride_h() == pool_param.stride3d())
      << "stride_h and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_stride_w() || pool_param.stride_w() == pool_param.stride3d())
      << "stride_w and stride3d can not be valued differently, "
      << "when they are specified by user";
    stride_c_ = stride_h_ = stride_w_ = pool_param.stride3d();
  } else if (pool_param.has_stride2d()) {
    CHECK(!pool_param.has_stride_h() || pool_param.stride_h() == pool_param.stride2d())
      << "stride_h and stride2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_stride_w() || pool_param.stride_w() == pool_param.stride2d())
      << "stride_w and stride2d can not be valued differently, "
      << "when they are specified by user";
    stride_c_ = pool_param.has_stride_c() ? pool_param.stride_c() : pool_param.stride3d();
    stride_h_ = stride_w_ = pool_param.stride2d();
  } else {
    stride_c_ = pool_param.has_stride_c() ? pool_param.stride_c() : pool_param.stride3d();
    stride_h_ = pool_param.has_stride_h() ? pool_param.stride_h() : pool_param.stride3d();
    stride_w_ = pool_param.has_stride_w() ? pool_param.stride_w() : pool_param.stride3d();
  }
  CHECK_GT(stride_c_, 0) << "Strider channels cannot be zero.";
  CHECK_GT(stride_h_, 0) << "Strider height cannot be zero.";
  CHECK_GT(stride_w_, 0) << "Strider width cannot be zero.";
  // Check global_pooling and kernel
  global_pooling2d_ = global_pooling3d_ = false;
  if (pool_param.has_global_pooling3d()) {
    CHECK(pad_c_ == 0 && stride_c_ == 1 &&
          pad_h_ == 0 && stride_h_ == 1 &&
          pad_w_ == 0 && stride_w_ == 1)
      << "With global_pooling3d: true; only pad_c = pad_h = pad_w = 0 "
      << "and stride_c = stride_h = stride_w = 1";
    CHECK(!pool_param.has_global_pooling2d() || pool_param.global_pooling2d() == pool_param.global_pooling3d())
      << "global_pooling2d and global_pooling3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_kernel3d() && !pool_param.has_kernel2d())
      << "kernel3d and kernel2d can not be valued "
      << "when global_pooling3d is specified by user";
    CHECK(!pool_param.has_kernel_c() || pool_param.kernel_c() == bottom[0]->shape(1))
      << "kernel_c can not be specified a value different with the channels of bottom[0], "
      << "when global_pooling3d is specified by user";
    CHECK(!pool_param.has_kernel_h() || pool_param.kernel_h() == bottom[0]->shape(2))
      << "kernel_h can not be specified a value different with the height of bottom[0], "
      << "when global_pooling3d is specified by user";
    CHECK(!pool_param.has_kernel_w() || pool_param.kernel_w() == bottom[0]->shape(3))
      << "kernel_w can not be specified a value different with the width of bottom[0], "
      << "when global_pooling3d is specified by user";
    global_pooling2d_ = pool_param.global_pooling3d();
    global_pooling3d_ = pool_param.global_pooling3d();
    kernel_c_ = bottom[0]->shape(1);
    kernel_h_ = bottom[0]->shape(2);
    kernel_w_ = bottom[0]->shape(3);
  } else if (pool_param.has_global_pooling2d()) {
    CHECK(pad_h_ == 0 && stride_h_ == 1 &&
          pad_w_ == 0 && stride_w_ == 1)
      << "With global_pooling2d: true; only pad_h = pad_w = 0 "
      << "and stride_h = stride_w = 1";
    CHECK(!pool_param.has_kernel3d() && !pool_param.has_kernel2d())
      << "kernel3d and kernel2d can not be valued "
      << "when global_pooling2d is specified by user";
    CHECK(!pool_param.has_kernel_h() || pool_param.kernel_h() == bottom[0]->shape(2))
      << "kernel_h can not be specified a value different with the height of bottom[0], "
      << "when global_pooling2d is specified by user";
    CHECK(!pool_param.has_kernel_w() || pool_param.kernel_w() == bottom[0]->shape(3))
      << "kernel_w can not be specified a value different with the width of bottom[0], "
      << "when global_pooling2d is specified by user";
    CHECK(pool_param.has_kernel_c())
      << "kernel_c should be valued "
      << "when global_pooling2d is specified by user";
    global_pooling2d_ = pool_param.global_pooling2d();
    kernel_c_ = pool_param.kernel_c();
    kernel_h_ = bottom[0]->shape(2);
    kernel_w_ = bottom[0]->shape(3);
  } else if (pool_param.has_kernel3d()) {
    CHECK(!pool_param.has_kernel2d() || pool_param.kernel2d() == pool_param.kernel3d())
      << "kernel2d and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_kernel_c() || pool_param.kernel_c() == pool_param.kernel3d())
      << "kernel_c and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_kernel_h() || pool_param.kernel_h() == pool_param.kernel3d())
      << "kernel_h and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_kernel_w() || pool_param.kernel_w() == pool_param.kernel3d())
      << "kernel_w and kernel3d can not be valued differently, "
      << "when they are specified by user";
    kernel_c_ = kernel_h_ = kernel_w_ = pool_param.kernel3d();
  } else if (pool_param.has_kernel2d()) {
    CHECK(!pool_param.has_kernel_h() || pool_param.kernel_h() == pool_param.kernel2d())
      << "kernel_h and kernel2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!pool_param.has_kernel_w() || pool_param.kernel_w() == pool_param.kernel2d())
      << "kernel_w and kernel2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(pool_param.has_kernel_c())
      << "kernel_c should be valued "
      << "when kernel2d is specified by user";
    kernel_c_ = pool_param.kernel_c();
    kernel_h_ = kernel_w_ = pool_param.kernel2d();
  } else {
    CHECK(pool_param.has_kernel_c() && pool_param.has_kernel_h() && pool_param.has_kernel_w())
      << "kernel_c, kernel_h and kernel_w should be specified "
      << "when kernel2d and kernel3d are not specified by user";
    kernel_c_ = pool_param.kernel_c();
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  CHECK_GT(kernel_c_, pad_c_) << "Kernel channels can not be less than or equal to Pad channels.";
  CHECK_GT(kernel_h_, pad_h_) << "Kernel height can not be less than or equal to Pad height.";
  CHECK_GT(kernel_w_, pad_w_) << "Kernel width can not be less than or equal to Pad width.";
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes())
    << "Input must have 4 axes, corresponding to (num, channels, height, width)";
  const int w_p = pad_w_;
  const int h_p = pad_h_;
  const int c_p = pad_c_;
  const int w_s = stride_w_;
  const int h_s = stride_h_;
  const int c_s = stride_c_;
  const int w_x = bottom[0]->shape(3);
  const int h_x = bottom[0]->shape(2);
  const int c_x = bottom[0]->shape(1);
  const int w_k = (global_pooling2d_ || global_pooling3d_) ? w_x : kernel_w_;
  const int h_k = (global_pooling2d_ || global_pooling3d_) ? h_x : kernel_h_;
  const int c_k = global_pooling3d_? c_x : kernel_c_;
  const int w_z = max(0, w_x + 2 * w_p - w_k) / w_s + 1;
  const int h_z = max(0, h_x + 2 * h_p - h_k) / h_s + 1;
  const int c_z = max(0, c_x + 2 * c_p - c_k) / c_s + 1;
  const int w_y = w_z + (ceil_w_ && max(0, w_x + 2 * w_p - w_k) % w_s && w_z * w_s < w_x + w_p);
  const int h_y = h_z + (ceil_h_ && max(0, h_x + 2 * h_p - h_k) % h_s && h_z * h_s < h_x + h_p);
  const int c_y = c_z + (ceil_c_ && max(0, c_x + 2 * c_p - c_k) % c_s && c_z * c_s < c_x + c_p);
  top[0]->Reshape(bottom[0]->shape(0), c_y, h_y, w_y);
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX ||
      this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    msk_rnd_.ReshapeLike(*top[0]);
    if (top.size() > 1) {
      top[1]->ReshapeLike(msk_rnd_);
      msk_rnd_.ShareData(*top[1]);
      msk_rnd_.ShareDiff(*top[1]);
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_AVE:
    ave_pooling_cpu(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->cpu_data(), top[0]->mutable_cpu_data()
    ); break;
  case PoolingParameter_PoolMethod_MAX:
    max_pooling_cpu(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), msk_rnd_.mutable_cpu_data()
    ); break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      caffe_rng_uniform(msk_rnd_.count(), Dtype(0), Dtype(1), msk_rnd_.mutable_cpu_diff());
      sto_pooling_train_cpu(
        bottom[0]->shape(0),
        bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_,
        stride_c_, stride_h_, stride_w_,
        pad_c_, pad_h_, pad_w_,
        ceil_c_, ceil_h_, ceil_w_,
        bottom[0]->cpu_data(), msk_rnd_.cpu_diff(), top[0]->mutable_cpu_data(), msk_rnd_.mutable_cpu_data()
      );
    } else {
      sto_pooling_test_cpu(
        bottom[0]->shape(0),
        bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_,
        stride_c_, stride_h_, stride_w_,
        pad_c_, pad_h_, pad_w_,
        ceil_c_, ceil_h_, ceil_w_,
        bottom[0]->cpu_data(), top[0]->mutable_cpu_data()
      );
    } break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_AVE:
    ave_unpooling_cpu(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff()
    ); break;
  case PoolingParameter_PoolMethod_MAX:
    max_unpooling_cpu(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), msk_rnd_.cpu_data(), bottom[0]->mutable_cpu_diff()
    ); break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    sto_unpooling_cpu(
      bottom[0]->shape(0),
      bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), msk_rnd_.cpu_data(), bottom[0]->mutable_cpu_diff()
    ); break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif
INSTANTIATE_CLASS(PoolingLayer);
} // namespace caffe