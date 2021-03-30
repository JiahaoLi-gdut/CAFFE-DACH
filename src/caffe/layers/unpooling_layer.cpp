#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

/////////////////////////////////////////////////////////////////////////////////ave unpooling

//same as ave_unpooling_cpu in pooling layer.
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

//same as ave_pooling_cpu in pooling layer.
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

/////////////////////////////////////////////////////////////////////////////////max unpooling

//same as max_unpooling_cpu in pooling layer.
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

//not same as max_pooling_cpu in pooling layer.
template <typename Dtype>
void max_pooling_cpu(const int n,
    const int  c_x, const int  h_x, const int  w_x,
    const int  c_k, const int  h_k, const int  w_k,
    const int  c_s, const int  h_s, const int  w_s,
    const int  c_p, const int  h_p, const int  w_p,
    const bool c_c, const bool h_c, const bool w_c,
    const Dtype* const x, const Dtype* const m, Dtype* const y) {
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
    y[y_i] = 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j] * (m[y_i] == x_j);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////tile unpooling

template <typename Dtype>
void tile_unpooling_cpu(const int n,
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
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_j = (x_c * h_x + x_h) * w_x + x_w;
          y[y_i] += x[x_i + x_j];
        }
      }
    }
  }
}

template <typename Dtype>
void tile_pooling_cpu(const int n,
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

/////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  UnpoolingParameter unpool_param = this->layer_param_.unpooling_param();
  // Check bottom dimension
  CHECK_EQ(4, bottom[0]->num_axes())
    << "Input must have 4 axes, corresponding to (num, channels, height, width)";
  // Check ceil
  if (unpool_param.has_ceil3d()) {
    CHECK(!unpool_param.has_ceil2d() || unpool_param.ceil2d() == unpool_param.ceil3d())
      << "ceil2d and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_ceil_c() || unpool_param.ceil_c() == unpool_param.ceil3d())
      << "ceil_c and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_ceil_h() || unpool_param.ceil_h() == unpool_param.ceil3d())
      << "ceil_h and ceil3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_ceil_w() || unpool_param.ceil_w() == unpool_param.ceil3d())
      << "ceil_w and ceil3d can not be valued differently, "
      << "when they are specified by user";
    ceil_c_ = ceil_h_ = ceil_w_ = unpool_param.ceil3d();
  } else if (unpool_param.has_ceil2d()) {
    CHECK(!unpool_param.has_ceil_h() || unpool_param.ceil_h() == unpool_param.ceil2d())
      << "ceil_h and ceil2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_ceil_w() || unpool_param.ceil_w() == unpool_param.ceil2d())
      << "ceil_w and ceil2d can not be valued differently, "
      << "when they are specified by user";
    ceil_c_ = unpool_param.has_ceil_c() ? unpool_param.ceil_c() : unpool_param.ceil3d();
    ceil_h_ = ceil_w_ = unpool_param.ceil2d();
  } else {
    ceil_c_ = unpool_param.has_ceil_c() ? unpool_param.ceil_c() : unpool_param.ceil3d();
    ceil_h_ = unpool_param.has_ceil_h() ? unpool_param.ceil_h() : unpool_param.ceil3d();
    ceil_w_ = unpool_param.has_ceil_w() ? unpool_param.ceil_w() : unpool_param.ceil3d();
  }
  // Check pad
  if (unpool_param.has_pad3d()) {
    CHECK(!unpool_param.has_pad2d() || unpool_param.pad2d() == unpool_param.pad3d())
      << "pad2d and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_pad_c() || unpool_param.pad_c() == unpool_param.pad3d())
      << "pad_c and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_pad_h() || unpool_param.pad_h() == unpool_param.pad3d())
      << "pad_h and pad3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_pad_w() || unpool_param.pad_w() == unpool_param.pad3d())
      << "pad_w and pad3d can not be valued differently, "
      << "when they are specified by user";
    pad_c_ = pad_h_ = pad_w_ = unpool_param.pad3d();
  } else if (unpool_param.has_pad2d()) {
    CHECK(!unpool_param.has_pad_h() || unpool_param.pad_h() == unpool_param.pad2d())
      << "pad_h and pad2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_pad_w() || unpool_param.pad_w() == unpool_param.pad2d())
      << "pad_w and pad2d can not be valued differently, "
      << "when they are specified by user";
    pad_c_ = unpool_param.has_pad_c() ? unpool_param.pad_c() : unpool_param.pad3d();
    pad_h_ = pad_w_ = unpool_param.pad2d();
  } else {
    pad_c_ = unpool_param.has_pad_c() ? unpool_param.pad_c() : unpool_param.pad3d();
    pad_h_ = unpool_param.has_pad_h() ? unpool_param.pad_h() : unpool_param.pad3d();
    pad_w_ = unpool_param.has_pad_w() ? unpool_param.pad_w() : unpool_param.pad3d();
  }
  // Check stride
  if (unpool_param.has_stride3d()) {
    CHECK(!unpool_param.has_stride2d() || unpool_param.stride2d() == unpool_param.stride3d())
      << "stride2d and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_stride_c() || unpool_param.stride_c() == unpool_param.stride3d())
      << "stride_c and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_stride_h() || unpool_param.stride_h() == unpool_param.stride3d())
      << "stride_h and stride3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_stride_w() || unpool_param.stride_w() == unpool_param.stride3d())
      << "stride_w and stride3d can not be valued differently, "
      << "when they are specified by user";
    stride_c_ = stride_h_ = stride_w_ = unpool_param.stride3d();
  } else if (unpool_param.has_stride2d()) {
    CHECK(!unpool_param.has_stride_h() || unpool_param.stride_h() == unpool_param.stride2d())
      << "stride_h and stride2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_stride_w() || unpool_param.stride_w() == unpool_param.stride2d())
      << "stride_w and stride2d can not be valued differently, "
      << "when they are specified by user";
    stride_c_ = unpool_param.has_stride_c() ? unpool_param.stride_c() : unpool_param.stride3d();
    stride_h_ = stride_w_ = unpool_param.stride2d();
  } else {
    stride_c_ = unpool_param.has_stride_c() ? unpool_param.stride_c() : unpool_param.stride3d();
    stride_h_ = unpool_param.has_stride_h() ? unpool_param.stride_h() : unpool_param.stride3d();
    stride_w_ = unpool_param.has_stride_w() ? unpool_param.stride_w() : unpool_param.stride3d();
  }
  CHECK_GT(stride_c_, 0) << "Strider channels cannot be zero.";
  CHECK_GT(stride_h_, 0) << "Strider height cannot be zero.";
  CHECK_GT(stride_w_, 0) << "Strider width cannot be zero.";
  // Check kernel
  if (unpool_param.has_kernel3d()) {
    CHECK(!unpool_param.has_kernel2d() || unpool_param.kernel2d() == unpool_param.kernel3d())
      << "kernel2d and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_kernel_c() || unpool_param.kernel_c() == unpool_param.kernel3d())
      << "kernel_c and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_kernel_h() || unpool_param.kernel_h() == unpool_param.kernel3d())
      << "kernel_h and kernel3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_kernel_w() || unpool_param.kernel_w() == unpool_param.kernel3d())
      << "kernel_w and kernel3d can not be valued differently, "
      << "when they are specified by user";
    kernel_c_ = kernel_h_ = kernel_w_ = unpool_param.kernel3d();
  } else if (unpool_param.has_kernel2d()) {
    CHECK(!unpool_param.has_kernel_h() || unpool_param.kernel_h() == unpool_param.kernel2d())
      << "kernel_h and kernel2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_kernel_w() || unpool_param.kernel_w() == unpool_param.kernel2d())
      << "kernel_w and kernel2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(unpool_param.has_kernel_c())
      << "kernel_c should be valued "
      << "when kernel2d is specified by user";
    kernel_c_ = unpool_param.kernel_c();
    kernel_h_ = kernel_w_ = unpool_param.kernel2d();
  } else {
    CHECK(unpool_param.has_kernel_c() && unpool_param.has_kernel_h() && unpool_param.has_kernel_w())
      << "kernel_c, kernel_h and kernel_w should be specified "
      << "when kernel2d and kernel3d are not specified by user";
    kernel_c_ = unpool_param.kernel_c();
    kernel_h_ = unpool_param.kernel_h();
    kernel_w_ = unpool_param.kernel_w();
  }
  CHECK_GT(kernel_c_, pad_c_) << "Kernel channels can not be less than or equal to Pad channels.";
  CHECK_GT(kernel_h_, pad_h_) << "Kernel height can not be less than or equal to Pad height.";
  CHECK_GT(kernel_w_, pad_w_) << "Kernel width can not be less than or equal to Pad width.";
  // Check cutext
  if (unpool_param.has_cutext3d()) {
    CHECK(!unpool_param.has_cutext2d() || unpool_param.cutext2d() == unpool_param.cutext3d())
      << "cutext2d and cutext3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_cutext_c() || unpool_param.cutext_c() == unpool_param.cutext3d())
      << "cutext_c and cutext3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_cutext_h() || unpool_param.cutext_h() == unpool_param.cutext3d())
      << "cutext_h and cutext3d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_cutext_w() || unpool_param.cutext_w() == unpool_param.cutext3d())
      << "cutext_w and cutext3d can not be valued differently, "
      << "when they are specified by user";
    cutext_c_ = cutext_h_ = cutext_w_ = unpool_param.cutext3d();
  } else if (unpool_param.has_cutext2d()) {
    CHECK(!unpool_param.has_cutext_h() || unpool_param.cutext_h() == unpool_param.cutext2d())
      << "cutext_h and cutext2d can not be valued differently, "
      << "when they are specified by user";
    CHECK(!unpool_param.has_cutext_w() || unpool_param.cutext_w() == unpool_param.cutext2d())
      << "cutext_w and cutext2d can not be valued differently, "
      << "when they are specified by user";
    cutext_c_ = unpool_param.has_cutext_c() ? unpool_param.cutext_c() : unpool_param.cutext3d();
    cutext_h_ = cutext_w_ = unpool_param.cutext2d();
  } else {
    cutext_c_ = unpool_param.has_cutext_c() ? unpool_param.cutext_c() : unpool_param.cutext3d();
    cutext_h_ = unpool_param.has_cutext_h() ? unpool_param.cutext_h() : unpool_param.cutext3d();
    cutext_w_ = unpool_param.has_cutext_w() ? unpool_param.cutext_w() : unpool_param.cutext3d();
  }
  if (ceil_c_) {
    CHECK_LT(cutext_c_, min(kernel_c_, stride_c_))
      << "Ceil_c: true; Cutext channels must be less than Kernel channels and Strider channels.";
  } else {
    CHECK_LT(cutext_c_, stride_c_)
      << "Ceil_c: false; Cutext channels must be less than Strider channels.";
  }
  if (ceil_h_) {
    CHECK_LT(cutext_h_, min(kernel_h_, stride_h_))
      << "Ceil_h:  true; Cutext height must be less than Kernel height and Strider height.";
  } else {
    CHECK_LT(cutext_h_, stride_h_)
      << "Ceil_h: false; Cutext height must be less than Strider height.";
  }
  if (ceil_w_) {
    CHECK_LT(cutext_w_, min(kernel_w_, stride_w_))
      << "Ceil_w:  true; Cutext width must be less than Kernel width and Strider width.";
  } else {
    CHECK_LT(cutext_w_, stride_w_)
      << "Ceil_w: false; Cutext width must be less than Strider width.";
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  const int w_p = pad_w_;
  const int h_p = pad_h_;
  const int c_p = pad_c_;
  const int w_s = stride_w_;
  const int h_s = stride_h_;
  const int c_s = stride_c_;
  const int w_k = kernel_w_;
  const int h_k = kernel_h_;
  const int c_k = kernel_c_;
  const int w_x = bottom[0]->shape(3);
  const int h_x = bottom[0]->shape(2);
  const int c_x = bottom[0]->shape(1);
  const int w_y = (w_x - 1) * w_s + w_k - w_p + (ceil_w_ ? -cutext_w_ : cutext_w_);
  const int h_y = (h_x - 1) * h_s + h_k - h_p + (ceil_h_ ? -cutext_h_ : cutext_h_);
  const int c_y = (c_x - 1) * c_s + c_k - c_p + (ceil_c_ ? -cutext_c_ : cutext_c_);
  top[0]->Reshape(bottom[0]->shape(0), c_y, h_y, w_y);
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_AVE:
    ave_unpooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->cpu_data(), top[0]->mutable_cpu_data()
    ); break;
  case UnpoolingParameter_UnpoolMethod_MAX:
    max_unpooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->cpu_data(), bottom[1]->cpu_data(), top[0]->mutable_cpu_data()
    ); break;
  case UnpoolingParameter_UnpoolMethod_TILE:
    tile_unpooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      bottom[0]->cpu_data(), top[0]->mutable_cpu_data()
    ); break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (!propagate_down[0]) return;
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_AVE:
    ave_pooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff()
    ); break;
  case UnpoolingParameter_UnpoolMethod_MAX:
     max_pooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), bottom[1]->cpu_data(), bottom[0]->mutable_cpu_diff()
    ); break;
  case UnpoolingParameter_UnpoolMethod_TILE:
    tile_pooling_cpu(
      top[0]->shape(0),
      top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_,
      stride_c_, stride_h_, stride_w_,
      pad_c_, pad_h_, pad_w_,
      ceil_c_, ceil_h_, ceil_w_,
      top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff()
    ); break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif
INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(Unpooling);
} // namespace caffe