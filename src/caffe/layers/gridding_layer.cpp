#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/gridding_layer.hpp"

namespace caffe {

using std::min;
using std::max;

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void incomplete_gridding_cpu_aoxpb2y(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = (w_y - 1) * min(w_k, w_s) + min(w_k, w_x - (w_y - 1) * w_s);
  const int h_a = (h_y - 1) * min(h_k, h_s) + min(h_k, h_x - (h_y - 1) * h_s);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = y_w * w_s;
    const int b_h = y_h * h_s;
    const int b_c = y_c * c_s;
    const int e_w = min(b_w + w_k, w_x);
    const int e_h = min(b_h + h_k, h_x);
    const int e_c = min(b_c + c_k, c_x);
    const int b_i = y_i % (c_y * h_y * w_y);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int a_c = y_c * min(c_k, c_s) + x_c - b_c;
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int a_h = y_h * min(h_k, h_s) + x_h - b_h;
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int a_w = y_w * min(w_k, w_s) + x_w - b_w;
          const int a_i = (a_c * h_a + a_h) * w_a + a_w;
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void incomplete_degridding_cpu_aoxpb2y(const int n,
    const int c_y, const int h_y, const int w_y,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_y - w_k) / w_s + 1;
  const int h_z = max(0, h_y - h_k) / h_s + 1;
  const int c_z = max(0, c_y - c_k) / c_s + 1;
  const int w_x = w_z + (max(0, w_y - w_k) % w_s && w_z * w_s < w_y);
  const int h_x = h_z + (max(0, h_y - h_k) % h_s && h_z * h_s < h_y);
  const int c_x = c_z + (max(0, c_y - c_k) % c_s && c_z * c_s < c_y);
  const int w_a = (w_x - 1) * min(w_k, w_s) + min(w_k, w_y - (w_x - 1) * w_s);
  const int h_a = (h_x - 1) * min(h_k, h_s) + min(h_k, h_y - (h_x - 1) * h_s);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w < w_k) ? 0 : ((y_w - w_k) / w_s + 1);
    const int b_h = (y_h < h_k) ? 0 : ((y_h - h_k) / h_s + 1);
    const int b_c = (y_c < c_k) ? 0 : ((y_c - c_k) / c_s + 1);
    const int e_w = min(y_w / w_s + 1, w_x);
    const int e_h = min(y_h / h_s + 1, h_x);
    const int e_c = min(y_c / c_s + 1, c_x);
    const int a_w = y_w - y_w / w_s * max(0, w_s - w_k);
    const int a_h = y_h - y_h / h_s * max(0, h_s - h_k);
    const int a_c = y_c - y_c / c_s * max(0, c_s - c_k);
    const int a_i = (a_c * h_a + a_h) * w_a + a_w;
    const int b_i = y_i % (c_y * h_y * w_y);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void incomplete_degridding_cpu_xoypb2a(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* x, const Dtype* y, const Dtype* b, Dtype* a) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = (w_y - 1) * min(w_k, w_s) + min(w_k, w_x - (w_y - 1) * w_s);
  const int h_a = (h_y - 1) * min(h_k, h_s) + min(h_k, h_x - (h_y - 1) * h_s);
  const int c_a = (c_y - 1) * min(c_k, c_s) + min(c_k, c_x - (c_y - 1) * c_s);
  const int m_a = c_a * h_a * w_a;
  for (int a_i = 0; a_i < m_a; ++a_i) {
    const int a_w = a_i % w_a;
    const int a_h = a_i / w_a % h_a;
    const int a_c = a_i / w_a / h_a;
    const int x_w = (w_s < w_k) ? a_w : (a_w / w_k * w_s + a_w % w_k);
    const int x_h = (h_s < h_k) ? a_h : (a_h / h_k * h_s + a_h % h_k);
    const int x_c = (c_s < c_k) ? a_c : (a_c / c_k * c_s + a_c % c_k);
    const int b_w = (x_w < w_k) ? 0 : ((x_w - w_k) / w_s + 1);
    const int b_h = (x_h < h_k) ? 0 : ((x_h - h_k) / h_s + 1);
    const int b_c = (x_c < c_k) ? 0 : ((x_c - c_k) / c_s + 1);
    const int e_w = min(x_w / w_s + 1, w_y);
    const int e_h = min(x_h / h_s + 1, h_y);
    const int e_c = min(x_c / c_s + 1, c_y);
    a[a_i] = b ? b[a_i] : 0;
    for (int y_n = 0; y_n < n; ++y_n) {
      const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
      for (int y_c = b_c; y_c < e_c; ++y_c) {
        for (int y_h = b_h; y_h < e_h; ++y_h) {
          for (int y_w = b_w; y_w < e_w; ++y_w) {
            const int y_i = ((y_n * c_y + y_c) * h_y + y_h) * w_y + y_w;
            a[a_i] += x[x_i] * y[y_i];
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void complete_gridding_cpu_aoxpb2y(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = (w_y - 1) * w_k + min(w_k, w_x - (w_y - 1) * w_s);
  const int h_a = (h_y - 1) * h_k + min(h_k, h_x - (h_y - 1) * h_s);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = y_w * w_s;
    const int b_h = y_h * h_s;
    const int b_c = y_c * c_s;
    const int e_w = min(b_w + w_k, w_x);
    const int e_h = min(b_h + h_k, h_x);
    const int e_c = min(b_c + c_k, c_x);
    const int b_i = y_i % (c_y * h_y * w_y);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int a_c = y_c * c_k + x_c - b_c;
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int a_h = y_h * h_k + x_h - b_h;
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int a_w = y_w * w_k + x_w - b_w;
          const int a_i = (a_c * h_a + a_h) * w_a + a_w;
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void complete_degridding_cpu_aoxpb2y(const int n,
    const int c_y, const int h_y, const int w_y,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_y - w_k) / w_s + 1;
  const int h_z = max(0, h_y - h_k) / h_s + 1;
  const int c_z = max(0, c_y - c_k) / c_s + 1;
  const int w_x = w_z + (max(0, w_y - w_k) % w_s && w_z * w_s < w_y);
  const int h_x = h_z + (max(0, h_y - h_k) % h_s && h_z * h_s < h_y);
  const int c_x = c_z + (max(0, c_y - c_k) % c_s && c_z * c_s < c_y);
  const int w_a = (w_x - 1) * w_k + min(w_k, w_y - (w_x - 1) * w_s);
  const int h_a = (h_x - 1) * h_k + min(h_k, h_y - (h_x - 1) * h_s);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w < w_k) ? 0 : ((y_w - w_k) / w_s + 1);
    const int b_h = (y_h < h_k) ? 0 : ((y_h - h_k) / h_s + 1);
    const int b_c = (y_c < c_k) ? 0 : ((y_c - c_k) / c_s + 1);
    const int e_w = min(y_w / w_s + 1, w_x);
    const int e_h = min(y_h / h_s + 1, h_x);
    const int e_c = min(y_c / c_s + 1, c_x);
    const int b_i = y_i % (c_y * h_y * w_y);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int a_c = y_c + x_c * (c_k - c_s);
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int a_h = y_h + x_h * (h_k - h_s);
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int a_w = y_w + x_w * (w_k - w_s);
          const int a_i = (a_c * h_a + a_h) * w_a + a_w;
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void complete_degridding_cpu_xoypb2a(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* x, const Dtype* y, const Dtype* b, Dtype* a) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = (w_y - 1) * w_k + min(w_k, w_x - (w_y - 1) * w_s);
  const int h_a = (h_y - 1) * h_k + min(h_k, h_x - (h_y - 1) * h_s);
  const int c_a = (c_y - 1) * c_k + min(c_k, c_x - (c_y - 1) * c_s);
  const int m_a = c_a * h_a * w_a;
  for (int a_i = 0; a_i < m_a; ++a_i) {
    const int a_w = a_i % w_a;
    const int a_h = a_i / w_a % h_a;
    const int a_c = a_i / w_a / h_a;
    const int y_w = a_w / w_k;
    const int y_h = a_h / h_k;
    const int y_c = a_c / c_k;
    const int x_w = y_w * w_s + a_w % w_k;
    const int x_h = y_h * h_s + a_h % h_k;
    const int x_c = y_c * c_s + a_c % c_k;
    a[a_i] = b ? b[a_i] : 0;
    for (int y_n = 0; y_n < n; ++y_n) {
      const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
      const int y_i = ((y_n * c_y + y_c) * h_y + y_h) * w_y + y_w;
      a[a_i] += x[x_i] * y[y_i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void shared_gridding_cpu_aoxpb2y(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = min(w_k, w_x);
  const int h_a = min(h_k, h_x);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = y_w * w_s;
    const int b_h = y_h * h_s;
    const int b_c = y_c * c_s;
    const int e_w = min(b_w + w_k, w_x);
    const int e_h = min(b_h + h_k, h_x);
    const int e_c = min(b_c + c_k, c_x);
    const int b_i = y_i % (y_c * y_h * y_w);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int a_c = x_c - b_c;
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int a_h = x_h - b_h;
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int a_w = x_w - b_w;
          const int a_i = (a_c * h_a + a_h) * w_a + a_w;
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void shared_degridding_cpu_aoxpb2y(const int n,
    const int c_y, const int h_y, const int w_y,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y) {
  const int w_z = max(0, w_y - w_k) / w_s + 1;
  const int h_z = max(0, h_y - h_k) / h_s + 1;
  const int c_z = max(0, c_y - c_k) / c_s + 1;
  const int w_x = w_z + (max(0, w_y - w_k) % w_s && w_z * w_s < w_y);
  const int h_x = h_z + (max(0, h_y - h_k) % h_s && h_z * h_s < h_y);
  const int c_x = c_z + (max(0, c_y - c_k) % c_s && c_z * c_s < c_y);
  const int w_a = min(w_k, w_y);
  const int h_a = min(h_k, h_y);
  const int m_y = n * c_y * h_y * w_y;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    const int y_w = y_i % w_y;
    const int y_h = y_i / w_y % h_y;
    const int y_c = y_i / w_y / h_y % c_y;
    const int y_n = y_i / w_y / h_y / c_y;
    const int b_w = (y_w < w_k) ? 0 : ((y_w - w_k) / w_s + 1);
    const int b_h = (y_h < h_k) ? 0 : ((y_h - h_k) / h_s + 1);
    const int b_c = (y_c < c_k) ? 0 : ((y_c - c_k) / c_s + 1);
    const int e_w = min(y_w / w_s + 1, w_x);
    const int e_h = min(y_h / h_s + 1, h_x);
    const int e_c = min(y_c / c_s + 1, c_x);
    const int b_i = y_i % (c_y * h_y * w_y);
    y[y_i] = b ? b[b_i] : 0;
    for (int x_c = b_c; x_c < e_c; ++x_c) {
      const int a_c = y_c - x_c * c_s;
      for (int x_h = b_h; x_h < e_h; ++x_h) {
        const int a_h = y_h - x_h * h_s;
        for (int x_w = b_w; x_w < e_w; ++x_w) {
          const int a_w = y_w - x_w * w_s;
          const int a_i = (a_c * h_a + a_h) * w_a + a_w;
          const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
          y[y_i] += a[a_i] * x[x_i];
        }
      }
    }
  }
}

template <typename Dtype>
void shared_degridding_cpu_xoypb2a(const int n,
    const int c_x, const int h_x, const int w_x,
    const int c_k, const int h_k, const int w_k,
    const int c_s, const int h_s, const int w_s,
    const Dtype* x, const Dtype* y, const Dtype* b, Dtype* a) {
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  const int w_a = min(w_k, w_x);
  const int h_a = min(h_k, h_x);
  const int c_a = min(c_k, c_x);
  const int m_a = c_a * h_a * w_a;
  for (int a_i = 0; a_i < m_a; ++a_i) {
    const int a_w = a_i % w_a;
    const int a_h = a_i / w_a % h_a;
    const int a_c = a_i / w_a / h_a;
    a[a_i] = b ? b[a_i] : 0;
    for (int y_c = 0, x_c = a_c; x_c < c_x; x_c += c_s, ++y_c) {
      for (int y_h = 0, x_h = a_h; x_h < h_x; x_h += h_s, ++y_h) {
        for (int y_w = 0, x_w = a_w; x_w < w_x; x_w += w_s, ++y_w) {
          for (int y_n = 0; y_n < n; ++y_n) {
            const int x_i = ((y_n * c_x + x_c) * h_x + x_h) * w_x + x_w;
            const int y_i = ((y_n * c_y + y_c) * h_y + y_h) * w_y + y_w;
            a[a_i] += x[x_i] * y[y_i];
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void caffe_cpu_multi_add(const int n,
    const int c_x, const int h_x, const int w_x, 
    const Dtype* x, const Dtype* b, Dtype* y) {
  const int m_y = c_x * h_x * w_x;
  for (int y_i = 0; y_i < m_y; ++y_i) {
    y[y_i] = b ? b[y_i] : 0;
    for (int x_n = 0; x_n < n; ++x_n) {
      const int x_i = x_n * m_y + y_i;
      y[y_i] += x[x_i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void GriddingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  GriddingParameter gridding_param = this->layer_param_.gridding_param();
  gridding_ = gridding_param.gridding();
  kernel_c_ = gridding_param.kernel_c();
  kernel_h_ = gridding_param.kernel_h();
  kernel_w_ = gridding_param.kernel_w();
  stride_c_ = gridding_param.stride_c();
  stride_h_ = gridding_param.stride_h();
  stride_w_ = gridding_param.stride_w();
  bias_term_ = gridding_param.bias_term();
  CHECK(gridding_ == "incomplete_gridding" || gridding_ == "complete_gridding" || gridding_ == "shared_gridding")
    << "Gridding implemented only for 'incomplete_gridding', 'complete_gridding' and 'shared_gridding'.";
  CHECK_GT(kernel_c_, 0) << "kernel_c cannot be zero.";
  CHECK_GT(kernel_h_, 0) << "kernel_h cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "kernel_w cannot be zero.";
  CHECK_GT(stride_c_, 0) << "stride_c cannot be zero.";
  CHECK_GT(stride_h_, 0) << "stride_h cannot be zero.";
  CHECK_GT(stride_w_, 0) << "stride_w cannot be zero.";
  const int w_k = kernel_w_;
  const int h_k = kernel_h_;
  const int c_k = kernel_c_;
  const int w_s = stride_w_;
  const int h_s = stride_h_;
  const int c_s = stride_c_;
  const int w_x = bottom[0]->shape(3);
  const int h_x = bottom[0]->shape(2);
  const int c_x = bottom[0]->shape(1);
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  int w_a = 0, h_a = 0, c_a = 0;
  if (gridding_ == "incomplete_gridding") {
    w_a = (w_y - 1) * min(w_k, w_s) + min(w_k, w_x - (w_y - 1) * w_s);
    h_a = (h_y - 1) * min(h_k, h_s) + min(h_k, h_x - (h_y - 1) * h_s);
    c_a = (c_y - 1) * min(c_k, c_s) + min(c_k, c_x - (c_y - 1) * c_s);
  }
  else if (gridding_ == "complete_gridding") {
    w_a = (w_y - 1) * w_k + min(w_k, w_x - (w_y - 1) * w_s);
    h_a = (h_y - 1) * h_k + min(h_k, h_x - (h_y - 1) * h_s);
    c_a = (c_y - 1) * c_k + min(c_k, c_x - (c_y - 1) * c_s);
  }
  else if (gridding_ == "shared_gridding") {
    w_a = min(w_k, w_x);
    h_a = min(h_k, h_x);
    c_a = min(c_k, c_x);
  }
  if (this->blobs_.size()) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size()) << "Incorrect number of weight blobs.";
    vector<int> weight_shape(3);
    weight_shape[0] = c_a;
    weight_shape[1] = h_a;
    weight_shape[2] = w_a;
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
        << weight_shaped_blob.shape_string() << "; instead, shape was "
        << this->blobs_[0]->shape_string();
    }
    if (bias_term_) {
      vector<int> bias_shape(3);
      bias_shape[0] = c_y;
      bias_shape[1] = h_y;
      bias_shape[2] = w_y;
      if (bias_shape != this->blobs_[1]->shape()) {
        Blob<Dtype> bias_shaped_blob(bias_shape);
        LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
      }
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(bias_term_ ? 2 : 1);
    vector<int> weight_shape(3);
    weight_shape[0] = c_a;
    weight_shape[1] = h_a;
    weight_shape[2] = w_a;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(gridding_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (bias_term_) {
      vector<int> bias_shape(3);
      bias_shape[0] = c_y;
      bias_shape[1] = h_y;
      bias_shape[2] = w_y;
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(gridding_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void GriddingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  const int w_k = kernel_w_;
  const int h_k = kernel_h_;
  const int c_k = kernel_c_;
  const int w_s = stride_w_;
  const int h_s = stride_h_;
  const int c_s = stride_c_;
  const int w_x = bottom[0]->shape(3);
  const int h_x = bottom[0]->shape(2);
  const int c_x = bottom[0]->shape(1);
  const int w_z = max(0, w_x - w_k) / w_s + 1;
  const int h_z = max(0, h_x - h_k) / h_s + 1;
  const int c_z = max(0, c_x - c_k) / c_s + 1;
  const int w_y = w_z + (max(0, w_x - w_k) % w_s && w_z * w_s < w_x);
  const int h_y = h_z + (max(0, h_x - h_k) % h_s && h_z * h_s < h_x);
  const int c_y = c_z + (max(0, c_x - c_k) % c_s && c_z * c_s < c_x);
  top[0]->Reshape(bottom[0]->shape(0), c_y, h_y, w_y);
}

template <typename Dtype>
void GriddingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const Dtype* bias = bias_term_ ? this->blobs_[1]->cpu_data() : 0;
  if (gridding_ == "incomplete_gridding") {
    incomplete_gridding_cpu_aoxpb2y<Dtype>(
      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->cpu_data(), bottom[0]->cpu_data(), bias, top[0]->mutable_cpu_data());
  }
  else if (gridding_ == "complete_gridding") {
    complete_gridding_cpu_aoxpb2y<Dtype>(
      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->cpu_data(), bottom[0]->cpu_data(), bias, top[0]->mutable_cpu_data());
  }
  else if (gridding_ == "shared_gridding") {
    shared_gridding_cpu_aoxpb2y<Dtype>(
      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->cpu_data(), bottom[0]->cpu_data(), bias, top[0]->mutable_cpu_data());
  }
}

template <typename Dtype>
void GriddingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  const Dtype* top_diff = top[0]->cpu_diff();
  if (bias_term_ && this->param_propagate_down_[1]) {
    caffe_cpu_multi_add<Dtype>(
      top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      top_diff, 0, this->blobs_[1]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[0]) {
    if (gridding_ == "incomplete_gridding") {
      incomplete_degridding_cpu_xoypb2a<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        bottom[0]->cpu_data(), top_diff, 0, this->blobs_[0]->mutable_cpu_diff());
    }
    else if (gridding_ == "complete_gridding") {
      complete_degridding_cpu_xoypb2a<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        bottom[0]->cpu_data(), top_diff, 0, this->blobs_[0]->mutable_cpu_diff());
    }
    else if (gridding_ == "shared_gridding") {
      shared_degridding_cpu_xoypb2a<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        bottom[0]->cpu_data(), top_diff, 0, this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (propagate_down[0]) {
    if (gridding_ == "incomplete_gridding") {
      incomplete_degridding_cpu_aoxpb2y<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->cpu_data(), top_diff, 0, bottom[0]->mutable_cpu_diff());
    }
    else if (gridding_ == "complete_gridding") {
      complete_degridding_cpu_aoxpb2y<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->cpu_data(), top_diff, 0, bottom[0]->mutable_cpu_diff());
    }
    else if (gridding_ == "shared_gridding") {
      shared_degridding_cpu_aoxpb2y<Dtype>(
        bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->cpu_data(), top_diff, 0, bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GriddingLayer);
#endif
INSTANTIATE_CLASS(GriddingLayer);
REGISTER_LAYER_CLASS(Gridding);
} // namespace caffe