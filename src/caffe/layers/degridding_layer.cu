#include <algorithm>
#include <vector>
#include "caffe/layers/degridding_layer.hpp"

namespace caffe {

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void incomplete_gridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void incomplete_degridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void incomplete_degridding_gpu_xoypb2a(const int n,
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
  CUDA_KERNEL_LOOP(a_i, m_a) {
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
__global__ void complete_gridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void complete_degridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void complete_degridding_gpu_xoypb2a(const int n,
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
  CUDA_KERNEL_LOOP(a_i, m_a) {
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
__global__ void shared_gridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void shared_degridding_gpu_aoxpb2y(const int n,
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
  CUDA_KERNEL_LOOP(y_i, m_y) {
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
__global__ void shared_degridding_gpu_xoypb2a(const int n,
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
  CUDA_KERNEL_LOOP(a_i, m_a) {
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
__global__ void caffe_gpu_multi_add(const int n,
    const int c_x, const int h_x, const int w_x, 
    const Dtype* x, const Dtype* b, Dtype* y) {
  const int m_y = c_x * h_x * w_x;
  CUDA_KERNEL_LOOP(y_i, m_y) {
    y[y_i] = b ? b[y_i] : 0;
    for (int x_n = 0; x_n < n; ++x_n) {
      const int x_i = x_n * m_y + y_i;
      y[y_i] += x[x_i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void DegriddingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  const Dtype* bias = bias_term_ ? this->blobs_[1]->gpu_data() : 0;
  if (degridding_ == "incomplete_degridding") {
    incomplete_degridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->gpu_data(), bottom[0]->gpu_data(), bias, top[0]->mutable_gpu_data());
  }
  else if (degridding_ == "complete_degridding") {
    complete_degridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->gpu_data(), bottom[0]->gpu_data(), bias, top[0]->mutable_gpu_data());
  }
  else if (degridding_ == "shared_degridding") {
    shared_degridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
      this->blobs_[0]->gpu_data(), bottom[0]->gpu_data(), bias, top[0]->mutable_gpu_data());
  }
}

template <typename Dtype>
void DegriddingLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (bias_term_ && this->param_propagate_down_[1]) {
    caffe_gpu_multi_add<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
      top_diff, 0, this->blobs_[1]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[0]) {
    if (degridding_ == "incomplete_degridding") {
      incomplete_degridding_gpu_xoypb2a<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        top_diff, bottom[0]->gpu_data(), 0, this->blobs_[0]->mutable_gpu_diff());
    }
    else if (degridding_ == "complete_degridding") {
      complete_degridding_gpu_xoypb2a<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        top_diff, bottom[0]->gpu_data(), 0, this->blobs_[0]->mutable_gpu_diff());
    }
    else if (degridding_ == "shared_degridding") {
      shared_degridding_gpu_xoypb2a<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        top_diff, bottom[0]->gpu_data(), 0, this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (propagate_down[0]) {
    if (degridding_ == "incomplete_degridding") {
      incomplete_gridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->gpu_data(), top_diff, 0, bottom[0]->mutable_gpu_diff());
    }
    else if (degridding_ == "complete_degridding") {
      complete_gridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->gpu_data(), top_diff, 0, bottom[0]->mutable_gpu_diff());
    }
    else if (degridding_ == "shared_degridding") {
      shared_gridding_gpu_aoxpb2y<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3),
        kernel_c_, kernel_h_, kernel_w_, stride_c_, stride_h_, stride_w_,
        this->blobs_[0]->gpu_data(), top_diff, 0, bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DegriddingLayer);
} // namespace caffe