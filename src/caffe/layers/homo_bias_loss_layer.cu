#include <curand_kernel.h>

#include "caffe/layers/homo_bias_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ClusterForwardForBias_gpu_backend(
    const int outer_numb_,     const int inner_numb_,
    const Dtype* middle_datum, const Dtype* middle_diffs,
          Dtype* clustr_datum,       Dtype* clustr_diffs) {
  CUDA_KERNEL_LOOP(inner_index, inner_numb_) {
          Dtype* clustr_datpt = clustr_datum + inner_index;
          Dtype* clustr_difpt = clustr_diffs + inner_index;
    const Dtype* middle_datpt = middle_datum + inner_index;
    const Dtype* middle_difpt = middle_diffs + inner_index;
          Dtype  reduce_datum = 0, reduce_count = 0;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
      if (static_cast<int>(*middle_difpt) > 0) {
        reduce_datum += *middle_datpt / *middle_difpt;
        reduce_count += 1;
      }
      middle_datpt += inner_numb_;
      middle_difpt += inner_numb_;
    }
    if (static_cast<int>(reduce_count) > 0) {
       reduce_count += *clustr_difpt;
      *clustr_datpt *= *clustr_difpt / reduce_count;
      *clustr_datpt +=  reduce_datum / reduce_count;
      *clustr_difpt  =  reduce_count;
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusterForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[2]]->mutable_cpu_data();
  Dtype* clustr_datum = clustr_blob_.mutable_gpu_data();
  Dtype* clustr_diffs = clustr_blob_.mutable_gpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  ClusterForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_, inner_numb_, middle_datum, middle_diffs, clustr_datum, clustr_diffs);
  caffe_gpu_asum(inner_numb_, clustr_datum, topper_datum);
  *topper_datum /= inner_numb_;
}
template void HomoBiasLossLayer<float>::ClusterForward_gpu(const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::ClusterForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ScatterForwardForBias_gpu_backend(
    const int    outer_numb_,  const int inner_numb_,
    const Dtype* middle_datum, const Dtype* middle_diffs,
          Dtype* scattr_datum,       Dtype* scattr_diffs) {
  CUDA_KERNEL_LOOP(inner_index, inner_numb_) {
          Dtype* scattr_datpt = scattr_datum + inner_index;
          Dtype* scattr_difpt = scattr_diffs + inner_index;
    const Dtype* middle_datpt = middle_datum + inner_index;
    const Dtype* middle_difpt = middle_diffs + inner_index;
          Dtype  reduce_datum = 0, reduce_count = 0;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
      if (static_cast<int>(*middle_difpt) > 0) {
        reduce_datum += *middle_datpt / *middle_difpt;
        reduce_count += 1;
      }
      middle_datpt += inner_numb_;
      middle_difpt += inner_numb_;
    }
    if (static_cast<int>(reduce_count) > 0) {
       reduce_count += *scattr_difpt;
      *scattr_datpt *= *scattr_difpt / reduce_count;
      *scattr_datpt +=  reduce_datum / reduce_count;
      *scattr_difpt  =  reduce_count;
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatterForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[3]]->mutable_cpu_data();
  Dtype* scattr_datum = scattr_blob_.mutable_gpu_data();
  Dtype* scattr_diffs = scattr_blob_.mutable_gpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  ScatterForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_, inner_numb_, middle_datum, middle_diffs, scattr_datum, scattr_diffs);
  caffe_gpu_asum(inner_numb_, scattr_datum, topper_datum);
  *topper_datum /= inner_numb_;
}
template void HomoBiasLossLayer<float>::ScatterForward_gpu(const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::ScatterForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ClusupdForwardForBias_gpu_backend(
    const int label_numb_,     const int label_nmax_,     const int inner_numb_,
    const Dtype* medium_datum, const Dtype* medium_diffs, Dtype* clusup_datum,
          Dtype* clusup_diffs,       Dtype* topper_datum) {
  const int round_count = label_numb_ * label_nmax_;
  CUDA_KERNEL_LOOP(round_index, round_count) {
          Dtype* topper_datpt = topper_datum + round_index;
          Dtype* clusup_datpt = clusup_datum + round_index * inner_numb_;
          Dtype* clusup_difpt = clusup_diffs + round_index * inner_numb_;
    const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
    const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
                *topper_datpt = 0;
    for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
      if (static_cast<int>(*medium_difpt) > 0) {
        Dtype reduce_count = *clusup_difpt + *medium_difpt;
        *clusup_datpt *= *clusup_difpt / reduce_count;
        *clusup_datpt += *medium_datpt / reduce_count;
        *clusup_difpt  =  reduce_count;
        *topper_datpt += *clusup_datpt;
      }
      ++medium_datpt; ++medium_difpt;
      ++clusup_datpt; ++clusup_difpt;
    }
    *topper_datpt /= inner_numb_;
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusupdForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[4]]->mutable_gpu_data();
  Dtype* clusup_datum = clusup_blob_.mutable_gpu_data();
  Dtype* clusup_diffs = clusup_blob_.mutable_gpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  ClusupdForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    label_numb_,  label_nmax_,  inner_numb_,
    medium_datum, medium_diffs, clusup_datum,
    clusup_diffs, topper_datum
  );
}
template void HomoBiasLossLayer<float>::ClusupdForward_gpu(const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::ClusupdForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ScatupdForwardForBias_gpu_backend(
    const int label_numb_,     const int label_nmax_,     const int inner_numb_,
    const Dtype* medium_datum, const Dtype* medium_diffs, Dtype* scatup_datum,
          Dtype* scatup_diffs,       Dtype* topper_datum) {
  const int round_count = label_numb_ * label_nmax_;
  CUDA_KERNEL_LOOP(round_index, round_count) {
          Dtype* topper_datpt = topper_datum + round_index;
          Dtype* scatup_datpt = scatup_datum + round_index * inner_numb_;
          Dtype* scatup_difpt = scatup_diffs + round_index * inner_numb_;
    const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
    const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
                *topper_datpt = 0;
    for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
      if (static_cast<int>(*medium_difpt) > 0) {
        Dtype reduce_count = *scatup_difpt + *medium_difpt;
        *scatup_datpt *= *scatup_difpt / reduce_count;
        *scatup_datpt += *medium_datpt / reduce_count;
        *scatup_difpt  =  reduce_count;
        *topper_datpt += *scatup_datpt;
      }
      ++medium_datpt; ++medium_difpt;
      ++scatup_datpt; ++scatup_difpt;
    }
    *topper_datpt /= inner_numb_;
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatupdForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[5]]->mutable_gpu_data();
  Dtype* scatup_datum = scatup_blob_.mutable_gpu_data();
  Dtype* scatup_diffs = scatup_blob_.mutable_gpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  ScatupdForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    label_numb_,  label_nmax_,  inner_numb_,
    medium_datum, medium_diffs, scatup_datum,
    scatup_diffs, topper_datum
  );
}
template void HomoBiasLossLayer<float>::ScatupdForward_gpu(const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::ScatupdForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void OvalizeForwardForBias_gpu_backend(
    const int match_numb_,     const int label_numb_,
    const Dtype* caches_datum, const Dtype* caches_diffs,
          Dtype* ovaliz_datum,       Dtype* ovaliz_diffs) {
  CUDA_KERNEL_LOOP(label_index, label_numb_) {
          Dtype* ovaliz_datpt = ovaliz_datum + label_index;
          Dtype* ovaliz_difpt = ovaliz_diffs + label_index;
    const Dtype* caches_datpt = caches_datum + label_index;
    const Dtype* caches_difpt = caches_diffs + label_index;
          Dtype  reduce_datum = 0, reduce_count = 0;
    for (int match_index = 0; match_index < match_numb_; ++match_index) {
      if (static_cast<int>(*caches_difpt) > 0) {
        reduce_datum += *caches_datpt / *caches_difpt;
        reduce_count += 1;
      }
      caches_datpt += label_numb_;
      caches_difpt += label_numb_;
    }
    if (static_cast<int>(reduce_count) > 0) {
       reduce_count += *ovaliz_difpt;
      *ovaliz_datpt *= *ovaliz_difpt / reduce_count;
      *ovaliz_datpt +=  reduce_datum / reduce_count;
      *ovaliz_difpt  =  reduce_count;
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[6]]->mutable_cpu_data();
  Dtype* ovaliz_datum = ovaliz_blob_.mutable_gpu_data();
  Dtype* ovaliz_diffs = ovaliz_blob_.mutable_gpu_diff();
  const Dtype* caches_datum = caches_blob_.mutable_gpu_data();
  const Dtype* caches_diffs = caches_blob_.mutable_gpu_diff();
  OvalizeForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_, label_numb_, caches_datum, caches_diffs, ovaliz_datum, ovaliz_diffs);
  caffe_gpu_asum(label_numb_, ovaliz_datum, topper_datum);
  *topper_datum /= label_numb_;
}
template void HomoBiasLossLayer<float>::OvalizeForward_gpu(const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::OvalizeForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void OvalizeMatcherForBias_gpu_backend(
    const int ovs2s_count,   const int ovt2t_count,
    const int ovs2t_count,   const int ovall_count,
    const int outer_numb_,   const int match_numb_,
    const int label_numb_,   const int ignore_label_,
    const int target_label_, const Dtype* bottom_label,
    const Dtype ovals2s_01stprop_, const Dtype ovals2s_02ndprop_,
    const Dtype ovalt2t_01stprop_, const Dtype ovalt2t_02ndprop_,
    const Dtype ovals2t_01stprop_, const Dtype ovals2t_02ndprop_,
    int*   mapidx_datum, int*   mapidx_diffs,
    int*   mapair_datum, int*   mapair_diffs,
    Dtype* maprop_datum, Dtype* maprop_diffs) {
  CUDA_KERNEL_LOOP(label_index, label_numb_) {
    int  mapidx_datnc = 0, mapidx_difnc = 0;
    int* mapidx_datpt = mapidx_datum + label_index;
    int* mapidx_difpt = mapidx_diffs + label_index;
    const Dtype* bottom_labpt = bottom_label + label_index;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
      if (static_cast<int>(*bottom_labpt) == target_label_) {
        *mapidx_difpt  = outer_index;
         mapidx_difpt += label_numb_;
         mapidx_difnc += 1;
      } else if (static_cast<int>(*bottom_labpt) != ignore_label_) {
        *mapidx_datpt  = outer_index;
         mapidx_datpt += label_numb_;
         mapidx_datnc += 1;
      }
      bottom_labpt += label_numb_;
    }
    int*   mapair_datpt = mapair_datum + label_index;
    int*   mapair_difpt = mapair_diffs + label_index;
    Dtype* maprop_datpt = maprop_datum + label_index;
    Dtype* maprop_difpt = maprop_diffs + label_index;
    if (ovall_count == 0 || mapidx_datnc < 2 || mapidx_difnc < 2) {
      for (int match_index = 0; match_index < match_numb_; ++match_index) {
        *mapair_datpt = *mapair_difpt = -1;
        *maprop_datpt = *maprop_difpt = +0;
        mapair_datpt += label_numb_;
        mapair_difpt += label_numb_;
        maprop_datpt += label_numb_;
        maprop_difpt += label_numb_;
      }
    } else {
      // shuffling mapidx_datum and mapidx_diffs, then filling mapair_datum
      curandState mapdat_state;
      if (ovall_count) curand_init(static_cast<unsigned>(*mapair_datpt), 0, 0, &mapdat_state);
      if (ovs2s_count || ovs2t_count) {
        for (int mapidx_datid = mapidx_datnc - 1; mapidx_datid > 0; --mapidx_datid) {
          int* mapidx_srcpt = mapidx_datpt - label_numb_ * (mapidx_datnc - mapidx_datid);
          int* mapidx_trgpt = mapidx_datpt - label_numb_ * (mapidx_datnc - curand(&mapdat_state) % (mapidx_datid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      if (ovt2t_count) {
        for (int mapidx_difid = mapidx_difnc - 1; mapidx_difid > 0; --mapidx_difid) {
          int* mapidx_srcpt = mapidx_difpt - label_numb_ * (mapidx_difnc - mapidx_difid);
          int* mapidx_trgpt = mapidx_difpt - label_numb_ * (mapidx_difnc - curand(&mapdat_state) % (mapidx_difid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      for (int mapair_datid = 0; mapair_datid < ovs2s_count; ++mapair_datid) {
        *maprop_datpt = ovals2s_01stprop_;
        *mapair_datpt = *(mapidx_datpt - label_numb_ * (mapidx_datnc - mapair_datid));
         maprop_datpt += label_numb_;
         mapair_datpt += label_numb_;
      }
      for (int mapair_datid = 0; mapair_datid < ovt2t_count; ++mapair_datid) {
        *maprop_datpt = ovalt2t_01stprop_;
        *mapair_datpt = *(mapidx_difpt - label_numb_ * (mapidx_difnc - mapair_datid));
         maprop_datpt += label_numb_;
         mapair_datpt += label_numb_;
      }
      for (int mapair_datid = 0; mapair_datid < ovs2t_count; ++mapair_datid) {
        *maprop_datpt = ovals2t_01stprop_;
        *mapair_datpt = *(mapidx_datpt - label_numb_ * (mapidx_datnc - mapair_datid));
         maprop_datpt += label_numb_;
         mapair_datpt += label_numb_;
      }
      for (int mapair_datid = 0; mapair_datid < ovs2t_count; ++mapair_datid) {
        *maprop_datpt = ovals2t_02ndprop_;
        *mapair_datpt = *(mapidx_difpt - label_numb_ * (mapidx_difnc - mapair_datid));
         maprop_datpt += label_numb_;
         mapair_datpt += label_numb_;
      }
      // shuffling mapidx_datum and mapidx_diffs, then filling mapair_diffs
      curandState mapdif_state;
      if (ovall_count) curand_init(static_cast<unsigned>(*mapair_difpt), 0, 0, &mapdif_state);
      if (ovs2s_count) {
        for (int mapidx_datid = mapidx_datnc - 1; mapidx_datid > 0; --mapidx_datid) {
          int* mapidx_srcpt = mapidx_datpt - label_numb_ * (mapidx_datnc - mapidx_datid);
          int* mapidx_trgpt = mapidx_datpt - label_numb_ * (mapidx_datnc - curand(&mapdif_state) % (mapidx_datid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      if (ovt2t_count || ovs2t_count) {
        for (int mapidx_difid = mapidx_difnc - 1; mapidx_difid > 0; --mapidx_difid) {
          int* mapidx_srcpt = mapidx_difpt - label_numb_ * (mapidx_difnc - mapidx_difid);
          int* mapidx_trgpt = mapidx_difpt - label_numb_ * (mapidx_difnc - curand(&mapdif_state)  % (mapidx_difid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      for (int mapair_difid = 0; mapair_difid < ovs2s_count; ++mapair_difid) {
        *maprop_difpt = ovals2s_02ndprop_;
        *mapair_difpt = *(mapidx_datpt - label_numb_ * (mapidx_datnc - mapair_difid));
         maprop_difpt += label_numb_;
         mapair_difpt += label_numb_;
      }
      for (int mapair_difid = 0; mapair_difid < ovt2t_count; ++mapair_difid) {
        *maprop_difpt = ovalt2t_02ndprop_;
        *mapair_difpt = *(mapidx_difpt - label_numb_ * (mapidx_difnc - mapair_difid));
         maprop_difpt += label_numb_;
         mapair_difpt += label_numb_;
      }
      for (int mapair_difid = 0; mapair_difid < ovs2t_count; ++mapair_difid) {
        *maprop_difpt = ovals2t_02ndprop_;
        *mapair_difpt = *(mapidx_difpt - label_numb_ * (mapidx_difnc - mapair_difid));
         maprop_difpt += label_numb_;
         mapair_difpt += label_numb_;
      }
      for (int mapair_difid = 0; mapair_difid < ovs2t_count; ++mapair_difid) {
        *maprop_difpt = ovals2t_01stprop_;
        *mapair_difpt = *(mapidx_datpt - label_numb_ * (mapidx_datnc - mapair_difid));
         maprop_difpt += label_numb_;
         mapair_difpt += label_numb_;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeMatcher_gpu(const vector<Blob<Dtype>*>& bottom) {
  const bool ovs2s_check = (ovals2s_01stprop_ != Dtype(0) || ovals2s_02ndprop_ != Dtype(0));
  const bool ovt2t_check = (ovalt2t_01stprop_ != Dtype(0) || ovalt2t_02ndprop_ != Dtype(0));
  const bool ovs2t_check = (ovals2t_01stprop_ != Dtype(0) || ovals2t_02ndprop_ != Dtype(0));
  const int otask_count = ovs2s_check + ovt2t_check + ovs2t_check * 2;
  const int ovs2s_count = ovs2s_check ? (match_numb_ / otask_count) : 0;
  const int ovt2t_count = ovt2t_check ? (match_numb_ / otask_count) : 0;
  const int ovs2t_count = ovs2t_check ? (match_numb_ / otask_count) : 0;
  const int ovall_count = ovs2s_count + ovt2t_count + ovs2t_count * 2;
  match_numb_ = ovall_count ? ovall_count : match_numb_;
  vector<int> mapidx_shape(2);
  vector<int> mapair_shape(2);
  mapidx_shape[0] = outer_numb_;
  mapidx_shape[1] = label_numb_;
  mapair_shape[0] = match_numb_;
  mapair_shape[1] = label_numb_;
  mapidx_blob_.Reshape(mapidx_shape);
  mapair_blob_.Reshape(mapair_shape);
  maprop_blob_.Reshape(mapair_shape);
  int* mapidx_datum = mapidx_blob_.mutable_gpu_data();
  int* mapidx_diffs = mapidx_blob_.mutable_gpu_diff();
  int* mapair_datum = mapair_blob_.mutable_gpu_data();
  int* mapair_diffs = mapair_blob_.mutable_gpu_diff();
  Dtype* maprop_datum = maprop_blob_.mutable_gpu_data();
  Dtype* maprop_diffs = maprop_blob_.mutable_gpu_diff();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  for (int label_index = 0; label_index < label_numb_; ++label_index) {
    caffe_gpu_set(1, rand(), mapair_datum + label_index);
    caffe_gpu_set(1, rand(), mapair_diffs + label_index);
  }
  OvalizeMatcherForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    ovs2s_count,   ovt2t_count,
    ovs2t_count,   ovall_count,
    outer_numb_,   match_numb_,
    label_numb_,   ignore_label_,
    target_label_, bottom_label,
    ovals2s_01stprop_, ovals2s_02ndprop_,
    ovalt2t_01stprop_, ovalt2t_02ndprop_,
    ovals2t_01stprop_, ovals2t_02ndprop_,
    mapidx_datum,  mapidx_diffs,
    mapair_datum,  mapair_diffs,
    maprop_datum,  maprop_diffs
  );
}
template void HomoBiasLossLayer<float>::OvalizeMatcher_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OvalizeMatcher_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusterMeasureForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int cluster_measure_, const Dtype* numidx_datum,
    const Dtype* biases_datum,  const Dtype* bottom_datum,
    const Dtype* bottom_label,        Dtype* middle_datum,
          Dtype* middle_diffs) {
  if (cluster_measure_ == 0) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += buffer_datum * buffer_datum;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 1) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += log(buffer_datum * buffer_datum + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 2) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 3) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += abs(buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 4) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += log(abs(buffer_datum) + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 5) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += 1 - exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int measure = 0;
       if (cluster_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (cluster_measure_ == "logsubsqr-sample-biases") measure = 1;
  else if (cluster_measure_ == "expsubsqr-sample-biases") measure = 2;
  else if (cluster_measure_ == "rawsubabs-sample-biases") measure = 3;
  else if (cluster_measure_ == "logsubabs-sample-biases") measure = 4;
  else if (cluster_measure_ == "expsubabs-sample-biases") measure = 5;
  ClusterMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    biases_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoBiasLossLayer<float>::ClusterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ClusterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterMeasureForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int scatter_measure_, const Dtype* numidx_datum,
    const Dtype* biases_datum,  const Dtype* bottom_datum,
    const Dtype* bottom_label,        Dtype* middle_datum,
          Dtype* middle_diffs) {
  if (scatter_measure_ == 0) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += buffer_datum * buffer_datum;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 1) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += buffer_datum * buffer_datum;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 2) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += log(buffer_datum * buffer_datum + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 3) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += log(buffer_datum * buffer_datum + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 4) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += 1 - exp(-buffer_datum * buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 5) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += 1 - exp(-buffer_datum * buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 6) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += abs(buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 7) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += abs(buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 8) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += log(abs(buffer_datum) + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 9) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += log(abs(buffer_datum) + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 10) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *biases_datpt;
          *middle_datpt += 1 - exp(-abs(buffer_datum));
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_measure_ == 11) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* biases_datit = biases_datum + biases_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datit - *biases_datpt;
          *middle_datpt += 1 - exp(-abs(buffer_datum));
          *middle_difpt += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int measure = 0;
       if (scatter_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (scatter_measure_ == "rawsubsqr-biases-biases") measure = 1;
  else if (scatter_measure_ == "logsubsqr-sample-biases") measure = 2;
  else if (scatter_measure_ == "logsubsqr-biases-biases") measure = 3;
  else if (scatter_measure_ == "expsubsqr-sample-biases") measure = 4;
  else if (scatter_measure_ == "expsubsqr-biases-biases") measure = 5;
  else if (scatter_measure_ == "rawsubabs-sample-biases") measure = 6;
  else if (scatter_measure_ == "rawsubabs-biases-biases") measure = 7;
  else if (scatter_measure_ == "logsubabs-sample-biases") measure = 8;
  else if (scatter_measure_ == "logsubabs-biases-biases") measure = 9;
  else if (scatter_measure_ == "expsubabs-sample-biases") measure = 10;
  else if (scatter_measure_ == "expsubabs-biases-biases") measure = 11;
  ScatterMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    biases_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoBiasLossLayer<float>::ScatterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ScatterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusupdMeasureForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int clusupd_measure_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* biases_datum,        Dtype* medium_datum,
          Dtype* medium_diffs) {
  if (clusupd_measure_ == 0) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += buffer_datum * buffer_datum;
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 1) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += log(buffer_datum * buffer_datum + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 2) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 3) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += abs(buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 4) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += log(abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 5) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int measure = 0;
       if (clusupd_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (clusupd_measure_ == "logsubsqr-sample-biases") measure = 1;
  else if (clusupd_measure_ == "expsubsqr-sample-biases") measure = 2;
  else if (clusupd_measure_ == "rawsubabs-sample-biases") measure = 3;
  else if (clusupd_measure_ == "logsubabs-sample-biases") measure = 4;
  else if (clusupd_measure_ == "expsubabs-sample-biases") measure = 5;
  ClusupdMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      bottom_datum,
    bottom_label, numidx_datum,
    biases_datum, medium_datum,
    medium_diffs
  );
}
template void HomoBiasLossLayer<float>::ClusupdMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ClusupdMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatupdMeasureForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int scatupd_measure_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* biases_datum,        Dtype* medium_datum,
          Dtype* medium_diffs) {
  if (scatupd_measure_ == 0) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += buffer_datum * buffer_datum;
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 1) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += buffer_datum * buffer_datum;
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 2) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += log(buffer_datum * buffer_datum + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 3) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += log(buffer_datum * buffer_datum + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 4) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 5) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 6) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += abs(buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 7) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += abs(buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 8) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += log(abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 9) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += log(abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 10) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_measure_ == 11) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* biases_datit = biases_datum + biases_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *biases_datit - *biases_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int measure = 0;
       if (scatupd_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (scatupd_measure_ == "rawsubsqr-biases-biases") measure = 1;
  else if (scatupd_measure_ == "logsubsqr-sample-biases") measure = 2;
  else if (scatupd_measure_ == "logsubsqr-biases-biases") measure = 3;
  else if (scatupd_measure_ == "expsubsqr-sample-biases") measure = 4;
  else if (scatupd_measure_ == "expsubsqr-biases-biases") measure = 5;
  else if (scatupd_measure_ == "rawsubabs-sample-biases") measure = 6;
  else if (scatupd_measure_ == "rawsubabs-biases-biases") measure = 7;
  else if (scatupd_measure_ == "logsubabs-sample-biases") measure = 8;
  else if (scatupd_measure_ == "logsubabs-biases-biases") measure = 9;
  else if (scatupd_measure_ == "expsubabs-sample-biases") measure = 10;
  else if (scatupd_measure_ == "expsubabs-biases-biases") measure = 11;
  ScatupdMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      bottom_datum,
    bottom_label, numidx_datum,
    biases_datum, medium_datum,
    medium_diffs
  );
}
template void HomoBiasLossLayer<float>::ScatupdMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ScatupdMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterMeasureForBias_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int  label_nmax_,
    const int  odotter_measure_, const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* bottom_datum,
    const Dtype* numidx_datum,   const Dtype* biases_datum,
          Dtype* medial_datum,         Dtype* medial_diffs) {
  if (odotter_measure_ == 0) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += srcbuf_datum * srcbuf_datum;
        *medial_difpt += trgbuf_datum * trgbuf_datum;
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
  else if (odotter_measure_ == 1) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += log(srcbuf_datum * srcbuf_datum + 1);
        *medial_difpt += log(trgbuf_datum * trgbuf_datum + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
  else if (odotter_measure_ == 2) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += 1 - exp(-srcbuf_datum * srcbuf_datum);
        *medial_difpt += 1 - exp(-trgbuf_datum * trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
  else if (odotter_measure_ == 3) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += abs(srcbuf_datum);
        *medial_difpt += abs(trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
  else if (odotter_measure_ == 4) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += log(abs(srcbuf_datum) + 1);
        *medial_difpt += log(abs(trgbuf_datum) + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
  else if (odotter_measure_ == 5) {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int labmx_index = round_index % label_nmax_;
      const int label_index = round_index / label_nmax_ % label_numb_;
      const int match_index = round_index / label_nmax_ / label_numb_;
      const int  mapair_shift = match_index * label_numb_ + label_index;
      const int  numidx_shift = label_index * label_nmax_ + labmx_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const Dtype* numidx_datpt = numidx_datum +  numidx_shift;
      const Dtype* biases_datpt = biases_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *medial_datpt += 1 - exp(-abs(srcbuf_datum));
        *medial_difpt += 1 - exp(-abs(trgbuf_datum));
        ++srcbot_datpt; ++trgbot_datpt;
        ++biases_datpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OdotterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> medial_shape(3);
  medial_shape[0] = match_numb_;
  medial_shape[1] = label_numb_;
  medial_shape[2] = label_nmax_;
  medial_blob_.Reshape(medial_shape);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* medial_datum = medial_blob_.mutable_gpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_gpu_diff();
  int measure = 0;
       if (odotter_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (odotter_measure_ == "logsubsqr-sample-biases") measure = 1;
  else if (odotter_measure_ == "expsubsqr-sample-biases") measure = 2;
  else if (odotter_measure_ == "rawsubabs-sample-biases") measure = 3;
  else if (odotter_measure_ == "logsubabs-sample-biases") measure = 4;
  else if (odotter_measure_ == "expsubabs-sample-biases") measure = 5;
  OdotterMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      mapair_datum,
    mapair_diffs, bottom_datum,
    numidx_datum, biases_datum,
    medial_datum, medial_diffs
  );
}
template void HomoBiasLossLayer<float>::OdotterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OdotterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeMeasureForBias_gpu_backend(
    const int match_numb_,     const int label_numb_,
    const int label_nmax_,     const int ovalize_measure_,
    const Dtype* medial_datum, const Dtype* medial_diffs,
          Dtype* caches_datum,       Dtype* caches_diffs) {
  if (ovalize_measure_ == 0) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = *caches_datpt * *caches_datpt;
    }
  }
  else if (ovalize_measure_ == 1) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = *caches_datpt * *caches_datpt;
    }
  }
  else if (ovalize_measure_ == 2) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = log(*caches_datpt * *caches_datpt + 1);
    }
  }
  else if (ovalize_measure_ == 3) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = log(*caches_datpt * *caches_datpt + 1);
    }
  }
  else if (ovalize_measure_ == 4) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = 1 - exp(-*caches_datpt * *caches_datpt);
    }
  }
  else if (ovalize_measure_ == 5) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = 1 - exp(-*caches_datpt * *caches_datpt);
    }
  }
  else if (ovalize_measure_ == 6) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = abs(*caches_datpt);
    }
  }
  else if (ovalize_measure_ == 7) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = abs(*caches_datpt);
    }
  }
  else if (ovalize_measure_ == 8) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = log(abs(*caches_datpt) + 1);
    }
  }
  else if (ovalize_measure_ == 9) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = log(abs(*caches_datpt) + 1);
    }
  }
  else if (ovalize_measure_ == 10) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += *medial_datpt - *medial_difpt;
        *caches_difpt += 1;
      }
      *caches_datpt = 1 - exp(-abs(*caches_datpt));
    }
  }
  else if (ovalize_measure_ == 11) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype* caches_datpt = caches_datum + round_index;
      Dtype* caches_difpt = caches_diffs + round_index;
            *caches_datpt = *caches_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        *caches_datpt += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        *caches_difpt += 1;
      }
      *caches_datpt = 1 - exp(-abs(*caches_datpt));
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> caches_shape(2);
  caches_shape[0] = match_numb_;
  caches_shape[1] = label_numb_;
  caches_blob_.Reshape(caches_shape);
  const Dtype* medial_datum = medial_blob_.gpu_data();
  const Dtype* medial_diffs = medial_blob_.gpu_diff();
  Dtype* caches_datum = caches_blob_.mutable_gpu_data();
  Dtype* caches_diffs = caches_blob_.mutable_gpu_diff();
  int measure = 0;
       if (ovalize_measure_ == "rawsubsqr-origin-origin") measure = 0;
  else if (ovalize_measure_ == "rawsubsqr-sqroot-sqroot") measure = 1;
  else if (ovalize_measure_ == "logsubsqr-origin-origin") measure = 2;
  else if (ovalize_measure_ == "logsubsqr-sqroot-sqroot") measure = 3;
  else if (ovalize_measure_ == "expsubsqr-origin-origin") measure = 4;
  else if (ovalize_measure_ == "expsubsqr-sqroot-sqroot") measure = 5;
  else if (ovalize_measure_ == "rawsubabs-origin-origin") measure = 6;
  else if (ovalize_measure_ == "rawsubabs-sqroot-sqroot") measure = 7;
  else if (ovalize_measure_ == "logsubabs-origin-origin") measure = 8;
  else if (ovalize_measure_ == "logsubabs-sqroot-sqroot") measure = 9;
  else if (ovalize_measure_ == "expsubabs-origin-origin") measure = 10;
  else if (ovalize_measure_ == "expsubabs-sqroot-sqroot") measure = 11;
  OvalizeMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  label_numb_,
    label_nmax_,  measure,
    medial_datum, medial_diffs,
    caches_datum, caches_diffs
  );
}
template void HomoBiasLossLayer<float>::OvalizeMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OvalizeMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusterRegularForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int cluster_regular_, const Dtype* numidx_datum,
    const Dtype* biases_datum,  const Dtype* bottom_datum,
    const Dtype* bottom_label,        Dtype* middle_datum,
          Dtype* middle_diffs) {
  if (cluster_regular_ == 0) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += 2 * buffer_datum;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 1) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 2) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *middle_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 3) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *middle_datpt += buffer_dsign;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 4) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *middle_datpt += buffer_dsign / (abs(buffer_datum) + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 5) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_shift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *middle_datpt += buffer_dsign * exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int regular = 0;
       if (cluster_regular_ == "rawsubsqr-sample-biases") regular = 0;
  else if (cluster_regular_ == "logsubsqr-sample-biases") regular = 1;
  else if (cluster_regular_ == "expsubsqr-sample-biases") regular = 2;
  else if (cluster_regular_ == "rawsubabs-sample-biases") regular = 3;
  else if (cluster_regular_ == "logsubabs-sample-biases") regular = 4;
  else if (cluster_regular_ == "expsubabs-sample-biases") regular = 5;
  ClusterRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      numidx_datum,
    biases_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoBiasLossLayer<float>::ClusterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ClusterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterRegularForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int scatter_regular_, const Dtype* numidx_datum,
    const Dtype* biases_datum,  const Dtype* bottom_datum,
    const Dtype* bottom_label,        Dtype* middle_datum,
          Dtype* middle_diffs) {
  if (scatter_regular_ == 0) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 1) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 2) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 3) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 4) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign / (abs(buffer_datum) + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 5) {
    const int round_count = outer_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int outer_index = round_index / inner_numb_;
            Dtype* middle_datpt = middle_datum + round_index; //middle datum pointer
            Dtype* middle_difpt = middle_diffs + round_index; //middle diffs pointer
      const Dtype* bottom_datpt = bottom_datum + round_index; //bottom datum pointer
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int biases_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* biases_datpt = biases_datum + biases_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *biases_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign * exp(-abs(buffer_datum));
          *middle_difpt += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int regular = 0;
       if (scatter_regular_ == "rawsubsqr-sample-biases") regular = 0;
  else if (scatter_regular_ == "logsubsqr-sample-biases") regular = 1;
  else if (scatter_regular_ == "expsubsqr-sample-biases") regular = 2;
  else if (scatter_regular_ == "rawsubabs-sample-biases") regular = 3;
  else if (scatter_regular_ == "logsubabs-sample-biases") regular = 4;
  else if (scatter_regular_ == "expsubabs-sample-biases") regular = 5;
  ScatterRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      numidx_datum,
    biases_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoBiasLossLayer<float>::ScatterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ScatterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusupdRegularForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int clusupd_regular_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* biases_datum,        Dtype* medium_datum,
          Dtype* medium_diffs) {
  if (clusupd_regular_ == 0) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        *medium_datpt += 2 * buffer_datum;
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_regular_ == 1) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        *medium_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_regular_ == 2) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        *medium_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_regular_ == 3) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign;
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_regular_ == 4) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign / (abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_regular_ == 5) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *biases_datpt - *bottom_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign * exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusupdRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int regular = 0;
       if (clusupd_regular_ == "rawsubsqr-sample-biases") regular = 0;
  else if (clusupd_regular_ == "logsubsqr-sample-biases") regular = 1;
  else if (clusupd_regular_ == "expsubsqr-sample-biases") regular = 2;
  else if (clusupd_regular_ == "rawsubabs-sample-biases") regular = 3;
  else if (clusupd_regular_ == "logsubabs-sample-biases") regular = 4;
  else if (clusupd_regular_ == "expsubabs-sample-biases") regular = 5;
  ClusupdRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      bottom_datum,
    bottom_label, numidx_datum,
    biases_datum, medium_datum,
    medium_diffs
  );
}
template void HomoBiasLossLayer<float>::ClusupdRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ClusupdRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatupdRegularForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int scatupd_regular_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* biases_datum,        Dtype* medium_datum,
          Dtype* medium_diffs) {
  if (scatupd_regular_ == 0) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 2 * buffer_datum;
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_regular_ == 1) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_regular_ == 2) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *medium_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_regular_ == 3) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign;
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_regular_ == 4) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign / (abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (scatupd_regular_ == 5) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* biases_datpt = biases_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *medium_datpt += buffer_dsign * exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatupdRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int regular = 0;
       if (scatupd_regular_ == "rawsubsqr-sample-biases") regular = 0;
  else if (scatupd_regular_ == "logsubsqr-sample-biases") regular = 1;
  else if (scatupd_regular_ == "expsubsqr-sample-biases") regular = 2;
  else if (scatupd_regular_ == "rawsubabs-sample-biases") regular = 3;
  else if (scatupd_regular_ == "logsubabs-sample-biases") regular = 4;
  else if (scatupd_regular_ == "expsubabs-sample-biases") regular = 5;
  ScatupdRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      bottom_datum,
    bottom_label, numidx_datum,
    biases_datum, medium_datum,
    medium_diffs
  );
}
template void HomoBiasLossLayer<float>::ScatupdRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ScatupdRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterReshuntForBias_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* maprop_datum,
    const Dtype* maprop_diffs,   const Dtype* storer_datum,
    const Dtype* storer_diffs,         Dtype* middle_datum,
          Dtype* middle_diffs) {
  CUDA_KERNEL_LOOP(inner_index, inner_numb_) {
    for (int match_index = 0; match_index < match_numb_; ++match_index) {
      for (int label_index = 0; label_index < label_numb_; ++label_index) {
        const int mapair_shift = match_index * label_numb_ + label_index;
        const int storer_shift = inner_index + inner_numb_ * mapair_shift;
        const int* mapair_datpt = mapair_datum + mapair_shift;
        const int* mapair_difpt = mapair_diffs + mapair_shift;
        const int srcmid_shift = inner_index + inner_numb_ * *mapair_datpt;
        const int trgmid_shift = inner_index + inner_numb_ * *mapair_difpt;
        const Dtype* maprop_datpt = maprop_datum + mapair_shift;
        const Dtype* maprop_difpt = maprop_diffs + mapair_shift;
        const Dtype* storer_datpt = storer_datum + storer_shift;
        const Dtype* storer_difpt = storer_diffs + storer_shift;
        if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
        Dtype* srcmid_datpt = middle_datum + srcmid_shift;
        Dtype* srcmid_difpt = middle_diffs + srcmid_shift;
        Dtype* trgmid_datpt = middle_datum + trgmid_shift;
        Dtype* trgmid_difpt = middle_diffs + trgmid_shift;
        *srcmid_datpt += *storer_datpt * *maprop_datpt;
        *trgmid_datpt += *storer_difpt * *maprop_difpt;
        *srcmid_difpt += 1;
        *trgmid_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OdotterReshunt_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* maprop_datum = maprop_blob_.gpu_data();
  const Dtype* maprop_diffs = maprop_blob_.gpu_diff();
  const Dtype* storer_datum = storer_blob_.gpu_data();
  const Dtype* storer_diffs = storer_blob_.gpu_diff();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(0), middle_datum);
  caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(0), middle_diffs);
  OdotterReshuntForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  mapair_datum,
    mapair_diffs, maprop_datum,
    maprop_diffs, storer_datum,
    storer_diffs, middle_datum,
    middle_diffs
  );
}
template void HomoBiasLossLayer<float>::OdotterReshunt_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OdotterReshunt_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterRegularForBias_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int  label_nmax_,
    const int  odotter_regular_, const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* bottom_datum,
    const Dtype* biases_datum,   const Dtype* medial_datum,
    const Dtype* medial_diffs,         Dtype* storer_datum,
          Dtype* storer_diffs) {
  if (odotter_regular_ == 0) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *storer_datpt += 2 * srcbuf_datum * *medial_datpt;
        *storer_difpt += 2 * trgbuf_datum * *medial_difpt;
      }
    }
  }
  else if (odotter_regular_ == 1) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *storer_datpt += 2 * srcbuf_datum / (srcbuf_datum * srcbuf_datum + 1) * *medial_datpt;
        *storer_difpt += 2 * trgbuf_datum / (trgbuf_datum * trgbuf_datum + 1) * *medial_difpt;
      }
    }
  }
  else if (odotter_regular_ == 2) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        *storer_datpt += 2 * srcbuf_datum * exp(-srcbuf_datum * srcbuf_datum) * *medial_datpt;
        *storer_difpt += 2 * trgbuf_datum * exp(-trgbuf_datum * trgbuf_datum) * *medial_difpt;
      }
    }
  }
  else if (odotter_regular_ == 3) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        *storer_datpt += srcbuf_dsign * *medial_datpt;
        *storer_difpt += trgbuf_dsign * *medial_difpt;
      }
    }
  }
  else if (odotter_regular_ == 4) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        *storer_datpt += srcbuf_dsign / (abs(srcbuf_datum) + 1) * *medial_datpt;
        *storer_difpt += trgbuf_dsign / (abs(trgbuf_datum) + 1) * *medial_difpt;
      }
    }
  }
  else if (odotter_regular_ == 5) {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index = round_index % inner_numb_;
      const int label_index = round_index / inner_numb_ % label_numb_;
      const int match_index = round_index / inner_numb_ / label_numb_;
      const int mapair_shift = match_index * label_numb_ + label_index;
      const int* mapair_datpt = mapair_datum + mapair_shift;
      const int* mapair_difpt = mapair_diffs + mapair_shift;
      const int srcbot_shift = inner_index + inner_numb_ * *mapair_datpt;
      const int trgbot_shift = inner_index + inner_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        const Dtype srcbuf_datum = *srcbot_datpt - *biases_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *biases_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        *storer_datpt += srcbuf_dsign * exp(-abs(srcbuf_datum)) * *medial_datpt;
        *storer_difpt += trgbuf_dsign * exp(-abs(trgbuf_datum)) * *medial_difpt;
      }
    }
  }
}            

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OdotterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> storer_shape(3);
  storer_shape[0] = match_numb_;
  storer_shape[1] = label_numb_;
  storer_shape[2] = inner_numb_;
  storer_blob_.Reshape(storer_shape);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  const Dtype* medial_datum = medial_blob_.gpu_data();
  const Dtype* medial_diffs = medial_blob_.gpu_diff();
  Dtype* storer_datum = storer_blob_.mutable_gpu_data();
  Dtype* storer_diffs = storer_blob_.mutable_gpu_diff();
  int regular = 0;
       if (odotter_regular_ == "rawsubsqr-sample-biases") regular = 0;
  else if (odotter_regular_ == "logsubsqr-sample-biases") regular = 1;
  else if (odotter_regular_ == "expsubsqr-sample-biases") regular = 2;
  else if (odotter_regular_ == "rawsubabs-sample-biases") regular = 3;
  else if (odotter_regular_ == "logsubabs-sample-biases") regular = 4;
  else if (odotter_regular_ == "expsubabs-sample-biases") regular = 5;
  OdotterRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      mapair_datum,
    mapair_diffs, bottom_datum,
    biases_datum, medial_datum,
    medial_diffs, storer_datum,
    storer_diffs
  );
}
template void HomoBiasLossLayer<float>::OdotterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OdotterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeRegularForBias_gpu_backend(
    const int match_numb_,     const int label_numb_,
    const int label_nmax_,     const int ovalize_regular_,
      Dtype* medial_datum,        Dtype* medial_diffs) {
  if (ovalize_regular_ == 0) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = 0 < buffer_diffs ? (+2 * buffer_datum / buffer_diffs) : 0;
          *medial_difpt = 0 < buffer_diffs ? (-2 * buffer_datum / buffer_diffs) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 1) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_datum / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_datum / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 2) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = *medial_difpt = (buffer_datum * buffer_datum + 1) * buffer_diffs;
          *medial_datpt = 0 < *medial_datpt ? (+2 * buffer_datum / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-2 * buffer_datum / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 3) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = (buffer_datum * buffer_datum + 1) * buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = (buffer_datum * buffer_datum + 1) * buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_datum / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_datum / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 4) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = 0 < buffer_diffs ? (+2 * buffer_datum * exp(-buffer_datum * buffer_datum) / buffer_diffs) : 0;
          *medial_difpt = 0 < buffer_diffs ? (-2 * buffer_datum * exp(-buffer_datum * buffer_datum) / buffer_diffs) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 5) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          *medial_datpt = buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_datum * exp(-buffer_datum * buffer_datum) / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_datum * exp(-buffer_datum * buffer_datum) / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 6) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = 0 < buffer_diffs ? (+buffer_dsign / buffer_diffs) : 0;
          *medial_difpt = 0 < buffer_diffs ? (-buffer_dsign / buffer_diffs) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 7) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_dsign / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_dsign / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 8) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = *medial_difpt = (abs(buffer_datum) + 1) * buffer_diffs;
          *medial_datpt = 0 < *medial_datpt ? (+buffer_dsign / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_dsign / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 9) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = (abs(buffer_datum) + 1) * buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = (abs(buffer_datum) + 1) * buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_dsign / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_dsign / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 10) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += *medial_datpt - *medial_difpt;
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = 0 < buffer_diffs ? (+buffer_dsign * exp(-abs(buffer_datum)) / buffer_diffs) : 0;
          *medial_difpt = 0 < buffer_diffs ? (-buffer_dsign * exp(-abs(buffer_datum)) / buffer_diffs) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
  else if (ovalize_regular_ == 11) {
    const int round_count = match_numb_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int medial_shift = round_index * label_nmax_ - 1;
      Dtype* medial_datpt = medial_datum + medial_shift;
      Dtype* medial_difpt = medial_diffs + medial_shift;
      Dtype buffer_datum = 0, buffer_diffs = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        ++medial_datpt; ++medial_difpt;
        if (*medial_datpt < 0 || *medial_difpt < 0) continue;
        buffer_datum += sqrt(*medial_datpt) - sqrt(*medial_difpt);
        buffer_diffs += 1;
      }
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
        if (*medial_datpt < 0 || *medial_difpt < 0) {
          *medial_datpt = *medial_difpt = -1;
        } else {
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *medial_datpt = buffer_diffs * sqrt(*medial_datpt);
          *medial_difpt = buffer_diffs * sqrt(*medial_difpt);
          *medial_datpt = 0 < *medial_datpt ? (+buffer_dsign * exp(-abs(buffer_datum)) / *medial_datpt) : 0;
          *medial_difpt = 0 < *medial_difpt ? (-buffer_dsign * exp(-abs(buffer_datum)) / *medial_difpt) : 0;
        }
        --medial_datpt; --medial_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  Dtype* medial_datum = medial_blob_.mutable_gpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_gpu_diff();
  int regular = 0;
       if (ovalize_regular_ == "rawsubsqr-origin-origin") regular = 0;
  else if (ovalize_regular_ == "rawsubsqr-sqroot-sqroot") regular = 1;
  else if (ovalize_regular_ == "logsubsqr-origin-origin") regular = 2;
  else if (ovalize_regular_ == "logsubsqr-sqroot-sqroot") regular = 3;
  else if (ovalize_regular_ == "expsubsqr-origin-origin") regular = 4;
  else if (ovalize_regular_ == "expsubsqr-sqroot-sqroot") regular = 5;
  else if (ovalize_regular_ == "rawsubabs-origin-origin") regular = 6;
  else if (ovalize_regular_ == "rawsubabs-sqroot-sqroot") regular = 7;
  else if (ovalize_regular_ == "logsubabs-origin-origin") regular = 8;
  else if (ovalize_regular_ == "logsubabs-sqroot-sqroot") regular = 9;
  else if (ovalize_regular_ == "expsubabs-origin-origin") regular = 10;
  else if (ovalize_regular_ == "expsubabs-sqroot-sqroot") regular = 11;
  OvalizeRegularForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  label_numb_,
    label_nmax_,  regular,
    medial_datum, medial_diffs
  );
}
template void HomoBiasLossLayer<float>::OvalizeRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OvalizeRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void PredictMeasureForBias_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int predict_measure_, const Dtype* numidx_datum,
    const Dtype* biases_datum,  const Dtype* bottom_datum,
    const Dtype* bottom_label,        Dtype* topper_datum) {
  if (predict_measure_ == 0) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= buffer_datum * buffer_datum;
      }
    }
  }
  else if (predict_measure_ == 1) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= log(buffer_datum * buffer_datum + 1);
      }
    }
  }
  else if (predict_measure_ == 2) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= 1 - exp(-buffer_datum * buffer_datum);
      }
    }
  }
  else if (predict_measure_ == 3) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= abs(buffer_datum);
      }
    }
  }
  else if (predict_measure_ == 4) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= log(abs(buffer_datum) + 1);
      }
    }
  }
  else if (predict_measure_ == 5) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int biases_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* biases_datpt = biases_datum + biases_shift;
        Dtype buffer_datum = *bottom_datpt - *biases_datpt;
        *topper_datpt -= 1 - exp(-abs(buffer_datum));
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::PredictMeasure_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[1]]->mutable_gpu_data();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* biases_datum = this->blobs_[1]->gpu_data();
  int measure = 0;
       if (predict_measure_ == "rawsubsqr-sample-biases") measure = 0;
  else if (predict_measure_ == "logsubsqr-sample-biases") measure = 1;
  else if (predict_measure_ == "expsubsqr-sample-biases") measure = 2;
  else if (predict_measure_ == "rawsubabs-sample-biases") measure = 3;
  else if (predict_measure_ == "logsubabs-sample-biases") measure = 4;
  else if (predict_measure_ == "expsubabs-sample-biases") measure = 5;
  PredictMeasureForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * label_nmax_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    biases_datum, bottom_datum,
    bottom_label, topper_datum
  );
}
template void HomoBiasLossLayer<float>::PredictMeasure_gpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void HomoBiasLossLayer<double>::PredictMeasure_gpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ClusterBackwardForBias_gpu_backend(
    const int outer_numb_,         const int inner_numb_,
    const int cluster_clipmode_,   const bool cluster_clipactv_,
    const Dtype cluster_clipnorm_, const Dtype cluster_clipprop_,
    const Dtype* topper_diffs,     const Dtype* middle_datum,
    const Dtype* middle_diffs,     const Dtype  blobal_datum,
    const Dtype  blobal_diffs,           Dtype* bottom_diffs) {
  if (cluster_clipmode_ == 0) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
            Dtype  sumsqr_diffs = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        sumsqr_diffs += *topper_difpt * *topper_difpt;
        ++middle_datpt; ++middle_difpt; ++topper_difpt;
      }
      const Dtype coeffi_alpha = cluster_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (cluster_clipmode_ == 1) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = cluster_clipnorm_ * cluster_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt;
      }
      const Dtype coeffi_alpha = cluster_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (cluster_clipmode_ == 2) {
    const Dtype coeffi_alpha = cluster_clipactv_ || blobal_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(blobal_diffs / blobal_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (cluster_clipmode_ == 3) {
    Dtype sumsqr_diffs = cluster_clipnorm_ * cluster_clipnorm_;
    const Dtype coeffi_alpha = cluster_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (cluster_clipmode_ == 4) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += 0 < *middle_difpt ? (cluster_clipprop_ * *middle_datpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt; ++bottom_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->gpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.gpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  Dtype  blobal_datum = 0, blobal_diffs = 0;
  int clipmode = 0;
       if (cluster_clipmode_ == "sample-diff-based") { clipmode = 0; }
  else if (cluster_clipmode_ == "sample-norm-based") { clipmode = 1; }
  else if (cluster_clipmode_ == "blobal-diff-based") { clipmode = 2;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs, &blobal_diffs);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (cluster_clipmode_ == "blobal-norm-based") { clipmode = 3;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (cluster_clipmode_ == "unclipped") { clipmode = 4; }
  ClusterBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          cluster_clipactv_,
    cluster_clipnorm_, cluster_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoBiasLossLayer<float>::ClusterBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ClusterBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterBackwardForBias_gpu_backend(
    const int outer_numb_,         const int inner_numb_,
    const int scatter_clipmode_,   const bool scatter_clipactv_,
    const Dtype scatter_clipnorm_, const Dtype scatter_clipprop_,
    const Dtype* topper_diffs,     const Dtype* middle_datum,
    const Dtype* middle_diffs,     const Dtype  blobal_datum,
    const Dtype  blobal_diffs,           Dtype* bottom_diffs) {
  if (scatter_clipmode_ == 0) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
            Dtype  sumsqr_diffs = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        sumsqr_diffs += *topper_difpt * *topper_difpt;
        ++middle_datpt; ++middle_difpt; ++topper_difpt;
      }
      const Dtype coeffi_alpha = scatter_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (scatter_clipmode_ == 1) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = scatter_clipnorm_ * scatter_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt;
      }
      const Dtype coeffi_alpha = scatter_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (scatter_clipmode_ == 2) {
    const Dtype coeffi_alpha = scatter_clipactv_ || blobal_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(blobal_diffs / blobal_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (scatter_clipmode_ == 3) {
    Dtype sumsqr_diffs = scatter_clipnorm_ * scatter_clipnorm_;
    const Dtype coeffi_alpha = scatter_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (scatter_clipmode_ == 4) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += 0 < *middle_difpt ? (scatter_clipprop_ * *middle_datpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt; ++bottom_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->gpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.gpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  Dtype  blobal_datum = 0, blobal_diffs = 0;
  int clipmode = 0;
       if (scatter_clipmode_ == "sample-diff-based") { clipmode = 0; }
  else if (scatter_clipmode_ == "sample-norm-based") { clipmode = 1; }
  else if (scatter_clipmode_ == "blobal-diff-based") { clipmode = 2;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs, &blobal_diffs);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (scatter_clipmode_ == "blobal-norm-based") { clipmode = 3;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (scatter_clipmode_ == "unclipped") { clipmode = 4; }
  ScatterBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          scatter_clipactv_,
    scatter_clipnorm_, scatter_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoBiasLossLayer<float>::ScatterBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::ScatterBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void TopdiffBackwardForBias_gpu_backend(
    const int outer_numb_,         const int inner_numb_,
    const int topdiff_clipmode_,   const bool topdiff_clipactv_,
    const Dtype topdiff_clipnorm_, const Dtype topdiff_clipprop_,
    const Dtype* topper_diffs,     const Dtype blobal_datum,
          Dtype* bottom_diffs) {
  if (topdiff_clipmode_ == 0) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = topdiff_clipnorm_ * topdiff_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += *topper_difpt * *topper_difpt;
        ++topper_difpt;
      }
      const Dtype coeffi_alpha = topdiff_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * topdiff_clipprop_) : 0) : topdiff_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --topper_difpt; --bottom_difpt;
        *bottom_difpt += coeffi_alpha * *topper_difpt;
      }
    }
  }
  else if (topdiff_clipmode_ == 1) {
    Dtype sumsqr_diffs = topdiff_clipnorm_ * topdiff_clipnorm_;
    const Dtype coeffi_alpha = topdiff_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * topdiff_clipprop_) : 0) : topdiff_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *topper_difpt;
        ++topper_difpt; ++bottom_difpt;
      }
    }
  }
  else if (topdiff_clipmode_ == 2) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += topdiff_clipprop_ * *topper_difpt;
        ++topper_difpt; ++bottom_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::TopdiffBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = top[outputs_activate_[0]]->gpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  Dtype  blobal_datum = 0;
  int clipmode = 0;
       if (topdiff_clipmode_ == "sample-norm-based") { clipmode = 0; }
  else if (topdiff_clipmode_ == "blobal-norm-based") { clipmode = 1;
    caffe_gpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs, &blobal_datum);
  }
  else if (topdiff_clipmode_ == "unclipped") { clipmode = 2; }
  TopdiffBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          topdiff_clipactv_,
    topdiff_clipnorm_, topdiff_clipprop_,
    topper_diffs,      blobal_datum,
    bottom_diffs
  );
}
template void HomoBiasLossLayer<float>::TopdiffBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::TopdiffBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeBackwardForBias_gpu_backend(
    const int outer_numb_,         const int inner_numb_,
    const int ovalize_clipmode_,   const bool ovalize_clipactv_,
    const Dtype ovalize_clipnorm_, const Dtype ovalize_clipprop_,
    const Dtype* topper_diffs,     const Dtype* middle_datum,
    const Dtype* middle_diffs,     const Dtype  blobal_datum,
    const Dtype  blobal_diffs,           Dtype* bottom_diffs) {
  if (ovalize_clipmode_ == 0) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
            Dtype  sumsqr_diffs = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        sumsqr_diffs += *topper_difpt * *topper_difpt;
        ++middle_datpt; ++middle_difpt; ++topper_difpt;
      }
      const Dtype coeffi_alpha = ovalize_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (ovalize_clipmode_ == 1) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_ + inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = ovalize_clipnorm_ * ovalize_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *middle_difpt ? (*middle_datpt * *middle_datpt / *middle_difpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt;
      }
      const Dtype coeffi_alpha = ovalize_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --middle_datpt; --middle_difpt; --bottom_difpt;
        *bottom_difpt += 0 < *middle_difpt ? (coeffi_alpha * *middle_datpt / *middle_difpt) : 0;
      }
    }
  }
  else if (ovalize_clipmode_ == 2) {
    const Dtype coeffi_alpha = ovalize_clipactv_ || blobal_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(blobal_diffs / blobal_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (ovalize_clipmode_ == 3) {
    Dtype sumsqr_diffs = ovalize_clipnorm_ * ovalize_clipnorm_;
    const Dtype coeffi_alpha = ovalize_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (ovalize_clipmode_ == 4) {
    CUDA_KERNEL_LOOP(outer_index, outer_numb_) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      const Dtype* middle_difpt = middle_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += 0 < *middle_difpt ? (ovalize_clipprop_ * *middle_datpt / *middle_difpt) : 0;
        ++middle_datpt; ++middle_difpt; ++bottom_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->gpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.gpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  Dtype  blobal_datum = 0, blobal_diffs = 0;
  int clipmode = 0;
       if (ovalize_clipmode_ == "sample-diff-based") { clipmode = 0; }
  else if (ovalize_clipmode_ == "sample-norm-based") { clipmode = 1; }
  else if (ovalize_clipmode_ == "blobal-diff-based") { clipmode = 2;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs, &blobal_diffs);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (ovalize_clipmode_ == "blobal-norm-based") { clipmode = 3;
    caffe_gpu_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    caffe_gpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum, &blobal_datum);
  }
  else if (ovalize_clipmode_ == "unclipped") { clipmode = 4; }
  OvalizeBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          ovalize_clipactv_,
    ovalize_clipnorm_, ovalize_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoBiasLossLayer<float>::OvalizeBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoBiasLossLayer<double>::OvalizeBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusupdBackwardForBias_gpu_backend(
    const int inner_numb_,         const int label_numb_,
    const int label_nmax_,         const int clusupd_clipmode_,
    const bool clusupd_clipactv_,  const Dtype clusupd_clipnorm_,
    const Dtype clusupd_clipprop_, const Dtype* medium_datum,
    const Dtype* medium_diffs,     const Dtype  blobal_datum,
          Dtype* biases_diffs) {
  if (clusupd_clipmode_ == 0) {
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_ + inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = clusupd_clipnorm_ * clusupd_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *medium_difpt ? (*medium_datpt * *medium_datpt / *medium_difpt / *medium_difpt) : 0;
        ++medium_datpt; ++medium_difpt;
      }
      const Dtype coeffi_alpha = clusupd_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * clusupd_clipprop_) : 0) : clusupd_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --medium_datpt; --medium_difpt; --biases_difpt;
        *biases_difpt += 0 < *medium_difpt ? (coeffi_alpha * *medium_datpt / *medium_difpt) : 0;
      }
    }
  }
  else if (clusupd_clipmode_ == 1) {
    Dtype sumsqr_diffs = clusupd_clipnorm_ * clusupd_clipnorm_;
    const Dtype coeffi_alpha = clusupd_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * clusupd_clipprop_) : 0) : clusupd_clipprop_;
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += coeffi_alpha * *medium_datpt;
        ++medium_datpt; ++biases_difpt;
      }
    }
  }
  else if (clusupd_clipmode_ == 2) {
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += 0 < *medium_difpt ? (clusupd_clipprop_ * *medium_datpt / *medium_difpt) : 0;
        ++medium_datpt; ++medium_difpt; ++biases_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusupdBackward_gpu() {
        Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.gpu_diff();
  Dtype* biases_diffs = this->blobs_[1]->mutable_gpu_diff();
  Dtype  blobal_datum = 0;
  int clipmode = 0;
       if (clusupd_clipmode_ == "biases-norm-based") { clipmode = 0; }
  else if (clusupd_clipmode_ == "blobal-norm-based") { clipmode = 1;
    caffe_gpu_div_nz(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_diffs, medium_datum);
    caffe_gpu_dot(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_datum, &blobal_datum);
  }
  else if (clusupd_clipmode_ == "unclipped") { clipmode = 2; }
  ClusupdBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    inner_numb_,       label_numb_,
    label_nmax_,       clipmode,
    clusupd_clipactv_, clusupd_clipnorm_,
    clusupd_clipprop_, medium_datum,
    medium_diffs,      blobal_datum,
    biases_diffs);
}
template void HomoBiasLossLayer<float>::ClusupdBackward_gpu();
template void HomoBiasLossLayer<double>::ClusupdBackward_gpu();

template <typename Dtype>
__global__ void ScatupdBackwardForBias_gpu_backend(
    const int inner_numb_,         const int label_numb_,
    const int label_nmax_,         const int scatupd_clipmode_,
    const bool scatupd_clipactv_,  const Dtype scatupd_clipnorm_,
    const Dtype scatupd_clipprop_, const Dtype* medium_datum,
    const Dtype* medium_diffs,     const Dtype  blobal_datum,
          Dtype* biases_diffs) {
  if (scatupd_clipmode_ == 0) {
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_ + inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
            Dtype  sumsqr_datum = 0;
      const Dtype  sumsqr_diffs = scatupd_clipnorm_ * scatupd_clipnorm_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        sumsqr_datum += 0 < *medium_difpt ? (*medium_datpt * *medium_datpt / *medium_difpt / *medium_difpt) : 0;
        ++medium_datpt; ++medium_difpt;
      }
      const Dtype coeffi_alpha = scatupd_clipactv_ || sumsqr_diffs < sumsqr_datum ?
        (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatupd_clipprop_) : 0) : scatupd_clipprop_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        --medium_datpt; --medium_difpt; --biases_difpt;
        *biases_difpt += 0 < *medium_difpt ? (coeffi_alpha * *medium_datpt / *medium_difpt) : 0;
      }
    }
  }
  else if (scatupd_clipmode_ == 1) {
    Dtype sumsqr_diffs = scatupd_clipnorm_ * scatupd_clipnorm_;
    const Dtype coeffi_alpha = scatupd_clipactv_ || sumsqr_diffs < blobal_datum ?
      (0 < blobal_datum ? (sqrt(sumsqr_diffs / blobal_datum) * scatupd_clipprop_) : 0) : scatupd_clipprop_;
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += coeffi_alpha * *medium_datpt;
        ++medium_datpt; ++biases_difpt;
      }
    }
  }
  else if (scatupd_clipmode_ == 2) {
    const int round_count = label_numb_ * label_nmax_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      const Dtype* medium_difpt = medium_diffs + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += 0 < *medium_difpt ? (scatupd_clipprop_ * *medium_datpt / *medium_difpt) : 0;
        ++medium_datpt; ++medium_difpt; ++biases_difpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatupdBackward_gpu() {
        Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.gpu_diff();
  Dtype* biases_diffs = this->blobs_[1]->mutable_gpu_diff();
  Dtype  blobal_datum = 0;
  int clipmode = 0;
       if (scatupd_clipmode_ == "biases-norm-based") { clipmode = 0; }
  else if (scatupd_clipmode_ == "blobal-norm-based") { clipmode = 1;
    caffe_gpu_div_nz(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_diffs, medium_datum);
    caffe_gpu_dot(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_datum, &blobal_datum);
  }
  else if (scatupd_clipmode_ == "unclipped") { clipmode = 2; }
  ScatupdBackwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    inner_numb_,       label_numb_,
    label_nmax_,       clipmode,
    scatupd_clipactv_, scatupd_clipnorm_,
    scatupd_clipprop_, medium_datum,
    medium_diffs,      blobal_datum,
    biases_diffs);
}
template void HomoBiasLossLayer<float>::ScatupdBackward_gpu();
template void HomoBiasLossLayer<double>::ScatupdBackward_gpu();

template <typename Dtype>
__global__ void ForwardForBias_gpu_backend(
    const int outer_numb_,   const int label_numb_,
    const int label_nmax_,   const int ignore_label_,
    const int target_label_, const Dtype* bottom_label,
    Dtype* numidx_datum) {
  const int round_count = label_numb_ * label_nmax_;
  CUDA_KERNEL_LOOP(round_index, round_count) {
    const int label_index = round_index / label_nmax_;
    const int labmx_index = round_index % label_nmax_;
    Dtype* numidx_datpt = numidx_datum + round_index;
    if (labmx_index != ignore_label_ && labmx_index != target_label_) {
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int label_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + label_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        ++*numidx_datpt;
      }
    }
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (this->phase_ != TRAIN) {
    if (outputs_activate_[1] != -1) {
      PredictMeasure_gpu(bottom, top);
    }
    if (outputs_activate_[2] != -1) {
      clustr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_data());
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_diff());
      ClusterMeasure_gpu(bottom); ClusterForward_gpu(top);
    }
    if (outputs_activate_[3] != -1) {
      scattr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_gpu_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_gpu_data());
      caffe_gpu_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_gpu_diff());
      ScatterMeasure_gpu(bottom); ScatterForward_gpu(top);
    }
    if (outputs_activate_[4] != -1) {
      clusup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_gpu_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_gpu_data());
      caffe_gpu_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_gpu_diff());
      ClusupdMeasure_gpu(bottom); ClusupdForward_gpu(top);
    }
    if (outputs_activate_[5] != -1) {
      scatup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_gpu_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_gpu_data());
      caffe_gpu_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_gpu_diff());
      ScatupdMeasure_gpu(bottom); ScatupdForward_gpu(top);
    }
    if (outputs_activate_[6] != -1) {
      ovaliz_blob_.Reshape(vector<int>(1, label_numb_));
      caffe_gpu_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_gpu_data());
      caffe_gpu_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_gpu_diff());
      OvalizeMatcher_gpu(bottom); OdotterMeasure_gpu(bottom);
      OvalizeMeasure_gpu(bottom); OvalizeForward_gpu(top);
    }
    return;
  }
  if (!preforward_flag && preforward_tag_) {
    preforward_tag_ = false;
  } else if (preforward_flag && !preforward_tag_) {
    preforward_beg_ = preforward_tag_ = true;
  }
  if ((!biashit_initmode_ && !preforward_tag_) || (biashit_initmode_ && preforward_beg_)) {
    const int numidx_count = label_numb_ * label_nmax_;
    caffe_gpu_set(numidx_count, Dtype(0), this->blobs_[0]->mutable_gpu_data());
    if (outputs_activate_[2] != -1) {
      clustr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_data());
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_diff());
    }
    if (outputs_activate_[3] != -1) {
      scattr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_gpu_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_gpu_data());
      caffe_gpu_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_gpu_diff());
    }
    if (outputs_activate_[4] != -1) {
      clusup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_gpu_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_gpu_data());
      caffe_gpu_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_gpu_diff());
    }
    if (outputs_activate_[5] != -1) {
      scatup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_gpu_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_gpu_data());
      caffe_gpu_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_gpu_diff());
    }
    if (outputs_activate_[6] != -1) {
      ovaliz_blob_.Reshape(vector<int>(1, label_numb_));
      caffe_gpu_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_gpu_data());
      caffe_gpu_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_gpu_diff());
    }
  }
  if (preforward_beg_) preforward_beg_ = false;
  if ((!biashit_initmode_ && !preforward_tag_) || (biashit_initmode_ && preforward_tag_)) {
    const Dtype* bottom_label = bottom[1]->gpu_data();
    Dtype* numidx_datum = this->blobs_[0]->mutable_gpu_data();
    ForwardForBias_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
      outer_numb_,   label_numb_,
      label_nmax_,   ignore_label_,
      target_label_, bottom_label,
      numidx_datum
    );
  }
  if (outputs_activate_[1] != -1) { PredictMeasure_gpu(bottom, top); }
  if (outputs_activate_[2] != -1 && !preforward_tag_) { ClusterMeasure_gpu(bottom); ClusterForward_gpu(top); }
  if (outputs_activate_[3] != -1 && !preforward_tag_) { ScatterMeasure_gpu(bottom); ScatterForward_gpu(top); }
  if (outputs_activate_[4] != -1 && !preforward_tag_) { ClusupdMeasure_gpu(bottom); ClusupdForward_gpu(top); }
  if (outputs_activate_[5] != -1 && !preforward_tag_) { ScatupdMeasure_gpu(bottom); ScatupdForward_gpu(top); }
  if (outputs_activate_[6] != -1 && !preforward_tag_) { OvalizeMatcher_gpu(bottom); OdotterMeasure_gpu(bottom);
                                                        OvalizeMeasure_gpu(bottom); OvalizeForward_gpu(top); }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << "Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    caffe_gpu_set(outer_numb_ * inner_numb_, Dtype(0), bottom[0]->mutable_gpu_diff());
    if (cluster_clipprop_ != Dtype(0) && cluster_interval_ &&
        solver_iter_ % cluster_interval_ >= cluster_postpone_ &&
        solver_iter_ % cluster_interval_ <  cluster_postpone_ + cluster_duration_) {
      ClusterRegular_gpu(bottom); ClusterBackward_gpu(top, bottom);
    }
    if (scatter_clipprop_ != Dtype(0) && scatter_interval_ &&
        solver_iter_ % scatter_interval_ >= scatter_postpone_ &&
        solver_iter_ % scatter_interval_ <  scatter_postpone_ + scatter_duration_) {
      ScatterRegular_gpu(bottom); ScatterBackward_gpu(top, bottom);
    }
    if (ovalize_clipprop_ != Dtype(0) && ovalize_interval_ &&
        solver_iter_ % ovalize_interval_ >= ovalize_postpone_ &&
        solver_iter_ % ovalize_interval_ <  ovalize_postpone_ + ovalize_duration_) {
      string  measure  = odotter_measure_;
      odotter_measure_ = odotter_regular_;
      OvalizeMatcher_gpu(bottom); OdotterMeasure_gpu(bottom);
      OvalizeRegular_gpu(bottom); OdotterRegular_gpu(bottom);
      OdotterReshunt_gpu(bottom); OvalizeBackward_gpu(top, bottom);
      odotter_measure_ = measure;
    }
    if (topdiff_clipprop_ != Dtype(0) && topdiff_interval_ &&
        solver_iter_ % topdiff_interval_ >= topdiff_postpone_ &&
        solver_iter_ % topdiff_interval_ <  topdiff_postpone_ + topdiff_duration_) {
      TopdiffBackward_gpu(top, bottom);
    }
  }
  if (this->param_propagate_down_[1]) {
    caffe_gpu_set(label_numb_ * label_nmax_ * inner_numb_, Dtype(0), this->blobs_[1]->mutable_gpu_diff());
    if (clusupd_clipprop_ != Dtype(0) && clusupd_interval_ &&
        solver_iter_ % clusupd_interval_ >= clusupd_postpone_ &&
        solver_iter_ % clusupd_interval_ <  clusupd_postpone_ + clusupd_duration_) {
      ClusupdRegular_gpu(bottom); ClusupdBackward_gpu();
    }
    if (scatupd_clipprop_ != Dtype(0) && scatupd_interval_ &&
        solver_iter_ % scatupd_interval_ >= scatupd_postpone_ &&
        solver_iter_ % scatupd_interval_ <  scatupd_postpone_ + scatupd_duration_) {
      ScatupdRegular_gpu(bottom); ScatupdBackward_gpu();
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HomoBiasLossLayer);
} // namespace caffe