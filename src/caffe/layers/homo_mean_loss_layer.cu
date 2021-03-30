#include <curand_kernel.h>

#include "caffe/layers/homo_mean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ClusterForwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ClusterForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[2]]->mutable_cpu_data();
  Dtype* clustr_datum = clustr_blob_.mutable_gpu_data();
  Dtype* clustr_diffs = clustr_blob_.mutable_gpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  ClusterForwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_, inner_numb_, middle_datum, middle_diffs, clustr_datum, clustr_diffs);
  caffe_gpu_asum(inner_numb_, clustr_datum, topper_datum);
  *topper_datum /= inner_numb_;
}
template void HomoMeanLossLayer<float>::ClusterForward_gpu(const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::ClusterForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ScatterForwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ScatterForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[3]]->mutable_cpu_data();
  Dtype* scattr_datum = scattr_blob_.mutable_gpu_data();
  Dtype* scattr_diffs = scattr_blob_.mutable_gpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  ScatterForwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_, inner_numb_, middle_datum, middle_diffs, scattr_datum, scattr_diffs);
  caffe_gpu_asum(inner_numb_, scattr_datum, topper_datum);
  *topper_datum /= inner_numb_;
}
template void HomoMeanLossLayer<float>::ScatterForward_gpu(const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::ScatterForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ClusupdForwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ClusupdForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[4]]->mutable_gpu_data();
  Dtype* clusup_datum = clusup_blob_.mutable_gpu_data();
  Dtype* clusup_diffs = clusup_blob_.mutable_gpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  ClusupdForwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    label_numb_,  label_nmax_,  inner_numb_,
    medium_datum, medium_diffs, clusup_datum,
    clusup_diffs, topper_datum
  );
}
template void HomoMeanLossLayer<float>::ClusupdForward_gpu(const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::ClusupdForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ScatupdForwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ScatupdForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[5]]->mutable_gpu_data();
  Dtype* scatup_datum = scatup_blob_.mutable_gpu_data();
  Dtype* scatup_diffs = scatup_blob_.mutable_gpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  ScatupdForwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    label_numb_,  label_nmax_,  inner_numb_,
    medium_datum, medium_diffs, scatup_datum,
    scatup_diffs, topper_datum
  );
}
template void HomoMeanLossLayer<float>::ScatupdForward_gpu(const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::ScatupdForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void OvalizeForwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OvalizeForward_gpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[6]]->mutable_cpu_data();
  Dtype* ovaliz_datum = ovaliz_blob_.mutable_gpu_data();
  Dtype* ovaliz_diffs = ovaliz_blob_.mutable_gpu_diff();
  const Dtype* caches_datum = caches_blob_.mutable_gpu_data();
  const Dtype* caches_diffs = caches_blob_.mutable_gpu_diff();
  OvalizeForwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_, label_numb_, caches_datum, caches_diffs, ovaliz_datum, ovaliz_diffs);
  caffe_gpu_asum(label_numb_, ovaliz_datum, topper_datum);
  *topper_datum /= label_numb_;
}
template void HomoMeanLossLayer<float>::OvalizeForward_gpu(const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::OvalizeForward_gpu(const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void OvalizeMatcherForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OvalizeMatcher_gpu(const vector<Blob<Dtype>*>& bottom) {
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
  OvalizeMatcherForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
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
template void HomoMeanLossLayer<float>::OvalizeMatcher_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OvalizeMatcher_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusterTestingForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int cluster_measure_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += 1 - exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ClusterTesting_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int measure = 0;
       if (cluster_measure_ == "rawsubsqr-overall-average" || cluster_measure_ == "rawsubsqr-nonself-average") measure = 0;
  else if (cluster_measure_ == "logsubsqr-overall-average" || cluster_measure_ == "logsubsqr-nonself-average") measure = 1;
  else if (cluster_measure_ == "expsubsqr-overall-average" || cluster_measure_ == "expsubsqr-nonself-average") measure = 2;
  else if (cluster_measure_ == "rawsubabs-overall-average" || cluster_measure_ == "rawsubabs-nonself-average") measure = 3;
  else if (cluster_measure_ == "logsubabs-overall-average" || cluster_measure_ == "logsubabs-nonself-average") measure = 4;
  else if (cluster_measure_ == "expsubabs-overall-average" || cluster_measure_ == "expsubabs-nonself-average") measure = 5;
  ClusterTestingForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusterTesting_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusterTesting_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusterMeasureForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int cluster_measure_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += buffer_datum * buffer_datum;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += log(buffer_datum * buffer_datum + 1);
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += log(buffer_datum * buffer_datum + 1);
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += 1 - exp(-buffer_datum * buffer_datum);
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 6) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += abs(buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 7) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += abs(buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 8) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += log(abs(buffer_datum) + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 9) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += log(abs(buffer_datum) + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 10) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += 1 - exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_measure_ == 11) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += 1 - exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ClusterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int measure = 0;
       if (cluster_measure_ == "rawsubsqr-overall-average") measure = 0;
  else if (cluster_measure_ == "rawsubsqr-nonself-average") measure = 1;
  else if (cluster_measure_ == "logsubsqr-overall-average") measure = 2;
  else if (cluster_measure_ == "logsubsqr-nonself-average") measure = 3;
  else if (cluster_measure_ == "expsubsqr-overall-average") measure = 4;
  else if (cluster_measure_ == "expsubsqr-nonself-average") measure = 5;
  else if (cluster_measure_ == "rawsubabs-overall-average") measure = 6;
  else if (cluster_measure_ == "rawsubabs-nonself-average") measure = 7;
  else if (cluster_measure_ == "logsubabs-overall-average") measure = 8;
  else if (cluster_measure_ == "logsubabs-nonself-average") measure = 9;
  else if (cluster_measure_ == "expsubabs-overall-average") measure = 10;
  else if (cluster_measure_ == "expsubabs-nonself-average") measure = 11;
  ClusterMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterMeasureForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int scatter_measure_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
          *middle_datpt += 1 - exp(-abs(buffer_datum));
          *middle_difpt += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ScatterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int measure = 0;
       if (scatter_measure_ == "rawsubsqr-samples-average") measure = 0;
  else if (scatter_measure_ == "rawsubsqr-average-average") measure = 1;
  else if (scatter_measure_ == "logsubsqr-samples-average") measure = 2;
  else if (scatter_measure_ == "logsubsqr-average-average") measure = 3;
  else if (scatter_measure_ == "expsubsqr-samples-average") measure = 4;
  else if (scatter_measure_ == "expsubsqr-average-average") measure = 5;
  else if (scatter_measure_ == "rawsubabs-samples-average") measure = 6;
  else if (scatter_measure_ == "rawsubabs-average-average") measure = 7;
  else if (scatter_measure_ == "logsubabs-samples-average") measure = 8;
  else if (scatter_measure_ == "logsubabs-average-average") measure = 9;
  else if (scatter_measure_ == "expsubabs-samples-average") measure = 10;
  else if (scatter_measure_ == "expsubabs-average-average") measure = 11;
  ScatterMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::ScatterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ScatterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusupdTestingForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int clusupd_measure_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* avgsum_datum,        Dtype* medium_datum,
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ClusupdTesting_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int measure = 0;
       if (clusupd_measure_ == "rawsubsqr-overall-average" || clusupd_measure_ == "rawsubsqr-nonself-average") measure = 0;
  else if (clusupd_measure_ == "logsubsqr-overall-average" || clusupd_measure_ == "logsubsqr-nonself-average") measure = 1;
  else if (clusupd_measure_ == "expsubsqr-overall-average" || clusupd_measure_ == "expsubsqr-nonself-average") measure = 2;
  else if (clusupd_measure_ == "rawsubabs-overall-average" || clusupd_measure_ == "rawsubabs-nonself-average") measure = 3;
  else if (clusupd_measure_ == "logsubabs-overall-average" || clusupd_measure_ == "logsubabs-nonself-average") measure = 4;
  else if (clusupd_measure_ == "expsubabs-overall-average" || clusupd_measure_ == "expsubabs-nonself-average") measure = 5;
  ClusupdTestingForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      bottom_datum,
    bottom_label, numidx_datum,
    avgsum_datum, medium_datum,
    medium_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusupdTesting_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusupdTesting_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusupdMeasureForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int clusupd_measure_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* avgsum_datum,        Dtype* medium_datum,
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += buffer_datum * buffer_datum;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += log(buffer_datum * buffer_datum + 1);
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += log(buffer_datum * buffer_datum + 1);
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += 1 - exp(-buffer_datum * buffer_datum);
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += 1 - exp(-buffer_datum * buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 6) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += abs(buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 7) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += abs(buffer_datum);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 8) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += log(abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 9) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += log(abs(buffer_datum) + 1);
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 10) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
  else if (clusupd_measure_ == 11) {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int inner_index  = round_index % inner_numb_;
      const int labmx_index  = round_index / inner_numb_ % label_nmax_;
      const int label_index  = round_index / inner_numb_ / label_nmax_;
      const int numidx_shift = round_index / inner_numb_;
            Dtype* medium_datpt = medium_datum + round_index;
            Dtype* medium_difpt = medium_diffs + round_index;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ClusupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int measure = 0;
       if (clusupd_measure_ == "rawsubsqr-overall-average") measure = 0;
  else if (clusupd_measure_ == "rawsubsqr-nonself-average") measure = 1;
  else if (clusupd_measure_ == "logsubsqr-overall-average") measure = 2;
  else if (clusupd_measure_ == "logsubsqr-nonself-average") measure = 3;
  else if (clusupd_measure_ == "expsubsqr-overall-average") measure = 4;
  else if (clusupd_measure_ == "expsubsqr-nonself-average") measure = 5;
  else if (clusupd_measure_ == "rawsubabs-overall-average") measure = 6;
  else if (clusupd_measure_ == "rawsubabs-nonself-average") measure = 7;
  else if (clusupd_measure_ == "logsubabs-overall-average") measure = 8;
  else if (clusupd_measure_ == "logsubabs-nonself-average") measure = 9;
  else if (clusupd_measure_ == "expsubabs-overall-average") measure = 10;
  else if (clusupd_measure_ == "expsubabs-nonself-average") measure = 11;
  ClusupdMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      bottom_datum,
    bottom_label, numidx_datum,
    avgsum_datum, medium_datum,
    medium_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusupdMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusupdMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatupdMeasureForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,      
    const int scatupd_measure_, const Dtype* bottom_datum,
    const Dtype* bottom_label,  const Dtype* numidx_datum,
    const Dtype* avgsum_datum,        Dtype* medium_datum,
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
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
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum + round_index;
                  *medium_datpt = *medium_difpt = 0;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift;
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift;
        if (static_cast<int>(*bottom_labpt) == labmx_index) continue;
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        if (static_cast<int>(*numidx_datit) < 1) continue;
        Dtype buffer_datum = *avgsum_datit - *avgsum_datpt;
        *medium_datpt += 1 - exp(-abs(buffer_datum));
        *medium_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ScatupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* medium_datum = medium_blob_.mutable_gpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
  int measure = 0;
       if (scatupd_measure_ == "rawsubsqr-samples-average") measure = 0;
  else if (scatupd_measure_ == "rawsubsqr-average-average") measure = 1;
  else if (scatupd_measure_ == "logsubsqr-samples-average") measure = 2;
  else if (scatupd_measure_ == "logsubsqr-average-average") measure = 3;
  else if (scatupd_measure_ == "expsubsqr-samples-average") measure = 4;
  else if (scatupd_measure_ == "expsubsqr-average-average") measure = 5;
  else if (scatupd_measure_ == "rawsubabs-samples-average") measure = 6;
  else if (scatupd_measure_ == "rawsubabs-average-average") measure = 7;
  else if (scatupd_measure_ == "logsubabs-samples-average") measure = 8;
  else if (scatupd_measure_ == "logsubabs-average-average") measure = 9;
  else if (scatupd_measure_ == "expsubabs-samples-average") measure = 10;
  else if (scatupd_measure_ == "expsubabs-average-average") measure = 11;
  ScatupdMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      bottom_datum,
    bottom_label, numidx_datum,
    avgsum_datum, medium_datum,
    medium_diffs
  );
}
template void HomoMeanLossLayer<float>::ScatupdMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ScatupdMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterTestingForMean_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int  label_nmax_,
    const int  odotter_measure_, const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* bottom_datum,
    const Dtype* numidx_datum,   const Dtype* avgsum_datum,
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += srcbuf_datum * srcbuf_datum;
        *medial_difpt += trgbuf_datum * trgbuf_datum;
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += log(srcbuf_datum * srcbuf_datum + 1);
        *medial_difpt += log(trgbuf_datum * trgbuf_datum + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += 1 - exp(-srcbuf_datum * srcbuf_datum);
        *medial_difpt += 1 - exp(-trgbuf_datum * trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += abs(srcbuf_datum);
        *medial_difpt += abs(trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += log(abs(srcbuf_datum) + 1);
        *medial_difpt += log(abs(trgbuf_datum) + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += 1 - exp(-abs(srcbuf_datum));
        *medial_difpt += 1 - exp(-abs(trgbuf_datum));
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::OdotterTesting_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> medial_shape(3);
  medial_shape[0] = match_numb_;
  medial_shape[1] = label_numb_;
  medial_shape[2] = label_nmax_;
  medial_blob_.Reshape(medial_shape);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* medial_datum = medial_blob_.mutable_gpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_gpu_diff();
  int measure = 0;
       if (odotter_measure_ == "rawsubsqr-overall-average" || odotter_measure_ == "rawsubsqr-nonself-average") measure = 0;
  else if (odotter_measure_ == "logsubsqr-overall-average" || odotter_measure_ == "logsubsqr-nonself-average") measure = 1;
  else if (odotter_measure_ == "expsubsqr-overall-average" || odotter_measure_ == "expsubsqr-nonself-average") measure = 2;
  else if (odotter_measure_ == "rawsubabs-overall-average" || odotter_measure_ == "rawsubabs-nonself-average") measure = 3;
  else if (odotter_measure_ == "logsubabs-overall-average" || odotter_measure_ == "logsubabs-nonself-average") measure = 4;
  else if (odotter_measure_ == "expsubabs-overall-average" || odotter_measure_ == "expsubabs-nonself-average") measure = 5;
  OdotterTestingForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      mapair_datum,
    mapair_diffs, bottom_datum,
    numidx_datum, avgsum_datum,
    medial_datum, medial_diffs
  );
}
template void HomoMeanLossLayer<float>::OdotterTesting_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OdotterTesting_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterMeasureForMean_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int  label_nmax_,
    const int  odotter_measure_, const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* bottom_datum,
    const Dtype* bottom_label,   const Dtype* numidx_datum,
    const Dtype* avgsum_datum,         Dtype* medial_datum,
          Dtype* medial_diffs) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += srcbuf_datum * srcbuf_datum;
        *medial_difpt += trgbuf_datum * trgbuf_datum;
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += srcbuf_datum * srcbuf_datum;
        *medial_difpt += trgbuf_datum * trgbuf_datum;
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += log(srcbuf_datum * srcbuf_datum + 1);
        *medial_difpt += log(trgbuf_datum * trgbuf_datum + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += log(srcbuf_datum * srcbuf_datum + 1);
        *medial_difpt += log(trgbuf_datum * trgbuf_datum + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += 1 - exp(-srcbuf_datum * srcbuf_datum);
        *medial_difpt += 1 - exp(-trgbuf_datum * trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += 1 - exp(-srcbuf_datum * srcbuf_datum);
        *medial_difpt += 1 - exp(-trgbuf_datum * trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 6) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += abs(srcbuf_datum);
        *medial_difpt += abs(trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 7) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += abs(srcbuf_datum);
        *medial_difpt += abs(trgbuf_datum);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 8) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += log(abs(srcbuf_datum) + 1);
        *medial_difpt += log(abs(trgbuf_datum) + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 9) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += log(abs(srcbuf_datum) + 1);
        *medial_difpt += log(abs(trgbuf_datum) + 1);
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 10) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        *medial_datpt += 1 - exp(-abs(srcbuf_datum));
        *medial_difpt += 1 - exp(-abs(trgbuf_datum));
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
  else if (odotter_measure_ == 11) {
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
      const Dtype* avgsum_datpt = avgsum_datum +  numidx_shift * inner_numb_;
      const Dtype* srcbot_datpt = bottom_datum + *mapair_datpt * inner_numb_;
      const Dtype* trgbot_datpt = bottom_datum + *mapair_difpt * inner_numb_;
      const Dtype* srcbot_labpt = bottom_label + *mapair_datpt * label_numb_ + label_index;
      const Dtype* trgbot_labpt = bottom_label + *mapair_difpt * label_numb_ + label_index;
      const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
      const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
      Dtype* medial_datpt = medial_datum + round_index;
      Dtype* medial_difpt = medial_diffs + round_index;
            *medial_datpt = *medial_difpt = -1;
      if (static_cast<int>(*numidx_datpt) < 1 + srclab_check) continue;
      if (static_cast<int>(*numidx_datpt) < 1 + trglab_check) continue;
      if (*mapair_datpt < 0 || *mapair_difpt < 0) continue;
            *medial_datpt = *medial_difpt = 0;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srcbuf_error * srclab_check;
        const Dtype avgsum_volum = *avgsum_datpt + trgbuf_error * trglab_check;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        *medial_datpt += 1 - exp(-abs(srcbuf_datum));
        *medial_difpt += 1 - exp(-abs(trgbuf_datum));
        ++srcbot_datpt; ++trgbot_datpt;
        ++avgsum_datpt;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::OdotterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> medial_shape(3);
  medial_shape[0] = match_numb_;
  medial_shape[1] = label_numb_;
  medial_shape[2] = label_nmax_;
  medial_blob_.Reshape(medial_shape);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* medial_datum = medial_blob_.mutable_gpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_gpu_diff();
  int measure = 0;
       if (odotter_measure_ == "rawsubsqr-overall-average") measure = 0;
  else if (odotter_measure_ == "rawsubsqr-nonself-average") measure = 1;
  else if (odotter_measure_ == "logsubsqr-overall-average") measure = 2;
  else if (odotter_measure_ == "logsubsqr-nonself-average") measure = 3;
  else if (odotter_measure_ == "expsubsqr-overall-average") measure = 4;
  else if (odotter_measure_ == "expsubsqr-nonself-average") measure = 5;
  else if (odotter_measure_ == "rawsubabs-overall-average") measure = 6;
  else if (odotter_measure_ == "rawsubabs-nonself-average") measure = 7;
  else if (odotter_measure_ == "logsubabs-overall-average") measure = 8;
  else if (odotter_measure_ == "logsubabs-nonself-average") measure = 9;
  else if (odotter_measure_ == "expsubabs-overall-average") measure = 10;
  else if (odotter_measure_ == "expsubabs-nonself-average") measure = 11;
  OdotterMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      mapair_datum,
    mapair_diffs, bottom_datum,
    bottom_label, numidx_datum,
    avgsum_datum, medial_datum,
    medial_diffs
  );
}
template void HomoMeanLossLayer<float>::OdotterMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OdotterMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeMeasureForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OvalizeMeasure_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> caches_shape(2);
  caches_shape[0] = match_numb_;
  caches_shape[1] = label_numb_;
  caches_blob_.Reshape(caches_shape);
  const Dtype* medial_datum = medial_blob_.gpu_data();
  const Dtype* medial_diffs = medial_blob_.gpu_diff();
  Dtype* caches_datum = caches_blob_.mutable_gpu_data();
  Dtype* caches_diffs = caches_blob_.mutable_gpu_diff();
  int measure = 0;
       if (ovalize_measure_ == "rawsubsqr-origins-origins") measure = 0;
  else if (ovalize_measure_ == "rawsubsqr-sqroots-sqroots") measure = 1;
  else if (ovalize_measure_ == "logsubsqr-origins-origins") measure = 2;
  else if (ovalize_measure_ == "logsubsqr-sqroots-sqroots") measure = 3;
  else if (ovalize_measure_ == "expsubsqr-origins-origins") measure = 4;
  else if (ovalize_measure_ == "expsubsqr-sqroots-sqroots") measure = 5;
  else if (ovalize_measure_ == "rawsubabs-origins-origins") measure = 6;
  else if (ovalize_measure_ == "rawsubabs-sqroots-sqroots") measure = 7;
  else if (ovalize_measure_ == "logsubabs-origins-origins") measure = 8;
  else if (ovalize_measure_ == "logsubabs-sqroots-sqroots") measure = 9;
  else if (ovalize_measure_ == "expsubabs-origins-origins") measure = 10;
  else if (ovalize_measure_ == "expsubabs-sqroots-sqroots") measure = 11;
  OvalizeMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  label_numb_,
    label_nmax_,  measure,
    medial_datum, medial_diffs,
    caches_datum, caches_diffs
  );
}
template void HomoMeanLossLayer<float>::OvalizeMeasure_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OvalizeMeasure_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ClusterRegularForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int cluster_regular_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_diffs = *numidx_datpt / (*numidx_datpt - 1);
        *middle_datpt += 2 * buffer_datum * buffer_diffs * buffer_diffs;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += (*numidx_datpt * 2 - 2) * buffer_datum / *numidx_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *middle_datpt += *numidx_datpt * 2 * buffer_datum / (*numidx_datpt - 1);
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
        *middle_datpt += (*numidx_datpt - 1) * buffer_datum / *numidx_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 6) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
        *middle_datpt += (*numidx_datpt - 1) * buffer_datum / *numidx_datpt;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 7) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 8) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *middle_datpt += (*numidx_datpt - 1) * buffer_dsign / *numidx_datpt;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 9) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        *middle_datpt += buffer_dsign;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 10) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        buffer_datum = buffer_dsign / (abs(buffer_datum) + 1);
        *middle_datpt += (*numidx_datpt - 1) * buffer_datum / *numidx_datpt;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 11) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += buffer_dsign / (abs(buffer_datum) + 1);
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 12) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 1) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        buffer_datum = buffer_dsign * exp(-abs(buffer_datum));
        *middle_datpt += (*numidx_datpt - 1) * buffer_datum / *numidx_datpt;
        *middle_difpt += 1;
      }
    }
  }
  else if (cluster_regular_ == 13) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
        if (static_cast<int>(*numidx_datpt) < 2) continue;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
        buffer_datum = *numidx_datpt * buffer_datum / (*numidx_datpt - 1);
        *middle_datpt += buffer_dsign * exp(-abs(buffer_datum));
        *middle_difpt += 1;
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ClusterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int regular = 0;
       if (cluster_regular_ == "rawsubsqr-overall-cluster") regular = 0;
  else if (cluster_regular_ == "rawsubsqr-nonself-cluster") regular = 1;
  else if (cluster_regular_ == "rawsubsqr-overall-average") regular = 2;
  else if (cluster_regular_ == "rawsubsqr-nonself-average") regular = 3;
  else if (cluster_regular_ == "logsubsqr-overall-average") regular = 4;
  else if (cluster_regular_ == "logsubsqr-nonself-average") regular = 5;
  else if (cluster_regular_ == "expsubsqr-overall-average") regular = 6;
  else if (cluster_regular_ == "expsubsqr-nonself-average") regular = 7;
  else if (cluster_regular_ == "rawsubabs-overall-average") regular = 8;
  else if (cluster_regular_ == "rawsubabs-nonself-average") regular = 9;
  else if (cluster_regular_ == "logsubabs-overall-average") regular = 10;
  else if (cluster_regular_ == "logsubabs-nonself-average") regular = 11;
  else if (cluster_regular_ == "expsubabs-overall-average") regular = 12;
  else if (cluster_regular_ == "expsubabs-nonself-average") regular = 13;
  ClusterRegularForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterRegularForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int scatter_regular_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          Dtype buffer_diffs = *avgsum_datpt - *avgsum_datit;
          *middle_datpt += 2 * (buffer_datum + buffer_diffs * *numidx_datpt / *numidx_datit);
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
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          *middle_datpt += 2 * buffer_datum * (1 + *numidx_datpt) / *numidx_datit;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum;
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
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          *middle_datpt += 2 * buffer_datum / *numidx_datit;
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1);
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
      const Dtype* bottom_labpt = bottom_label + outer_index * label_numb_; //bottom label pointer
                  *middle_datpt = *middle_difpt = 0;
      for (int label_index = 0; label_index < label_numb_; ++label_index, ++bottom_labpt) {
        if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
        if (static_cast<int>(*bottom_labpt) < 0) continue;
        const int numidx_drift = label_index * label_nmax_ + static_cast<int>(*bottom_labpt);
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          *middle_datpt += 2 * buffer_datum / (buffer_datum * buffer_datum + 1) / *numidx_datit;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 6) {
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          *middle_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 7) {
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          *middle_datpt += 2 * buffer_datum * exp(-buffer_datum * buffer_datum) / *numidx_datit;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 8) {
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 9) {
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign / *numidx_datit;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 10) {
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign / (abs(buffer_datum) + 1);
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 11) {
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign / (abs(buffer_datum) + 1) / *numidx_datit;
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 12) {
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
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *bottom_datpt;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign * exp(-abs(buffer_datum));
          *middle_difpt += 1;
        }
      }
    }
  }
  else if (scatter_regular_ == 13) {
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
        const int avgsum_drift = inner_index + inner_numb_ * numidx_drift;
        const Dtype* numidx_datit = numidx_datum + numidx_drift; //num iterator
        const Dtype* avgsum_datit = avgsum_datum + avgsum_drift; //avg iterator
        if (static_cast<int>(*numidx_datit) < 1) continue;
        for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index) {
          if (labmx_index == static_cast<int>(*bottom_labpt)) continue;
          const int numidx_shift = label_index * label_nmax_ + labmx_index;
          const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
          const Dtype* numidx_datpt = numidx_datum + numidx_shift; //num pointer
          const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift; //avg pointer
          if (static_cast<int>(*numidx_datpt) < 1) continue;
          Dtype buffer_datum = *avgsum_datpt - *avgsum_datit;
          Dtype buffer_dsign = buffer_datum < 0 ? -1 : (buffer_datum > 0 ? 1 : 0);
          *middle_datpt += buffer_dsign * exp(-abs(buffer_datum)) / *numidx_datit;
          *middle_difpt += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::ScatterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  Dtype* middle_datum = middle_blob_.mutable_gpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_gpu_diff();
  int regular = 0;
       if (scatter_regular_ == "rawsubsqr-samples-cluster") regular = 0;
  else if (scatter_regular_ == "rawsubsqr-cluster-cluster") regular = 1;
  else if (scatter_regular_ == "rawsubsqr-samples-average") regular = 2;
  else if (scatter_regular_ == "rawsubsqr-average-average") regular = 3;
  else if (scatter_regular_ == "logsubsqr-samples-average") regular = 4;
  else if (scatter_regular_ == "logsubsqr-average-average") regular = 5;
  else if (scatter_regular_ == "expsubsqr-samples-average") regular = 6;
  else if (scatter_regular_ == "expsubsqr-average-average") regular = 7;
  else if (scatter_regular_ == "rawsubabs-samples-average") regular = 8;
  else if (scatter_regular_ == "rawsubabs-average-average") regular = 9;
  else if (scatter_regular_ == "logsubabs-samples-average") regular = 10;
  else if (scatter_regular_ == "logsubabs-average-average") regular = 11;
  else if (scatter_regular_ == "expsubabs-samples-average") regular = 12;
  else if (scatter_regular_ == "expsubabs-average-average") regular = 13;
  ScatterRegularForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::ScatterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ScatterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterReshuntForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OdotterReshunt_gpu(const vector<Blob<Dtype>*>& bottom) {
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
  OdotterReshuntForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  mapair_datum,
    mapair_diffs, maprop_datum,
    maprop_diffs, storer_datum,
    storer_diffs, middle_datum,
    middle_diffs
  );
}
template void HomoMeanLossLayer<float>::OdotterReshunt_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OdotterReshunt_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OdotterRegularForMean_gpu_backend(
    const int  match_numb_,      const int  inner_numb_,
    const int  label_numb_,      const int  label_nmax_,
    const int  odotter_regular_, const int* mapair_datum,
    const int* mapair_diffs,     const Dtype* bottom_datum,
    const Dtype* bottom_label,   const Dtype* numidx_datum,
    const Dtype* avgsum_datum,   const Dtype* medial_datum,
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum / (srcbuf_datum * srcbuf_datum + 1) * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum / (trgbuf_datum * trgbuf_datum + 1) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum / (srcbuf_datum * srcbuf_datum + 1) * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum / (trgbuf_datum * trgbuf_datum + 1) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum * exp(-srcbuf_datum * srcbuf_datum) * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum * exp(-trgbuf_datum * trgbuf_datum) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_diffs = 2 * srcbuf_datum * exp(-srcbuf_datum * srcbuf_datum) * *medial_datpt;
        const Dtype trgbuf_diffs = 2 * trgbuf_datum * exp(-trgbuf_datum * trgbuf_datum) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
      }
    }
  }
  else if (odotter_regular_ == 6) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
      }
    }
  }
  else if (odotter_regular_ == 7) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
      }
    }
  }
  else if (odotter_regular_ == 8) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign / (abs(srcbuf_datum) + 1) * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign / (abs(trgbuf_datum) + 1) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
      }
    }
  }
  else if (odotter_regular_ == 9) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign / (abs(srcbuf_datum) + 1) * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign / (abs(trgbuf_datum) + 1) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
      }
    }
  }
  else if (odotter_regular_ == 10) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_coeff = srclab_check / *numidx_datpt;
        const Dtype trgbuf_coeff = trglab_check / *numidx_datpt;
        const Dtype srcbuf_datum = *srcbot_datpt - *avgsum_datpt;
        const Dtype trgbuf_datum = *trgbot_datpt - *avgsum_datpt;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign * exp(-abs(srcbuf_datum)) * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign * exp(-abs(trgbuf_datum)) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srcbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
        *storer_difpt += trgbuf_diffs - trgbuf_coeff * (srcbuf_diffs + trgbuf_diffs);
      }
    }
  }
  else if (odotter_regular_ == 11) {
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
      const int srclab_shift = label_index + label_numb_ * *mapair_datpt;
      const int trglab_shift = label_index + label_numb_ * *mapair_difpt;
      const int medial_shift = mapair_shift * label_nmax_;
      const Dtype* srcbot_datpt = bottom_datum + srcbot_shift;
      const Dtype* trgbot_datpt = bottom_datum + trgbot_shift;
      const Dtype* srcbot_labpt = bottom_label + srclab_shift;
      const Dtype* trgbot_labpt = bottom_label + trglab_shift;
      const Dtype* medial_datpt = medial_datum + medial_shift;
      const Dtype* medial_difpt = medial_diffs + medial_shift;
            Dtype* storer_datpt = storer_datum + round_index;
            Dtype* storer_difpt = storer_diffs + round_index;
                  *storer_datpt = *storer_difpt = 0;
      for (int labmx_index = 0; labmx_index < label_nmax_; ++labmx_index, ++medial_datpt, ++medial_difpt) {
        if (*medial_datpt * *medial_difpt > 0) continue;
        const bool srclab_check = (static_cast<int>(*srcbot_labpt) == labmx_index);
        const bool trglab_check = (static_cast<int>(*trgbot_labpt) == labmx_index);
        const int numidx_shift = label_index * label_nmax_ + labmx_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* numidx_datpt = numidx_datum + numidx_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype srcbuf_error = (*avgsum_datpt - *srcbot_datpt) / (*numidx_datpt - srclab_check);
        const Dtype trgbuf_error = (*avgsum_datpt - *trgbot_datpt) / (*numidx_datpt - trglab_check);
        const Dtype avgsum_value = *avgsum_datpt + srclab_check * srcbuf_error;
        const Dtype avgsum_volum = *avgsum_datpt + trglab_check * trgbuf_error;
        const Dtype srcbuf_datum = *srcbot_datpt - avgsum_value;
        const Dtype trgbuf_datum = *trgbot_datpt - avgsum_volum;
        const Dtype srcbuf_dsign = srcbuf_datum < 0 ? -1 : (srcbuf_datum > 0 ? 1 : 0);
        const Dtype trgbuf_dsign = trgbuf_datum < 0 ? -1 : (trgbuf_datum > 0 ? 1 : 0);
        const Dtype srcbuf_diffs = srcbuf_dsign * exp(-abs(srcbuf_datum)) * *medial_datpt;
        const Dtype trgbuf_diffs = trgbuf_dsign * exp(-abs(trgbuf_datum)) * *medial_difpt;
        *storer_datpt += srcbuf_diffs - srclab_check * trgbuf_diffs / (*numidx_datpt - trglab_check);
        *storer_difpt += trgbuf_diffs - trglab_check * srcbuf_diffs / (*numidx_datpt - srclab_check);
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::OdotterRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> storer_shape(3);
  storer_shape[0] = match_numb_;
  storer_shape[1] = label_numb_;
  storer_shape[2] = inner_numb_;
  storer_blob_.Reshape(storer_shape);
  const int* mapair_datum = mapair_blob_.gpu_data();
  const int* mapair_diffs = mapair_blob_.gpu_diff();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  const Dtype* medial_datum = medial_blob_.gpu_data();
  const Dtype* medial_diffs = medial_blob_.gpu_diff();
  Dtype* storer_datum = storer_blob_.mutable_gpu_data();
  Dtype* storer_diffs = storer_blob_.mutable_gpu_diff();
  int regular = 0;
       if (odotter_regular_ == "rawsubsqr-overall-average") regular = 0;
  else if (odotter_regular_ == "rawsubsqr-nonself-average") regular = 1;
  else if (odotter_regular_ == "logsubsqr-overall-average") regular = 2;
  else if (odotter_regular_ == "logsubsqr-nonself-average") regular = 3;
  else if (odotter_regular_ == "expsubsqr-overall-average") regular = 4;
  else if (odotter_regular_ == "expsubsqr-nonself-average") regular = 5;
  else if (odotter_regular_ == "rawsubabs-overall-average") regular = 6;
  else if (odotter_regular_ == "rawsubabs-nonself-average") regular = 7;
  else if (odotter_regular_ == "logsubabs-overall-average") regular = 8;
  else if (odotter_regular_ == "logsubabs-nonself-average") regular = 9;
  else if (odotter_regular_ == "expsubabs-overall-average") regular = 10;
  else if (odotter_regular_ == "expsubabs-nonself-average") regular = 11;
  OdotterRegularForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    regular,      mapair_datum,
    mapair_diffs, bottom_datum,
    bottom_label, numidx_datum,
    avgsum_datum, medial_datum,
    medial_diffs, storer_datum,
    storer_diffs
  );
}
template void HomoMeanLossLayer<float>::OdotterRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OdotterRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeRegularForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OvalizeRegular_gpu(const vector<Blob<Dtype>*>& bottom) {
  Dtype* medial_datum = medial_blob_.mutable_gpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_gpu_diff();
  int regular = 0;
       if (ovalize_regular_ == "rawsubsqr-origins-origins") regular = 0;
  else if (ovalize_regular_ == "rawsubsqr-sqroots-sqroots") regular = 1;
  else if (ovalize_regular_ == "logsubsqr-origins-origins") regular = 2;
  else if (ovalize_regular_ == "logsubsqr-sqroots-sqroots") regular = 3;
  else if (ovalize_regular_ == "expsubsqr-origins-origins") regular = 4;
  else if (ovalize_regular_ == "expsubsqr-sqroots-sqroots") regular = 5;
  else if (ovalize_regular_ == "rawsubabs-origins-origins") regular = 6;
  else if (ovalize_regular_ == "rawsubabs-sqroots-sqroots") regular = 7;
  else if (ovalize_regular_ == "logsubabs-origins-origins") regular = 8;
  else if (ovalize_regular_ == "logsubabs-sqroots-sqroots") regular = 9;
  else if (ovalize_regular_ == "expsubabs-origins-origins") regular = 10;
  else if (ovalize_regular_ == "expsubabs-sqroots-sqroots") regular = 11;
  OvalizeRegularForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(match_numb_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    match_numb_,  label_numb_,
    label_nmax_,  regular,
    medial_datum, medial_diffs
  );
}
template void HomoMeanLossLayer<float>::OvalizeRegular_gpu(const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OvalizeRegular_gpu(const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void PredictTestingForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int predict_measure_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= 1 - exp(-abs(buffer_datum));
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::PredictTesting_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[1]]->mutable_gpu_data();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  int measure = 0;
       if (predict_measure_ == "rawsubsqr-overall-average" || predict_measure_ == "rawsubsqr-nonself-average") measure = 0;
  else if (predict_measure_ == "logsubsqr-overall-average" || predict_measure_ == "logsubsqr-nonself-average") measure = 1;
  else if (predict_measure_ == "expsubsqr-overall-average" || predict_measure_ == "expsubsqr-nonself-average") measure = 2;
  else if (predict_measure_ == "rawsubabs-overall-average" || predict_measure_ == "rawsubabs-nonself-average") measure = 3;
  else if (predict_measure_ == "logsubabs-overall-average" || predict_measure_ == "logsubabs-nonself-average") measure = 4;
  else if (predict_measure_ == "expsubabs-overall-average" || predict_measure_ == "expsubabs-nonself-average") measure = 5;
  PredictTestingForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * label_nmax_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, topper_datum
  );
}
template void HomoMeanLossLayer<float>::PredictTesting_gpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::PredictTesting_gpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void PredictMeasureForMean_gpu_backend(
    const int outer_numb_,      const int inner_numb_,
    const int label_numb_,      const int label_nmax_,
    const int predict_measure_, const Dtype* numidx_datum,
    const Dtype* avgsum_datum,  const Dtype* bottom_datum,
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
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
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= buffer_datum * buffer_datum;
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= log(buffer_datum * buffer_datum + 1);
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
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= log(buffer_datum * buffer_datum + 1);
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= 1 - exp(-buffer_datum * buffer_datum);
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
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= 1 - exp(-buffer_datum * buffer_datum);
      }
    }
  }
  else if (predict_measure_ == 6) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= abs(buffer_datum);
      }
    }
  }
  else if (predict_measure_ == 7) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= abs(buffer_datum);
      }
    }
  }
  else if (predict_measure_ == 8) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= log(abs(buffer_datum) + 1);
      }
    }
  }
  else if (predict_measure_ == 9) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= log(abs(buffer_datum) + 1);
      }
    }
  }
  else if (predict_measure_ == 10) {
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
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        Dtype buffer_datum = *bottom_datpt - *avgsum_datpt;
        *topper_datpt -= 1 - exp(-abs(buffer_datum));
      }
    }
  }
  else if (predict_measure_ == 11) {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    CUDA_KERNEL_LOOP(round_index, round_count) {
      const int label_index = round_index % label_numb_;
      const int labmx_index = round_index / label_numb_ % label_nmax_;
      const int outer_index = round_index / label_numb_ / label_nmax_;
      const int botlab_shift = outer_index * label_numb_ + label_index;
      const int numidx_shift = label_index * label_nmax_ + labmx_index;
      const Dtype* bottom_labpt = bottom_label + botlab_shift;
      const Dtype* numidx_datpt = numidx_datum + numidx_shift;
      const bool botlab_check = (static_cast<int>(*bottom_labpt) == labmx_index);
            Dtype* topper_datpt = topper_datum + round_index;
                  *topper_datpt = 0;
      if (static_cast<int>(*bottom_labpt) >= label_nmax_) continue;
      if (static_cast<int>(*bottom_labpt) < 0) continue;
      if (static_cast<int>(*numidx_datpt) < 2) continue;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int avgsum_shift = inner_index + inner_numb_ * numidx_shift;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* avgsum_datpt = avgsum_datum + avgsum_shift;
        const Dtype  avgsum_error = (*avgsum_datpt - *bottom_datpt) / (*numidx_datpt - 1);
        const Dtype  avgsum_value = *avgsum_datpt + avgsum_error * botlab_check;
        Dtype buffer_datum = *bottom_datpt - avgsum_value;
        *topper_datpt -= 1 - exp(-abs(buffer_datum));
      }
    }
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::PredictMeasure_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[1]]->mutable_gpu_data();
  const Dtype* bottom_datum = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->gpu_data();
  const Dtype* avgsum_datum = this->blobs_[1]->gpu_data();
  int measure = 0;
       if (predict_measure_ == "rawsubsqr-overall-average") measure = 0;
  else if (predict_measure_ == "rawsubsqr-nonself-average") measure = 1;
  else if (predict_measure_ == "logsubsqr-overall-average") measure = 2;
  else if (predict_measure_ == "logsubsqr-nonself-average") measure = 3;
  else if (predict_measure_ == "expsubsqr-overall-average") measure = 4;
  else if (predict_measure_ == "expsubsqr-nonself-average") measure = 5;
  else if (predict_measure_ == "rawsubabs-overall-average") measure = 6;
  else if (predict_measure_ == "rawsubabs-nonself-average") measure = 7;
  else if (predict_measure_ == "logsubabs-overall-average") measure = 8;
  else if (predict_measure_ == "logsubabs-nonself-average") measure = 9;
  else if (predict_measure_ == "expsubabs-overall-average") measure = 10;
  else if (predict_measure_ == "expsubabs-nonself-average") measure = 11;
  PredictMeasureForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_ * label_nmax_ * label_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,  inner_numb_,
    label_numb_,  label_nmax_,
    measure,      numidx_datum,
    avgsum_datum, bottom_datum,
    bottom_label, topper_datum
  );
}
template void HomoMeanLossLayer<float>::PredictMeasure_gpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void HomoMeanLossLayer<double>::PredictMeasure_gpu(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template <typename Dtype>
__global__ void ClusterBackwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ClusterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
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
  ClusterBackwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          cluster_clipactv_,
    cluster_clipnorm_, cluster_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoMeanLossLayer<float>::ClusterBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ClusterBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void ScatterBackwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::ScatterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
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
  ScatterBackwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          scatter_clipactv_,
    scatter_clipnorm_, scatter_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoMeanLossLayer<float>::ScatterBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::ScatterBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void TopdiffBackwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::TopdiffBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = top[outputs_activate_[0]]->gpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_gpu_diff();
  Dtype  blobal_datum = 0;
  int clipmode = 0;
       if (topdiff_clipmode_ == "sample-norm-based") { clipmode = 0; }
  else if (topdiff_clipmode_ == "blobal-norm-based") { clipmode = 1;
    caffe_gpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs, &blobal_datum);
  }
  else if (topdiff_clipmode_ == "unclipped") { clipmode = 2; }
  TopdiffBackwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          topdiff_clipactv_,
    topdiff_clipnorm_, topdiff_clipprop_,
    topper_diffs,      blobal_datum,
    bottom_diffs
  );
}
template void HomoMeanLossLayer<float>::TopdiffBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::TopdiffBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void OvalizeBackwardForMean_gpu_backend(
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
void HomoMeanLossLayer<Dtype>::OvalizeBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
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
  OvalizeBackwardForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(outer_numb_), CAFFE_CUDA_NUM_THREADS>>>(
    outer_numb_,       inner_numb_,
    clipmode,          ovalize_clipactv_,
    ovalize_clipnorm_, ovalize_clipprop_,
    topper_diffs,      middle_datum,
    middle_diffs,      blobal_datum,
    blobal_diffs,      bottom_diffs
  );
}
template void HomoMeanLossLayer<float>::OvalizeBackward_gpu(const vector<Blob<float>*>& top, const vector<Blob<float>*>& bottom);
template void HomoMeanLossLayer<double>::OvalizeBackward_gpu(const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);

template <typename Dtype>
__global__ void AvgsumForMean_gpu_backend(
    const int outer_numb_,     const int inner_numb_,
    const int label_numb_,     const int label_nmax_,
    const int ignore_label_,   const int target_label_,
    const Dtype* bottom_datum, const Dtype* bottom_label,
    const Dtype* numidx_datum, Dtype* medium_datum,
          Dtype* medium_diffs, Dtype* avgsum_datum) {
  const int round_count = label_numb_ * label_nmax_ * inner_numb_;
  CUDA_KERNEL_LOOP(round_index, round_count) {
    const int inner_index  = round_index % inner_numb_;
    const int labmx_index  = round_index / inner_numb_ % label_nmax_;
    const int label_index  = round_index / inner_numb_ / label_nmax_;
    const int numidx_shift = round_index / inner_numb_;
          Dtype* medium_datpt = medium_datum + round_index;
          Dtype* medium_difpt = medium_diffs + round_index;
    const Dtype* numidx_datpt = numidx_datum + numidx_shift;
          Dtype* avgsum_datpt = avgsum_datum + round_index;
                *medium_datpt = *medium_difpt = 0;
    if (labmx_index != ignore_label_ && labmx_index != target_label_) {
      for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
        const int bottom_shift = outer_index * inner_numb_ + inner_index;
        const int botlab_shift = outer_index * label_numb_ + label_index;
        const Dtype* bottom_datpt = bottom_datum + bottom_shift;
        const Dtype* bottom_labpt = bottom_label + botlab_shift;
        if (static_cast<int>(*bottom_labpt) != labmx_index) continue;
        *medium_datpt += *bottom_datpt;
        *medium_difpt += 1;
      }
      if (static_cast<int>(*medium_difpt) > 0) {
        *avgsum_datpt *= *numidx_datpt / (*numidx_datpt + *medium_difpt);
        *avgsum_datpt += *medium_datpt / (*numidx_datpt + *medium_difpt);
      }
    }
  }
}

template <typename Dtype>
__global__ void NumidxForMean_gpu_backend(
    const int inner_numb_, const int label_numb_,
    const int label_nmax_, const Dtype* medium_diffs,
    Dtype* numidx_datum) {
  const int loops_count = label_numb_ * label_nmax_;
  CUDA_KERNEL_LOOP(loops_index, loops_count) {
          Dtype* numidx_datpt = numidx_datum + loops_index;
    const Dtype* medium_difpt = medium_diffs + inner_numb_ * loops_index;
    *numidx_datpt += *medium_difpt;
  }
}

template <typename Dtype>
void HomoMeanLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (this->phase_ != TRAIN) {
    if (outputs_activate_[1] != -1) {
      PredictTesting_gpu(bottom, top);
    }
    if (outputs_activate_[2] != -1) {
      clustr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_data());
      caffe_gpu_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_gpu_diff());
      ClusterTesting_gpu(bottom); ClusterForward_gpu(top);
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
      ClusupdTesting_gpu(bottom); ClusupdForward_gpu(top);
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
      OvalizeMatcher_gpu(bottom); OdotterTesting_gpu(bottom);
      OvalizeMeasure_gpu(bottom); OvalizeForward_gpu(top);
    }
    return;
  }
  if (!preforward_flag && preforward_tag_) {
    preforward_tag_ = false;
  } else if (preforward_flag && !preforward_tag_) {
    preforward_beg_ = preforward_tag_ = true;
  }
  if ((!average_initmode_ && !preforward_tag_) || (average_initmode_ && preforward_beg_)) {
    const int numidx_count = label_numb_ * label_nmax_;
    const int avgsum_count = inner_numb_ * numidx_count;
    caffe_gpu_set(numidx_count, Dtype(0), this->blobs_[0]->mutable_gpu_data());
    caffe_gpu_set(avgsum_count, Dtype(0), this->blobs_[1]->mutable_gpu_data());
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
  if ((!average_initmode_ && !preforward_tag_) || (average_initmode_ && preforward_tag_)) {
    medium_blob_.ReshapeLike(*this->blobs_[1]);
    const Dtype* bottom_datum = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->gpu_data();
    Dtype* numidx_datum = this->blobs_[0]->mutable_gpu_data();
    Dtype* avgsum_datum = this->blobs_[1]->mutable_gpu_data();
    Dtype* medium_datum = medium_blob_.mutable_gpu_data();
    Dtype* medium_diffs = medium_blob_.mutable_gpu_diff();
    AvgsumForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_ * inner_numb_), CAFFE_CUDA_NUM_THREADS>>>(
      outer_numb_,   inner_numb_,
      label_numb_,   label_nmax_,
      ignore_label_, target_label_,
      bottom_datum,  bottom_label,
      numidx_datum,  medium_datum,
      medium_diffs,  avgsum_datum
    );
    NumidxForMean_gpu_backend<Dtype><<<CAFFE_GET_BLOCKS(label_numb_ * label_nmax_), CAFFE_CUDA_NUM_THREADS>>>(
      inner_numb_, label_numb_,
      label_nmax_, medium_diffs,
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
void HomoMeanLossLayer<Dtype>::Backward_gpu(
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
}

INSTANTIATE_LAYER_GPU_FUNCS(HomoMeanLossLayer);
} // namespace caffe