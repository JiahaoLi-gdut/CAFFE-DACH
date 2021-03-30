#include <vector>

#include "caffe/layers/class_latch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClassLatchLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (preforward_tag_) {
    srand((unsigned)time(NULL)); latch_flag_ = true;
    // storer_datum -> storer_diffs
    for (int bottom_index = 0; bottom_index < bottom.size(); bottom_index += 2) {
      const int topper_index = bottom_index / 2;
      const int label_axisu = label_axis_[topper_index];
      const int label_nmaxu = label_nmax_[topper_index];
      const int latch_nminu = latch_nmin_[topper_index];
            int latch_numbu = latch_numb_[topper_index];
      const int outer_numbu = bottom[bottom_index]->count(0, label_axisu);
      const int inner_numbu = bottom[bottom_index]->count() / outer_numbu;
      const int label_numbu = bottom[bottom_index + 1]->count() / outer_numbu;
      const int labmx_index = solver_iter_ % label_nmaxu;
      const int label_index = solver_iter_ / label_nmaxu % label_numbu;
      const Dtype* bottom_label = bottom[bottom_index + 1]->cpu_data();
      Dtype* storer_datum = storer_blob_[topper_index]->mutable_gpu_data();
      Dtype* storer_diffs = storer_blob_[topper_index]->mutable_gpu_diff();
      for (int outer_index = 0; outer_index < outer_numbu; ++outer_index) {
        const Dtype* bottom_labpt = bottom_label + outer_index * label_numbu + label_index;
        if (static_cast<int>(*bottom_labpt) == labmx_index) {
          if (latch_numbu < outer_numbu) {
            const Dtype* storer_datpt = storer_datum + outer_index * inner_numbu;
                  Dtype* storer_difpt = storer_diffs + latch_numbu * inner_numbu;
            caffe_copy(inner_numbu, storer_datpt, storer_difpt);
            latch_numb_[topper_index] = ++latch_numbu;
          } else if (rand() % 2) {
            const int latch_index = rand() % latch_numbu;
            const Dtype* storer_datpt = storer_datum + outer_index * inner_numbu;
                  Dtype* storer_difpt = storer_diffs + latch_index * inner_numbu;
            caffe_copy(inner_numbu, storer_datpt, storer_difpt);
          }
        }
      }
      latch_flag_ &= (latch_nminu < latch_numbu);
    }
  }
  if (preforward_flag) {
    // bottom_datum -> (storer_datum and topper_datum)
    for (int bottom_index = 0; bottom_index < bottom.size(); bottom_index += 2) {
      const int topper_index = bottom_index / 2;
      const int label_axisu = label_axis_[topper_index];
      const int outer_numbu = bottom[bottom_index]->count(0, label_axisu);
      const int inner_numbu = bottom[bottom_index]->count() / outer_numbu;
      const Dtype* bottom_datum = bottom[bottom_index]->gpu_data();
      Dtype* storer_datum = storer_blob_[topper_index]->mutable_gpu_data();
      Dtype* topper_datum = top[topper_index]->mutable_gpu_data();
      caffe_copy(outer_numbu * inner_numbu, bottom_datum, storer_datum);
      caffe_copy(outer_numbu * inner_numbu, bottom_datum, topper_datum);
      if (!preforward_tag_) latch_numb_[topper_index] = 0;
    }
    if (!preforward_tag_) preforward_tag_ = true;
  } else {
    // (storer_diffs or bottom_datum) -> topper_datum
    for (int bottom_index = 0; bottom_index < bottom.size(); bottom_index += 2) {
      const int topper_index = bottom_index / 2;
      const int label_axisu = label_axis_[topper_index];
      const int latch_nminu = latch_nmin_[topper_index];
      const int latch_numbu = latch_numb_[topper_index];
      const int outer_numbu = bottom[bottom_index]->count(0, label_axisu);
      const int inner_numbu = bottom[bottom_index]->count() / outer_numbu;
      const Dtype* bottom_datum = bottom[bottom_index]->gpu_data();
      Dtype* storer_diffs = storer_blob_[topper_index]->mutable_gpu_diff();
      Dtype* topper_datum = top[topper_index]->mutable_gpu_data();
      if ((latch_mode_ && latch_flag_) || (!latch_mode_ && latch_nminu < latch_numbu)) {
        for (int outer_index = latch_numbu; outer_index < outer_numbu; ++outer_index) {
          const int latch_index = rand() % latch_numbu;
          const Dtype* storer_difpt = storer_diffs + latch_index * inner_numbu;
                Dtype* storer_difit = storer_diffs + outer_index * inner_numbu;
          caffe_copy(inner_numbu, storer_difpt, storer_difit);
        }
        caffe_copy(outer_numbu * inner_numbu, storer_diffs, topper_datum);
        latch_numb_[topper_index] = outer_numbu;
      } else {
        caffe_copy(outer_numbu * inner_numbu, bottom_datum, topper_datum);
      }
    }
    if (preforward_tag_) preforward_tag_ = false;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClassLatchLayer);
} // namespace caffe