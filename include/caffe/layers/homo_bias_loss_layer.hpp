#ifndef CAFFE_HOMO_BIAS_LOSS_LAYER_HPP_
#define CAFFE_HOMO_BIAS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class HomoBiasLossLayer : public Layer<Dtype> {
public:
  explicit HomoBiasLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        preforward_beg_(false),
        preforward_tag_(false),
        solver_iter_(0) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HomoBiasLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 7; }
  void SolverIterChangedHandle(const void* message);

protected:
  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype ClusterNormalizer();
  virtual Dtype ScatterNormalizer();
  virtual Dtype ClusupdNormalizer();
  virtual Dtype ScatupdNormalizer();
  virtual Dtype OvalizeNormalizer();
  virtual void ClusterForward_cpu(const vector<Blob<Dtype>*>& top);
  virtual void ClusterForward_gpu(const vector<Blob<Dtype>*>& top);
  virtual void ScatterForward_cpu(const vector<Blob<Dtype>*>& top);
  virtual void ScatterForward_gpu(const vector<Blob<Dtype>*>& top);
  virtual void ClusupdForward_cpu(const vector<Blob<Dtype>*>& top);
  virtual void ClusupdForward_gpu(const vector<Blob<Dtype>*>& top);
  virtual void ScatupdForward_cpu(const vector<Blob<Dtype>*>& top);
  virtual void ScatupdForward_gpu(const vector<Blob<Dtype>*>& top);
  virtual void OvalizeForward_cpu(const vector<Blob<Dtype>*>& top);
  virtual void OvalizeForward_gpu(const vector<Blob<Dtype>*>& top);
  virtual void OvalizeMatcher_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeMatcher_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusterMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusterMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusupdMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatupdMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeMeasure_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeMeasure_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusterRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusterRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusupdRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ClusupdRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatupdRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void ScatupdRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterReshunt_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterReshunt_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OdotterRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeRegular_cpu(const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeRegular_gpu(const vector<Blob<Dtype>*>& bottom);
  virtual void PredictMeasure_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void PredictMeasure_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void ClusterBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void ClusterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void ScatterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void TopdiffBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void TopdiffBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void OvalizeBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void ClusupdBackward_cpu();
  virtual void ClusupdBackward_gpu();
  virtual void ScatupdBackward_cpu();
  virtual void ScatupdBackward_gpu();

  /// @copydoc HomoBiasLossLayer
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      const bool preforward_flag);
  virtual void Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag);
  virtual void Backward_gpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom,
      const bool prebackward_flag);

  string predict_measure_;
  string cluster_measure_;
  string scatter_measure_;
  string clusupd_measure_;
  string scatupd_measure_;
  string odotter_measure_;
  string ovalize_measure_;
  string cluster_regular_;
  string scatter_regular_;
  string clusupd_regular_;
  string scatupd_regular_;
  string odotter_regular_;
  string ovalize_regular_;
  string cluster_clipmode_;
  string scatter_clipmode_;
  string topdiff_clipmode_;
  string clusupd_clipmode_;
  string scatupd_clipmode_;
  string ovalize_clipmode_;
  bool cluster_clipactv_;
  bool scatter_clipactv_;
  bool topdiff_clipactv_;
  bool clusupd_clipactv_;
  bool scatupd_clipactv_;
  bool ovalize_clipactv_;
  Dtype cluster_clipnorm_;
  Dtype scatter_clipnorm_;
  Dtype topdiff_clipnorm_;
  Dtype clusupd_clipnorm_;
  Dtype scatupd_clipnorm_;
  Dtype ovalize_clipnorm_;
  Dtype cluster_clipprop_;
  Dtype cluster_lowrprop_;
  Dtype cluster_upprprop_;
  Dtype cluster_feedrate_;
  Dtype cluster_feedsize_;
  Dtype cluster_tradeoff_;
  Dtype scatter_clipprop_;
  Dtype scatter_lowrprop_;
  Dtype scatter_upprprop_;
  Dtype scatter_feedrate_;
  Dtype scatter_feedsize_;
  Dtype scatter_tradeoff_;
  Dtype topdiff_clipprop_;
  Dtype topdiff_lowrprop_;
  Dtype topdiff_upprprop_;
  Dtype topdiff_feedrate_;
  Dtype topdiff_feedsize_;
  Dtype topdiff_tradeoff_;
  Dtype clusupd_clipprop_;
  Dtype clusupd_lowrprop_;
  Dtype clusupd_upprprop_;
  Dtype clusupd_feedrate_;
  Dtype clusupd_feedsize_;
  Dtype clusupd_tradeoff_;
  Dtype scatupd_clipprop_;
  Dtype scatupd_lowrprop_;
  Dtype scatupd_upprprop_;
  Dtype scatupd_feedrate_;
  Dtype scatupd_feedsize_;
  Dtype scatupd_tradeoff_;
  Dtype ovalize_clipprop_;
  Dtype ovalize_lowrprop_;
  Dtype ovalize_upprprop_;
  Dtype ovalize_feedrate_;
  Dtype ovalize_feedsize_;
  Dtype ovalize_tradeoff_;
  Dtype ovals2s_01stprop_;
  Dtype ovals2s_02ndprop_;
  Dtype ovalt2t_01stprop_;
  Dtype ovalt2t_02ndprop_;
  Dtype ovals2t_01stprop_;
  Dtype ovals2t_02ndprop_;
  int cluster_interval_;
  int scatter_interval_;
  int topdiff_interval_;
  int clusupd_interval_;
  int scatupd_interval_;
  int ovalize_interval_;
  int cluster_postpone_;
  int scatter_postpone_;
  int topdiff_postpone_;
  int clusupd_postpone_;
  int scatupd_postpone_;
  int ovalize_postpone_;
  int cluster_duration_;
  int scatter_duration_;
  int topdiff_duration_;
  int clusupd_duration_;
  int scatupd_duration_;
  int ovalize_duration_;
  HomoBiasLossParameter_NormalizationMode cluster_normalization_;
  HomoBiasLossParameter_NormalizationMode scatter_normalization_;
  HomoBiasLossParameter_NormalizationMode clusupd_normalization_;
  HomoBiasLossParameter_NormalizationMode scatupd_normalization_;
  HomoBiasLossParameter_NormalizationMode ovalize_normalization_;
  bool biashit_initmode_;
  int match_numb_;
  int label_nmax_;
  int label_axis_;
  int target_label_;
  int ignore_label_;
  vector<int> outputs_activate_;

  bool preforward_beg_;
  bool preforward_tag_;
  int outer_numb_;
  int inner_numb_;
  int label_numb_;
  int solver_iter_;

  Blob<int>   mapidx_blob_;
  Blob<int>   mapair_blob_;
  Blob<Dtype> maprop_blob_;
  Blob<Dtype> middle_blob_;
  Blob<Dtype> medium_blob_;
  Blob<Dtype> medial_blob_;
  Blob<Dtype> storer_blob_;
  Blob<Dtype> caches_blob_;

  Blob<Dtype> clustr_blob_;
  Blob<Dtype> scattr_blob_;
  Blob<Dtype> clusup_blob_;
  Blob<Dtype> scatup_blob_;
  Blob<Dtype> ovaliz_blob_;

  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;
};
} // namespace caffe
#endif // CAFFE_HOMO_BIAS_LOSS_LAYER_HPP_