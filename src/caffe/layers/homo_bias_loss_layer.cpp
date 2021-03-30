#include "caffe/filler.hpp"
#include "caffe/layers/homo_bias_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/messenger.hpp"

namespace caffe {

template <typename Dtype>
class SolverIterChangedHandlerForHomoBiasLossLayer : public Listener {
public:
  SolverIterChangedHandlerForHomoBiasLossLayer(HomoBiasLossLayer<Dtype>* homo_bias_loss_layer)
    : homo_bias_loss_layer_(homo_bias_loss_layer) {
  }
  void handle(const void* message) {
    homo_bias_loss_layer_->SolverIterChangedHandle(message);
  }
private:
  HomoBiasLossLayer<Dtype>* homo_bias_loss_layer_;
};

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  HomoBiasLossParameter homo_bias_loss_param = this->layer_param_.homo_bias_loss_param();
  predict_measure_  = homo_bias_loss_param.predict_measure();
  cluster_measure_  = homo_bias_loss_param.cluster_measure();
  scatter_measure_  = homo_bias_loss_param.scatter_measure();
  clusupd_measure_  = homo_bias_loss_param.clusupd_measure();
  scatupd_measure_  = homo_bias_loss_param.scatupd_measure();
  odotter_measure_  = homo_bias_loss_param.odotter_measure();
  ovalize_measure_  = homo_bias_loss_param.ovalize_measure();
  cluster_regular_  = homo_bias_loss_param.cluster_regular();
  scatter_regular_  = homo_bias_loss_param.scatter_regular();
  clusupd_regular_  = homo_bias_loss_param.clusupd_regular();
  scatupd_regular_  = homo_bias_loss_param.scatupd_regular();
  odotter_regular_  = homo_bias_loss_param.odotter_regular();
  ovalize_regular_  = homo_bias_loss_param.ovalize_regular();
  cluster_clipmode_ = homo_bias_loss_param.cluster_clipmode();
  scatter_clipmode_ = homo_bias_loss_param.scatter_clipmode();
  topdiff_clipmode_ = homo_bias_loss_param.topdiff_clipmode();
  clusupd_clipmode_ = homo_bias_loss_param.clusupd_clipmode();
  scatupd_clipmode_ = homo_bias_loss_param.scatupd_clipmode();
  ovalize_clipmode_ = homo_bias_loss_param.ovalize_clipmode();
  cluster_clipactv_ = homo_bias_loss_param.cluster_clipactv();
  scatter_clipactv_ = homo_bias_loss_param.scatter_clipactv();
  topdiff_clipactv_ = homo_bias_loss_param.topdiff_clipactv();
  clusupd_clipactv_ = homo_bias_loss_param.clusupd_clipactv();
  scatupd_clipactv_ = homo_bias_loss_param.scatupd_clipactv();
  ovalize_clipactv_ = homo_bias_loss_param.ovalize_clipactv();
  cluster_clipnorm_ = homo_bias_loss_param.cluster_clipnorm();
  scatter_clipnorm_ = homo_bias_loss_param.scatter_clipnorm();
  topdiff_clipnorm_ = homo_bias_loss_param.topdiff_clipnorm();
  clusupd_clipnorm_ = homo_bias_loss_param.clusupd_clipnorm();
  scatupd_clipnorm_ = homo_bias_loss_param.scatupd_clipnorm();
  ovalize_clipnorm_ = homo_bias_loss_param.ovalize_clipnorm();
  cluster_clipprop_ = homo_bias_loss_param.cluster_clipprop();
  cluster_lowrprop_ = homo_bias_loss_param.cluster_lowrprop();
  cluster_upprprop_ = homo_bias_loss_param.cluster_upprprop();
  cluster_feedrate_ = homo_bias_loss_param.cluster_feedrate();
  cluster_feedsize_ = homo_bias_loss_param.cluster_feedsize();
  cluster_tradeoff_ = homo_bias_loss_param.cluster_tradeoff();
  scatter_clipprop_ = homo_bias_loss_param.scatter_clipprop();
  scatter_lowrprop_ = homo_bias_loss_param.scatter_lowrprop();
  scatter_upprprop_ = homo_bias_loss_param.scatter_upprprop();
  scatter_feedrate_ = homo_bias_loss_param.scatter_feedrate();
  scatter_feedsize_ = homo_bias_loss_param.scatter_feedsize();
  scatter_tradeoff_ = homo_bias_loss_param.scatter_tradeoff();
  topdiff_clipprop_ = homo_bias_loss_param.topdiff_clipprop();
  topdiff_lowrprop_ = homo_bias_loss_param.topdiff_lowrprop();
  topdiff_upprprop_ = homo_bias_loss_param.topdiff_upprprop();
  topdiff_feedrate_ = homo_bias_loss_param.topdiff_feedrate();
  topdiff_feedsize_ = homo_bias_loss_param.topdiff_feedsize();
  topdiff_tradeoff_ = homo_bias_loss_param.topdiff_tradeoff();
  clusupd_clipprop_ = homo_bias_loss_param.clusupd_clipprop();
  clusupd_lowrprop_ = homo_bias_loss_param.clusupd_lowrprop();
  clusupd_upprprop_ = homo_bias_loss_param.clusupd_upprprop();
  clusupd_feedrate_ = homo_bias_loss_param.clusupd_feedrate();
  clusupd_feedsize_ = homo_bias_loss_param.clusupd_feedsize();
  clusupd_tradeoff_ = homo_bias_loss_param.clusupd_tradeoff();
  scatupd_clipprop_ = homo_bias_loss_param.scatupd_clipprop();
  scatupd_lowrprop_ = homo_bias_loss_param.scatupd_lowrprop();
  scatupd_upprprop_ = homo_bias_loss_param.scatupd_upprprop();
  scatupd_feedrate_ = homo_bias_loss_param.scatupd_feedrate();
  scatupd_feedsize_ = homo_bias_loss_param.scatupd_feedsize();
  scatupd_tradeoff_ = homo_bias_loss_param.scatupd_tradeoff();
  ovalize_clipprop_ = homo_bias_loss_param.ovalize_clipprop();
  ovalize_lowrprop_ = homo_bias_loss_param.ovalize_lowrprop();
  ovalize_upprprop_ = homo_bias_loss_param.ovalize_upprprop();
  ovalize_feedrate_ = homo_bias_loss_param.ovalize_feedrate();
  ovalize_feedsize_ = homo_bias_loss_param.ovalize_feedsize();
  ovalize_tradeoff_ = homo_bias_loss_param.ovalize_tradeoff();
  ovals2s_01stprop_ = homo_bias_loss_param.ovals2s_01stprop();
  ovals2s_02ndprop_ = homo_bias_loss_param.ovals2s_02ndprop();
  ovalt2t_01stprop_ = homo_bias_loss_param.ovalt2t_01stprop();
  ovalt2t_02ndprop_ = homo_bias_loss_param.ovalt2t_02ndprop();
  ovals2t_01stprop_ = homo_bias_loss_param.ovals2t_01stprop();
  ovals2t_02ndprop_ = homo_bias_loss_param.ovals2t_02ndprop();
  cluster_interval_ = homo_bias_loss_param.cluster_interval();
  scatter_interval_ = homo_bias_loss_param.scatter_interval();
  topdiff_interval_ = homo_bias_loss_param.topdiff_interval();
  clusupd_interval_ = homo_bias_loss_param.clusupd_interval();
  scatupd_interval_ = homo_bias_loss_param.scatupd_interval();
  ovalize_interval_ = homo_bias_loss_param.ovalize_interval();
  cluster_postpone_ = homo_bias_loss_param.cluster_postpone();
  scatter_postpone_ = homo_bias_loss_param.scatter_postpone();
  topdiff_postpone_ = homo_bias_loss_param.topdiff_postpone();
  clusupd_postpone_ = homo_bias_loss_param.clusupd_postpone();
  scatupd_postpone_ = homo_bias_loss_param.scatupd_postpone();
  ovalize_postpone_ = homo_bias_loss_param.ovalize_postpone();
  cluster_duration_ = homo_bias_loss_param.cluster_duration();
  scatter_duration_ = homo_bias_loss_param.scatter_duration();
  topdiff_duration_ = homo_bias_loss_param.topdiff_duration();
  clusupd_duration_ = homo_bias_loss_param.clusupd_duration();
  scatupd_duration_ = homo_bias_loss_param.scatupd_duration();
  ovalize_duration_ = homo_bias_loss_param.ovalize_duration();
  cluster_normalization_ = homo_bias_loss_param.cluster_normalization();
  scatter_normalization_ = homo_bias_loss_param.scatter_normalization();
  clusupd_normalization_ = homo_bias_loss_param.clusupd_normalization();
  scatupd_normalization_ = homo_bias_loss_param.scatupd_normalization();
  ovalize_normalization_ = homo_bias_loss_param.ovalize_normalization();
  biashit_initmode_ = homo_bias_loss_param.biashit_initmode();
  match_numb_ = homo_bias_loss_param.match_numb();
  label_nmax_ = homo_bias_loss_param.label_nmax();
  label_axis_ = bottom[0]->CanonicalAxisIndex(homo_bias_loss_param.label_axis());
  outer_numb_ = bottom[0]->count(0, label_axis_);
  inner_numb_ = bottom[0]->count() / outer_numb_;
  label_numb_ = bottom[1]->count() / outer_numb_;
  match_numb_ = match_numb_ ? match_numb_ : outer_numb_;
  target_label_ = homo_bias_loss_param.target_label();
  ignore_label_ = homo_bias_loss_param.ignore_label();
  cluster_clipprop_ = std::max(Dtype(1e-7), cluster_clipprop_ / ClusterNormalizer());
  scatter_clipprop_ = std::max(Dtype(1e-7), scatter_clipprop_ / ScatterNormalizer());
  clusupd_clipprop_ = std::max(Dtype(1e-7), clusupd_clipprop_ / ClusupdNormalizer());
  scatupd_clipprop_ = std::max(Dtype(1e-7), scatupd_clipprop_ / ScatupdNormalizer());
  ovalize_clipprop_ = std::max(Dtype(1e-7), ovalize_clipprop_ / OvalizeNormalizer());
  int blobs_count = 0;
  outputs_activate_.push_back((homo_bias_loss_param.replica_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.predict_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.cluster_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.scatter_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.clusupd_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.scatupd_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  outputs_activate_.push_back((homo_bias_loss_param.ovalize_activate() && blobs_count < top.size()) ? blobs_count++ : -1);
  CHECK(predict_measure_ == "rawsubsqr-sample-biases" || predict_measure_ == "logsubsqr-sample-biases" ||
        predict_measure_ == "expsubsqr-sample-biases" || predict_measure_ == "rawsubabs-sample-biases" ||
        predict_measure_ == "logsubabs-sample-biases" || predict_measure_ == "expsubabs-sample-biases")
      << "illegal predict measure: " << predict_measure_ << "!";
  CHECK(cluster_measure_ == "rawsubsqr-sample-biases" || cluster_measure_ == "logsubsqr-sample-biases" ||
        cluster_measure_ == "expsubsqr-sample-biases" || cluster_measure_ == "rawsubabs-sample-biases" ||
        cluster_measure_ == "logsubabs-sample-biases" || cluster_measure_ == "expsubabs-sample-biases")
      << "illegal cluster measure: " << cluster_measure_ << "!";
  CHECK(scatter_measure_ == "rawsubsqr-sample-biases" || scatter_measure_ == "rawsubsqr-biases-biases" ||
        scatter_measure_ == "logsubsqr-sample-biases" || scatter_measure_ == "logsubsqr-biases-biases" ||
        scatter_measure_ == "expsubsqr-sample-biases" || scatter_measure_ == "expsubsqr-biases-biases" ||
        scatter_measure_ == "rawsubabs-sample-biases" || scatter_measure_ == "rawsubabs-biases-biases" ||
        scatter_measure_ == "logsubabs-sample-biases" || scatter_measure_ == "logsubabs-biases-biases" ||
        scatter_measure_ == "expsubabs-sample-biases" || scatter_measure_ == "expsubabs-biases-biases")
      << "illegal scatter measure: " << scatter_measure_ << "!";
  CHECK(clusupd_measure_ == "rawsubsqr-sample-biases" || clusupd_measure_ == "logsubsqr-sample-biases" ||
        clusupd_measure_ == "expsubsqr-sample-biases" || clusupd_measure_ == "rawsubabs-sample-biases" ||
        clusupd_measure_ == "logsubabs-sample-biases" || clusupd_measure_ == "expsubabs-sample-biases")
      << "illegal clusupd measure: " << clusupd_measure_ << "!";
  CHECK(scatupd_measure_ == "rawsubsqr-sample-biases" || scatupd_measure_ == "rawsubsqr-biases-biases" ||
        scatupd_measure_ == "logsubsqr-sample-biases" || scatupd_measure_ == "logsubsqr-biases-biases" ||
        scatupd_measure_ == "expsubsqr-sample-biases" || scatupd_measure_ == "expsubsqr-biases-biases" ||
        scatupd_measure_ == "rawsubabs-sample-biases" || scatupd_measure_ == "rawsubabs-biases-biases" ||
        scatupd_measure_ == "logsubabs-sample-biases" || scatupd_measure_ == "logsubabs-biases-biases" ||
        scatupd_measure_ == "expsubabs-sample-biases" || scatupd_measure_ == "expsubabs-biases-biases")
      << "illegal scatupd measure: " << scatupd_measure_ << "!";
  CHECK(odotter_measure_ == "rawsubsqr-sample-biases" || odotter_measure_ == "logsubsqr-sample-biases" ||
        odotter_measure_ == "expsubsqr-sample-biases" || odotter_measure_ == "rawsubabs-sample-biases" ||
        odotter_measure_ == "logsubabs-sample-biases" || odotter_measure_ == "expsubabs-sample-biases")
      << "illegal odotter measure: " << odotter_measure_ << "!";
  CHECK(ovalize_measure_ == "rawsubsqr-origin-origin" || ovalize_measure_ == "rawsubsqr-sqroot-sqroot" ||
        ovalize_measure_ == "logsubsqr-origin-origin" || ovalize_measure_ == "logsubsqr-sqroot-sqroot" ||
        ovalize_measure_ == "expsubsqr-origin-origin" || ovalize_measure_ == "expsubsqr-sqroot-sqroot" ||
        ovalize_measure_ == "rawsubabs-origin-origin" || ovalize_measure_ == "rawsubabs-sqroot-sqroot" ||
        ovalize_measure_ == "logsubabs-origin-origin" || ovalize_measure_ == "logsubabs-sqroot-sqroot" ||
        ovalize_measure_ == "expsubabs-origin-origin" || ovalize_measure_ == "expsubabs-sqroot-sqroot")
      << "illegal ovalize measure: " << ovalize_measure_ << "!";
  CHECK(cluster_regular_ == "rawsubsqr-sample-biases" || cluster_regular_ == "logsubsqr-sample-biases" ||
        cluster_regular_ == "expsubsqr-sample-biases" || cluster_regular_ == "rawsubabs-sample-biases" ||
        cluster_regular_ == "logsubabs-sample-biases" || cluster_regular_ == "expsubabs-sample-biases")
      << "illegal cluster regular: " << cluster_regular_ << "!";
  CHECK(scatter_regular_ == "rawsubsqr-sample-biases" || scatter_regular_ == "logsubsqr-sample-biases" ||
        scatter_regular_ == "expsubsqr-sample-biases" || scatter_regular_ == "rawsubabs-sample-biases" ||
        scatter_regular_ == "logsubabs-sample-biases" || scatter_regular_ == "expsubabs-sample-biases")
      << "illegal scatter regular: " << scatter_regular_ << "!";
  CHECK(clusupd_regular_ == "rawsubsqr-sample-biases" || clusupd_regular_ == "logsubsqr-sample-biases" ||
        clusupd_regular_ == "expsubsqr-sample-biases" || clusupd_regular_ == "rawsubabs-sample-biases" ||
        clusupd_regular_ == "logsubabs-sample-biases" || clusupd_regular_ == "expsubabs-sample-biases")
      << "illegal clusupd regular: " << clusupd_regular_ << "!";
  CHECK(scatupd_regular_ == "rawsubsqr-sample-biases" || scatupd_regular_ == "logsubsqr-sample-biases" ||
        scatupd_regular_ == "expsubsqr-sample-biases" || scatupd_regular_ == "rawsubabs-sample-biases" ||
        scatupd_regular_ == "logsubabs-sample-biases" || scatupd_regular_ == "expsubabs-sample-biases")
      << "illegal scatupd regular: " << scatupd_regular_ << "!";
  CHECK(odotter_regular_ == "rawsubsqr-sample-biases" || odotter_regular_ == "logsubsqr-sample-biases" ||
        odotter_regular_ == "expsubsqr-sample-biases" || odotter_regular_ == "rawsubabs-sample-biases" ||
        odotter_regular_ == "logsubabs-sample-biases" || odotter_regular_ == "expsubabs-sample-biases")
      << "illegal odotter regular: " << odotter_regular_ << "!";
  CHECK(ovalize_regular_ == "rawsubsqr-origin-origin" || ovalize_regular_ == "rawsubsqr-sqroot-sqroot" ||
        ovalize_regular_ == "logsubsqr-origin-origin" || ovalize_regular_ == "logsubsqr-sqroot-sqroot" ||
        ovalize_regular_ == "expsubsqr-origin-origin" || ovalize_regular_ == "expsubsqr-sqroot-sqroot" ||
        ovalize_regular_ == "rawsubabs-origin-origin" || ovalize_regular_ == "rawsubabs-sqroot-sqroot" ||
        ovalize_regular_ == "logsubabs-origin-origin" || ovalize_regular_ == "logsubabs-sqroot-sqroot" ||
        ovalize_regular_ == "expsubabs-origin-origin" || ovalize_regular_ == "expsubabs-sqroot-sqroot")
      << "illegal ovalize regular: " << ovalize_regular_ << "!";
  CHECK(cluster_clipmode_ == "sample-diff-based" || cluster_clipmode_ == "sample-norm-based" ||
        cluster_clipmode_ == "blobal-diff-based" || cluster_clipmode_ == "blobal-norm-based" ||
        cluster_clipmode_ == "unclipped")
      << "illegal cluster clipmode: " << cluster_clipmode_ << "!";
  CHECK(scatter_clipmode_ == "sample-diff-based" || scatter_clipmode_ == "sample-norm-based" ||
        scatter_clipmode_ == "blobal-diff-based" || scatter_clipmode_ == "blobal-norm-based" ||
        scatter_clipmode_ == "unclipped")
      << "illegal scatter clipmode: " << scatter_clipmode_ << "!";
  CHECK(topdiff_clipmode_ == "sample-norm-based" || topdiff_clipmode_ == "blobal-norm-based" ||
        topdiff_clipmode_ == "unclipped")
      << "illegal topdiff clipmode: " << topdiff_clipmode_ << "!";
  CHECK(clusupd_clipmode_ == "biases-norm-based" || clusupd_clipmode_ == "blobal-norm-based" ||
        clusupd_clipmode_ == "unclipped")
      << "illegal clusupd clipmode: " << clusupd_clipmode_ << "!";
  CHECK(scatupd_clipmode_ == "biases-norm-based" || scatupd_clipmode_ == "blobal-norm-based" ||
        scatupd_clipmode_ == "unclipped")
      << "illegal scatupd clipmode: " << scatupd_clipmode_ << "!";
  CHECK(ovalize_clipmode_ == "sample-diff-based" || ovalize_clipmode_ == "sample-norm-based" ||
        ovalize_clipmode_ == "blobal-diff-based" || ovalize_clipmode_ == "blobal-norm-based" ||
        ovalize_clipmode_ == "unclipped")
      << "illegal ovalize clipmode: " << ovalize_clipmode_ << "!";
  CHECK(outputs_activate_[0] != -1 ||
       (cluster_clipmode_ != "sample-diff-based" && cluster_clipmode_ != "blobal-diff-based"))
      << "when replica activate: false, illegal cluster clipmode: " << cluster_clipmode_ << "!";
  CHECK(outputs_activate_[0] != -1 ||
       (scatter_clipmode_ != "sample-diff-based" && scatter_clipmode_ != "blobal-diff-based"))
      << "when replica activate: false, illegal scatter clipmode: " << scatter_clipmode_ << "!";
  CHECK(outputs_activate_[0] != -1 || topdiff_clipprop_ == Dtype(0))
      << "when replica activate: false, topdiff clipprop should be set zero!";
  CHECK(outputs_activate_[0] != -1)
      << "Because of some reasons, we now only set replica activate: true!";
  CHECK(0 < label_nmax_)
      << "illegal label nmax: " << label_nmax_ << "!";
  if (this->blobs_.size()) {
    CHECK_EQ(2, this->blobs_.size()) << "Incorrect number of parameter blobs.";
    vector<int> numidx_shape(2);
    vector<int> biases_shape(3);
    numidx_shape[0] = label_numb_;
    numidx_shape[1] = label_nmax_;
    biases_shape[0] = label_numb_;
    biases_shape[1] = label_nmax_;
    biases_shape[2] = inner_numb_;
    if (numidx_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> numidx_shaped_blob(numidx_shape);
      LOG(FATAL) << "Incorrect numidx shape: expected shape "
        << numidx_shaped_blob.shape_string() << "; instead, shape was "
        << this->blobs_[0]->shape_string();
    }
    if (biases_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> biases_shaped_blob(biases_shape);
      LOG(FATAL) << "Incorrect biases shape: expected shape "
        << biases_shaped_blob.shape_string() << "; instead, shape was "
        << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {
    vector<int> numidx_shape(2);
    vector<int> biases_shape(3);
    numidx_shape[0] = label_numb_;
    numidx_shape[1] = label_nmax_;
    biases_shape[0] = label_numb_;
    biases_shape[1] = label_nmax_;
    biases_shape[2] = inner_numb_;
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(numidx_shape)); //data: num & diff: idx
    this->blobs_[1].reset(new Blob<Dtype>(biases_shape)); //data: avg & diff: sum
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(homo_bias_loss_param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  SyncMessenger::AddListener(
    "Any", "Any", "Solver", "Any", "SOLVER_ITER_CHANGED", 1, 0, 1,
    new SolverIterChangedHandlerForHomoBiasLossLayer<Dtype>(this)
  );
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (outputs_activate_[0] != -1) {
    top[outputs_activate_[0]]->ReshapeLike(*bottom[0]);
    top[outputs_activate_[0]]->ShareData(*bottom[0]);
  }
  if (outputs_activate_[1] != -1) {
    vector<int> topper_shape(3);
    topper_shape[0] = outer_numb_;
    topper_shape[1] = label_nmax_;
    topper_shape[2] = label_numb_;
    top[outputs_activate_[1]]->Reshape(topper_shape);
  }
  if (outputs_activate_[2] != -1) {
    vector<int> topper_shape(0);
    top[outputs_activate_[2]]->Reshape(topper_shape);
  }
  if (outputs_activate_[3] != -1) {
    vector<int> topper_shape(0);
    top[outputs_activate_[3]]->Reshape(topper_shape);
  }
  if (outputs_activate_[4] != -1) {
    vector<int> topper_shape(2);
    topper_shape[0] = label_numb_;
    topper_shape[1] = label_nmax_;
    top[outputs_activate_[4]]->Reshape(topper_shape);
  }
  if (outputs_activate_[5] != -1) {
    vector<int> topper_shape(2);
    topper_shape[0] = label_numb_;
    topper_shape[1] = label_nmax_;
    top[outputs_activate_[5]]->Reshape(topper_shape);
  }
  if (outputs_activate_[6] != -1) {
    vector<int> topper_shape(0);
    top[outputs_activate_[6]]->Reshape(topper_shape);
  }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::SolverIterChangedHandle(const void* message) {
  solver_iter_ = *(static_cast<const int*>(message));
  HomoBiasLossParameter homo_bias_loss_param = this->layer_param_.homo_bias_loss_param();
  if (!homo_bias_loss_param.has_cluster_clipprop()) {
    Dtype height   = cluster_upprprop_ - cluster_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / cluster_feedsize_);
    cluster_clipprop_ = cluster_tradeoff_ * log(cluster_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + cluster_lowrprop_;
    cluster_clipprop_ += (1 - cluster_tradeoff_) * (2 / (1 + exp(-cluster_feedrate_ * progress)) - 1) * height + cluster_lowrprop_;
    cluster_clipprop_ = std::max(Dtype(1e-7), cluster_clipprop_ / ClusterNormalizer());
  }
  if (!homo_bias_loss_param.has_scatter_clipprop()) {
    Dtype height   = scatter_upprprop_ - scatter_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / scatter_feedsize_);
    scatter_clipprop_ = scatter_tradeoff_ * log(scatter_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + scatter_lowrprop_;
    scatter_clipprop_ += (1 - scatter_tradeoff_) * (2 / (1 + exp(-scatter_feedrate_ * progress)) - 1) * height + scatter_lowrprop_;
    scatter_clipprop_ = std::max(Dtype(1e-7), scatter_clipprop_ / ScatterNormalizer());
  }
  if (!homo_bias_loss_param.has_topdiff_clipprop()) {
    Dtype height   = topdiff_upprprop_ - topdiff_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / topdiff_feedsize_);
    topdiff_clipprop_ = topdiff_tradeoff_ * log(topdiff_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + topdiff_lowrprop_;
    topdiff_clipprop_ += (1 - topdiff_tradeoff_) * (2 / (1 + exp(-topdiff_feedrate_ * progress)) - 1) * height + topdiff_lowrprop_;
  }
  if (!homo_bias_loss_param.has_clusupd_clipprop()) {
    Dtype height   = clusupd_upprprop_ - clusupd_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / clusupd_feedsize_);
    clusupd_clipprop_ = clusupd_tradeoff_ * log(clusupd_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + clusupd_lowrprop_;
    clusupd_clipprop_ += (1 - clusupd_tradeoff_) * (2 / (1 + exp(-clusupd_feedrate_ * progress)) - 1) * height + clusupd_lowrprop_;
    clusupd_clipprop_ = std::max(Dtype(1e-7), clusupd_clipprop_ / ClusupdNormalizer());
  }
  if (!homo_bias_loss_param.has_scatupd_clipprop()) {
    Dtype height   = scatupd_upprprop_ - scatupd_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / scatupd_feedsize_);
    scatupd_clipprop_ = scatupd_tradeoff_ * log(scatupd_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + scatupd_lowrprop_;
    scatupd_clipprop_ += (1 - scatupd_tradeoff_) * (2 / (1 + exp(-scatupd_feedrate_ * progress)) - 1) * height + scatupd_lowrprop_;
    scatupd_clipprop_ = std::max(Dtype(1e-7), scatupd_clipprop_ / ScatupdNormalizer());
  }
  if (!homo_bias_loss_param.has_ovalize_clipprop()) {
    Dtype height   = ovalize_upprprop_ - ovalize_lowrprop_;
    Dtype progress = std::min(Dtype(1), solver_iter_ / ovalize_feedsize_);
    ovalize_clipprop_ = ovalize_tradeoff_ * log(ovalize_feedrate_ * progress * (exp(Dtype(1)) - 1) + 1) * height + ovalize_lowrprop_;
    ovalize_clipprop_ += (1 - ovalize_tradeoff_) * (2 / (1 + exp(-ovalize_feedrate_ * progress)) - 1) * height + ovalize_lowrprop_;
    ovalize_clipprop_ = std::max(Dtype(1e-7), ovalize_clipprop_ / OvalizeNormalizer());
  }
}

template <typename Dtype>
Dtype HomoBiasLossLayer<Dtype>::ClusterNormalizer() {
  Dtype normalizer;
  switch (cluster_normalization_) {
    case HomoBiasLossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_numb_ * inner_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown cluster normalization mode: "
          << HomoBiasLossParameter_NormalizationMode_Name(cluster_normalization_);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
Dtype HomoBiasLossLayer<Dtype>::ScatterNormalizer() {
  Dtype normalizer;
  switch (scatter_normalization_) {
    case HomoBiasLossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_numb_ * inner_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown scatter normalization mode: "
          << HomoBiasLossParameter_NormalizationMode_Name(scatter_normalization_);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
Dtype HomoBiasLossLayer<Dtype>::ClusupdNormalizer() {
  Dtype normalizer;
  switch (clusupd_normalization_) {
    case HomoBiasLossParameter_NormalizationMode_FULL:
      normalizer = Dtype(label_numb_ * label_nmax_ * inner_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(label_numb_ * label_nmax_);
      break;
    case HomoBiasLossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown clusupd normalization mode: "
          << HomoBiasLossParameter_NormalizationMode_Name(clusupd_normalization_);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
Dtype HomoBiasLossLayer<Dtype>::ScatupdNormalizer() {
  Dtype normalizer;
  switch (scatupd_normalization_) {
    case HomoBiasLossParameter_NormalizationMode_FULL:
      normalizer = Dtype(label_numb_ * label_nmax_ * inner_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(label_numb_ * label_nmax_);
      break;
    case HomoBiasLossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown scatupd normalization mode: "
          << HomoBiasLossParameter_NormalizationMode_Name(scatupd_normalization_);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
Dtype HomoBiasLossLayer<Dtype>::OvalizeNormalizer() {
  Dtype normalizer;
  switch (ovalize_normalization_) {
    case HomoBiasLossParameter_NormalizationMode_FULL:
      normalizer = Dtype(match_numb_ * inner_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(match_numb_);
      break;
    case HomoBiasLossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown ovalize normalization mode: "
          << HomoBiasLossParameter_NormalizationMode_Name(ovalize_normalization_);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusterForward_cpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[2]]->mutable_cpu_data();
  Dtype* clustr_datum = clustr_blob_.mutable_cpu_data();
  Dtype* clustr_diffs = clustr_blob_.mutable_cpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
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
  *topper_datum  = caffe_cpu_asum(inner_numb_, clustr_datum);
  *topper_datum /= inner_numb_;
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ScatterForward_cpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[3]]->mutable_cpu_data();
  Dtype* scattr_datum = scattr_blob_.mutable_cpu_data();
  Dtype* scattr_diffs = scattr_blob_.mutable_cpu_diff();
  const Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  const Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
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
  *topper_datum  = caffe_cpu_asum(inner_numb_, scattr_datum);
  *topper_datum /= inner_numb_;
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::ClusupdForward_cpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[4]]->mutable_cpu_data();
  Dtype* clusup_datum = clusup_blob_.mutable_cpu_data();
  Dtype* clusup_diffs = clusup_blob_.mutable_cpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  const int round_count = label_numb_ * label_nmax_;
  for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatupdForward_cpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[5]]->mutable_cpu_data();
  Dtype* scatup_datum = scatup_blob_.mutable_cpu_data();
  Dtype* scatup_diffs = scatup_blob_.mutable_cpu_diff();
  const Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  const Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  const int round_count = label_numb_ * label_nmax_;
  for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::OvalizeForward_cpu(const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[6]]->mutable_cpu_data();
  Dtype* ovaliz_datum = ovaliz_blob_.mutable_cpu_data();
  Dtype* ovaliz_diffs = ovaliz_blob_.mutable_cpu_diff();
  const Dtype* caches_datum = caches_blob_.mutable_cpu_data();
  const Dtype* caches_diffs = caches_blob_.mutable_cpu_diff();
  for (int label_index = 0; label_index < label_numb_; ++label_index) {
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
  *topper_datum  = caffe_cpu_asum(label_numb_, ovaliz_datum);
  *topper_datum /= label_numb_;
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::OvalizeMatcher_cpu(const vector<Blob<Dtype>*>& bottom) {
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
  int* mapidx_datum = mapidx_blob_.mutable_cpu_data();
  int* mapidx_diffs = mapidx_blob_.mutable_cpu_diff();
  int* mapair_datum = mapair_blob_.mutable_cpu_data();
  int* mapair_diffs = mapair_blob_.mutable_cpu_diff();
  Dtype* maprop_datum = maprop_blob_.mutable_cpu_data();
  Dtype* maprop_diffs = maprop_blob_.mutable_cpu_diff();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  for (int label_index = 0; label_index < label_numb_; ++label_index) {
    *(mapair_datum + label_index) = rand();
    *(mapair_diffs + label_index) = rand();
  }
  for (int label_index = 0; label_index < label_numb_; ++label_index) {
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
      if (ovall_count) srand(*mapair_datpt);
      if (ovs2s_count || ovs2t_count) {
        for (int mapidx_datid = mapidx_datnc - 1; mapidx_datid > 0; --mapidx_datid) {
          int* mapidx_srcpt = mapidx_datpt - label_numb_ * (mapidx_datnc - mapidx_datid);
          int* mapidx_trgpt = mapidx_datpt - label_numb_ * (mapidx_datnc - rand() % (mapidx_datid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      if (ovt2t_count) {
        for (int mapidx_difid = mapidx_difnc - 1; mapidx_difid > 0; --mapidx_difid) {
          int* mapidx_srcpt = mapidx_difpt - label_numb_ * (mapidx_difnc - mapidx_difid);
          int* mapidx_trgpt = mapidx_difpt - label_numb_ * (mapidx_difnc - rand() % (mapidx_difid + 1));
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
      if (ovall_count) srand(*mapair_difpt);
      if (ovs2s_count) {
        for (int mapidx_datid = mapidx_datnc - 1; mapidx_datid > 0; --mapidx_datid) {
          int* mapidx_srcpt = mapidx_datpt - label_numb_ * (mapidx_datnc - mapidx_datid);
          int* mapidx_trgpt = mapidx_datpt - label_numb_ * (mapidx_datnc - rand() % (mapidx_datid + 1));
          int  mapidx_value = *mapidx_srcpt;
          *mapidx_srcpt = *mapidx_trgpt;
          *mapidx_trgpt =  mapidx_value;
        }
      }
      if (ovt2t_count || ovs2t_count) {
        for (int mapidx_difid = mapidx_difnc - 1; mapidx_difid > 0; --mapidx_difid) {
          int* mapidx_srcpt = mapidx_difpt - label_numb_ * (mapidx_difnc - mapidx_difid);
          int* mapidx_trgpt = mapidx_difpt - label_numb_ * (mapidx_difnc - rand() % (mapidx_difid + 1));
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
void HomoBiasLossLayer<Dtype>::ClusterMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  if (cluster_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_measure_ == "logsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_measure_ == "expsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_measure_ == "rawsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_measure_ == "logsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_measure_ == "expsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatterMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  if (scatter_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "rawsubsqr-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "logsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "logsubsqr-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "expsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "expsubsqr-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "rawsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "rawsubabs-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "logsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "logsubabs-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "expsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_measure_ == "expsubabs-biases-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ClusupdMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  if (clusupd_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_measure_ == "logsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_measure_ == "expsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_measure_ == "rawsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_measure_ == "logsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_measure_ == "expsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatupdMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  if (scatupd_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "rawsubsqr-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "logsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "logsubsqr-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "expsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "expsubsqr-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "rawsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "rawsubabs-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "logsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "logsubabs-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "expsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_measure_ == "expsubabs-biases-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::OdotterMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> medial_shape(3);
  medial_shape[0] = match_numb_;
  medial_shape[1] = label_numb_;
  medial_shape[2] = label_nmax_;
  medial_blob_.Reshape(medial_shape);
  const int* mapair_datum = mapair_blob_.cpu_data();
  const int* mapair_diffs = mapair_blob_.cpu_diff();
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* medial_datum = medial_blob_.mutable_cpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_cpu_diff();
  if (odotter_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_measure_ == "logsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_measure_ == "expsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_measure_ == "rawsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_measure_ == "logsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_measure_ == "expsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::OvalizeMeasure_cpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> caches_shape(2);
  caches_shape[0] = match_numb_;
  caches_shape[1] = label_numb_;
  caches_blob_.Reshape(caches_shape);
  const Dtype* medial_datum = medial_blob_.cpu_data();
  const Dtype* medial_diffs = medial_blob_.cpu_diff();
  Dtype* caches_datum = caches_blob_.mutable_cpu_data();
  Dtype* caches_diffs = caches_blob_.mutable_cpu_diff();
  if (ovalize_measure_ == "rawsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "rawsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "logsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "logsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "expsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "expsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "rawsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "rawsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "logsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "logsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "expsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_measure_ == "expsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ClusterRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  if (cluster_regular_ == "rawsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_regular_ == "logsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_regular_ == "expsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_regular_ == "rawsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_regular_ == "logsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (cluster_regular_ == "expsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatterRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  if (scatter_regular_ == "rawsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_regular_ == "logsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_regular_ == "expsubsqr-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_regular_ == "rawsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_regular_ == "logsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatter_regular_ == "expsubabs-sample-biases") {
    const int round_count = outer_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ClusupdRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  if (clusupd_regular_ == "rawsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_regular_ == "logsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_regular_ == "expsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_regular_ == "rawsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_regular_ == "logsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_regular_ == "expsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatupdRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  medium_blob_.ReshapeLike(*this->blobs_[1]);
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  Dtype* medium_diffs = medium_blob_.mutable_cpu_diff();
  if (scatupd_regular_ == "rawsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_regular_ == "logsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_regular_ == "expsubsqr-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_regular_ == "rawsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_regular_ == "logsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_regular_ == "expsubabs-sample-biases") {
    const int round_count = label_numb_ * label_nmax_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::OdotterReshunt_cpu(const vector<Blob<Dtype>*>& bottom) {
  middle_blob_.ReshapeLike(*bottom[0]);
  const int* mapair_datum = mapair_blob_.cpu_data();
  const int* mapair_diffs = mapair_blob_.cpu_diff();
  const Dtype* maprop_datum = maprop_blob_.cpu_data();
  const Dtype* maprop_diffs = maprop_blob_.cpu_diff();
  const Dtype* storer_datum = storer_blob_.cpu_data();
  const Dtype* storer_diffs = storer_blob_.cpu_diff();
  Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  Dtype* middle_diffs = middle_blob_.mutable_cpu_diff();
  caffe_set(outer_numb_ * inner_numb_, Dtype(0), middle_datum);
  caffe_set(outer_numb_ * inner_numb_, Dtype(0), middle_diffs);
  for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
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
void HomoBiasLossLayer<Dtype>::OdotterRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  vector<int> storer_shape(3);
  storer_shape[0] = match_numb_;
  storer_shape[1] = label_numb_;
  storer_shape[2] = inner_numb_;
  storer_blob_.Reshape(storer_shape);
  const int* mapair_datum = mapair_blob_.cpu_data();
  const int* mapair_diffs = mapair_blob_.cpu_diff();
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  const Dtype* medial_datum = medial_blob_.cpu_data();
  const Dtype* medial_diffs = medial_blob_.cpu_diff();
  Dtype* storer_datum = storer_blob_.mutable_cpu_data();
  Dtype* storer_diffs = storer_blob_.mutable_cpu_diff();
  if (odotter_regular_ == "rawsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_regular_ == "logsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_regular_ == "expsubsqr-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_regular_ == "rawsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_regular_ == "logsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (odotter_regular_ == "expsubabs-sample-biases") {
    const int round_count = match_numb_ * label_numb_ * inner_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::OvalizeRegular_cpu(const vector<Blob<Dtype>*>& bottom) {
  Dtype* medial_datum = medial_blob_.mutable_cpu_data();
  Dtype* medial_diffs = medial_blob_.mutable_cpu_diff();
  if (ovalize_regular_ == "rawsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "rawsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "logsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "logsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "expsubsqr-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "expsubsqr-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "rawsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "rawsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "logsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "logsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "expsubabs-origin-origin") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (ovalize_regular_ == "expsubabs-sqroot-sqroot") {
    const int round_count = match_numb_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::PredictMeasure_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* topper_datum = top[outputs_activate_[1]]->mutable_cpu_data();
  const Dtype* bottom_datum = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* numidx_datum = this->blobs_[0]->cpu_data();
  const Dtype* biases_datum = this->blobs_[1]->cpu_data();
  if (predict_measure_ == "rawsubsqr-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (predict_measure_ == "logsubsqr-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (predict_measure_ == "expsubsqr-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (predict_measure_ == "rawsubabs-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (predict_measure_ == "logsubabs-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (predict_measure_ == "expsubabs-sample-biases") {
    const int round_count = outer_numb_ * label_nmax_ * label_numb_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ClusterBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->cpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  const Dtype* middle_diffs = middle_blob_.cpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_cpu_diff();
  if (cluster_clipmode_ == "sample-diff-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (cluster_clipmode_ == "sample-norm-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (cluster_clipmode_ == "blobal-diff-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = caffe_cpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs);
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = cluster_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (cluster_clipmode_ == "blobal-norm-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = cluster_clipnorm_ * cluster_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = cluster_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * cluster_clipprop_) : 0) : cluster_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (cluster_clipmode_ == "unclipped") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
void HomoBiasLossLayer<Dtype>::ScatterBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->cpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  const Dtype* middle_diffs = middle_blob_.cpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_cpu_diff();
  if (scatter_clipmode_ == "sample-diff-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (scatter_clipmode_ == "sample-norm-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (scatter_clipmode_ == "blobal-diff-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = caffe_cpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs);
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = scatter_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (scatter_clipmode_ == "blobal-norm-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = scatter_clipnorm_ * scatter_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = scatter_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatter_clipprop_) : 0) : scatter_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (scatter_clipmode_ == "unclipped") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
void HomoBiasLossLayer<Dtype>::TopdiffBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = top[outputs_activate_[0]]->cpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_cpu_diff();
  if (topdiff_clipmode_ == "sample-norm-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (topdiff_clipmode_ == "blobal-norm-based") {
    Dtype sumsqr_diffs = topdiff_clipnorm_ * topdiff_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs);
    const Dtype coeffi_alpha = topdiff_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * topdiff_clipprop_) : 0) : topdiff_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* topper_difpt = topper_diffs + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *topper_difpt;
        ++topper_difpt; ++bottom_difpt;
      }
    }
  }
  else if (topdiff_clipmode_ == "unclipped") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
void HomoBiasLossLayer<Dtype>::OvalizeBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* topper_diffs = outputs_activate_[0] != -1 ?
                          top[outputs_activate_[0]]->cpu_diff() : NULL;
        Dtype* middle_datum = middle_blob_.mutable_cpu_data();
  const Dtype* middle_diffs = middle_blob_.cpu_diff();
  Dtype* bottom_diffs = bottom[0]->mutable_cpu_diff();
  if (ovalize_clipmode_ == "sample-diff-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (ovalize_clipmode_ == "sample-norm-based") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
  else if (ovalize_clipmode_ == "blobal-diff-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = caffe_cpu_dot(outer_numb_ * inner_numb_, topper_diffs, topper_diffs);
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = ovalize_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (ovalize_clipmode_ == "blobal-norm-based") {
    caffe_div_nz(outer_numb_ * inner_numb_, middle_datum, middle_diffs, middle_datum);
    Dtype sumsqr_diffs = ovalize_clipnorm_ * ovalize_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(outer_numb_ * inner_numb_, middle_datum, middle_datum);
    const Dtype coeffi_alpha = ovalize_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * ovalize_clipprop_) : 0) : ovalize_clipprop_;
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
            Dtype* bottom_difpt = bottom_diffs + outer_index * inner_numb_;
      const Dtype* middle_datpt = middle_datum + outer_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *bottom_difpt += coeffi_alpha * *middle_datpt;
        ++middle_datpt; ++bottom_difpt;
      }
    }
  }
  else if (ovalize_clipmode_ == "unclipped") {
    for (int outer_index = 0; outer_index < outer_numb_; ++outer_index) {
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
void HomoBiasLossLayer<Dtype>::ClusupdBackward_cpu() {
        Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  const Dtype* medium_diffs = medium_blob_.cpu_diff();
  Dtype* biases_diffs = this->blobs_[1]->mutable_cpu_diff();
  if (clusupd_clipmode_ == "biases-norm-based") {
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (clusupd_clipmode_ == "blobal-norm-based") {
    caffe_div_nz(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_diffs, medium_datum);
    Dtype sumsqr_diffs = clusupd_clipnorm_ * clusupd_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_datum);
    const Dtype coeffi_alpha = clusupd_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * clusupd_clipprop_) : 0) : clusupd_clipprop_;
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += coeffi_alpha * *medium_datpt;
        ++medium_datpt; ++biases_difpt;
      }
    }
  }
  else if (clusupd_clipmode_ == "unclipped") {
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::ScatupdBackward_cpu() {
        Dtype* medium_datum = medium_blob_.mutable_cpu_data();
  const Dtype* medium_diffs = medium_blob_.cpu_diff();
  Dtype* biases_diffs = this->blobs_[1]->mutable_cpu_diff();
  if (scatupd_clipmode_ == "biases-norm-based") {
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  else if (scatupd_clipmode_ == "blobal-norm-based") {
    caffe_div_nz(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_diffs, medium_datum);
    Dtype sumsqr_diffs = scatupd_clipnorm_ * scatupd_clipnorm_;
    Dtype sumsqr_datum = caffe_cpu_dot(label_numb_ * label_nmax_ * inner_numb_, medium_datum, medium_datum);
    const Dtype coeffi_alpha = scatupd_clipactv_ || sumsqr_diffs < sumsqr_datum ?
      (0 < sumsqr_datum ? (sqrt(sumsqr_diffs / sumsqr_datum) * scatupd_clipprop_) : 0) : scatupd_clipprop_;
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
            Dtype* biases_difpt = biases_diffs + round_index * inner_numb_;
      const Dtype* medium_datpt = medium_datum + round_index * inner_numb_;
      for (int inner_index = 0; inner_index < inner_numb_; ++inner_index) {
        *biases_difpt += coeffi_alpha * *medium_datpt;
        ++medium_datpt; ++biases_difpt;
      }
    }
  }
  else if (scatupd_clipmode_ == "unclipped") {
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
void HomoBiasLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const bool preforward_flag) {
  if (this->phase_ != TRAIN) {
    if (outputs_activate_[1] != -1) {
      PredictMeasure_cpu(bottom, top);
    }
    if (outputs_activate_[2] != -1) {
      clustr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_cpu_data());
      caffe_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_cpu_diff());
      ClusterMeasure_cpu(bottom); ClusterForward_cpu(top);
    }
    if (outputs_activate_[3] != -1) {
      scattr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_cpu_data());
      caffe_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_cpu_diff());
      ScatterMeasure_cpu(bottom); ScatterForward_cpu(top);
    }
    if (outputs_activate_[4] != -1) {
      clusup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_cpu_data());
      caffe_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_cpu_diff());
      ClusupdMeasure_cpu(bottom); ClusupdForward_cpu(top);
    }
    if (outputs_activate_[5] != -1) {
      scatup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_cpu_data());
      caffe_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_cpu_diff());
      ScatupdMeasure_cpu(bottom); ScatupdForward_cpu(top);
    }
    if (outputs_activate_[6] != -1) {
      ovaliz_blob_.Reshape(vector<int>(1, label_numb_));
      caffe_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_cpu_data());
      caffe_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_cpu_diff());
      OvalizeMatcher_cpu(bottom); OdotterMeasure_cpu(bottom);
      OvalizeMeasure_cpu(bottom); OvalizeForward_cpu(top);
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
    caffe_set(numidx_count, Dtype(0), this->blobs_[0]->mutable_cpu_data());
    if (outputs_activate_[2] != -1) {
      clustr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_cpu_data());
      caffe_set(clustr_blob_.count(), Dtype(0), clustr_blob_.mutable_cpu_diff());
    }
    if (outputs_activate_[3] != -1) {
      scattr_blob_.Reshape(vector<int>(1, inner_numb_));
      caffe_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_cpu_data());
      caffe_set(scattr_blob_.count(), Dtype(0), scattr_blob_.mutable_cpu_diff());
    }
    if (outputs_activate_[4] != -1) {
      clusup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_cpu_data());
      caffe_set(clusup_blob_.count(), Dtype(0), clusup_blob_.mutable_cpu_diff());
    }
    if (outputs_activate_[5] != -1) {
      scatup_blob_.ReshapeLike(*this->blobs_[1]);
      caffe_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_cpu_data());
      caffe_set(scatup_blob_.count(), Dtype(0), scatup_blob_.mutable_cpu_diff());
    }
    if (outputs_activate_[6] != -1) {
      ovaliz_blob_.Reshape(vector<int>(1, label_numb_));
      caffe_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_cpu_data());
      caffe_set(ovaliz_blob_.count(), Dtype(0), ovaliz_blob_.mutable_cpu_diff());
    }
  }
  if (preforward_beg_) preforward_beg_ = false;
  if ((!biashit_initmode_ && !preforward_tag_) || (biashit_initmode_ && preforward_tag_)) {
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* numidx_datum = this->blobs_[0]->mutable_cpu_data();
    const int round_count = label_numb_ * label_nmax_;
    for (int round_index = 0; round_index < round_count; ++round_index) {
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
  if (outputs_activate_[1] != -1) { PredictMeasure_cpu(bottom, top); }
  if (outputs_activate_[2] != -1 && !preforward_tag_) { ClusterMeasure_cpu(bottom); ClusterForward_cpu(top); }
  if (outputs_activate_[3] != -1 && !preforward_tag_) { ScatterMeasure_cpu(bottom); ScatterForward_cpu(top); }
  if (outputs_activate_[4] != -1 && !preforward_tag_) { ClusupdMeasure_cpu(bottom); ClusupdForward_cpu(top); }
  if (outputs_activate_[5] != -1 && !preforward_tag_) { ScatupdMeasure_cpu(bottom); ScatupdForward_cpu(top); }
  if (outputs_activate_[6] != -1 && !preforward_tag_) { OvalizeMatcher_cpu(bottom); OdotterMeasure_cpu(bottom);
                                                        OvalizeMeasure_cpu(bottom); OvalizeForward_cpu(top); }
}

template <typename Dtype>
void HomoBiasLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,
    const bool prebackward_flag) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << "Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    caffe_set(outer_numb_ * inner_numb_, Dtype(0), bottom[0]->mutable_cpu_diff());
    if (cluster_clipprop_ != Dtype(0) && cluster_interval_ &&
        solver_iter_ % cluster_interval_ >= cluster_postpone_ &&
        solver_iter_ % cluster_interval_ <  cluster_postpone_ + cluster_duration_) {
      ClusterRegular_cpu(bottom); ClusterBackward_cpu(top, bottom);
    }
    if (scatter_clipprop_ != Dtype(0) && scatter_interval_ &&
        solver_iter_ % scatter_interval_ >= scatter_postpone_ &&
        solver_iter_ % scatter_interval_ <  scatter_postpone_ + scatter_duration_) {
      ScatterRegular_cpu(bottom); ScatterBackward_cpu(top, bottom);
    }
    if (ovalize_clipprop_ != Dtype(0) && ovalize_interval_ &&
        solver_iter_ % ovalize_interval_ >= ovalize_postpone_ &&
        solver_iter_ % ovalize_interval_ <  ovalize_postpone_ + ovalize_duration_) {
      string  measure  = odotter_measure_;
      odotter_measure_ = odotter_regular_;
      OvalizeMatcher_cpu(bottom); OdotterMeasure_cpu(bottom);
      OvalizeRegular_cpu(bottom); OdotterRegular_cpu(bottom);
      OdotterReshunt_cpu(bottom); OvalizeBackward_cpu(top, bottom);
      odotter_measure_ = measure;
    }
    if (topdiff_clipprop_ != Dtype(0) && topdiff_interval_ &&
        solver_iter_ % topdiff_interval_ >= topdiff_postpone_ &&
        solver_iter_ % topdiff_interval_ <  topdiff_postpone_ + topdiff_duration_) {
      TopdiffBackward_cpu(top, bottom);
    }
  }
  if (this->param_propagate_down_[1]) {
    caffe_set(label_numb_ * label_nmax_ * inner_numb_, Dtype(0), this->blobs_[1]->mutable_cpu_diff());
    if (clusupd_clipprop_ != Dtype(0) && clusupd_interval_ &&
        solver_iter_ % clusupd_interval_ >= clusupd_postpone_ &&
        solver_iter_ % clusupd_interval_ <  clusupd_postpone_ + clusupd_duration_) {
      ClusupdRegular_cpu(bottom); ClusupdBackward_cpu();
    }
    if (scatupd_clipprop_ != Dtype(0) && scatupd_interval_ &&
        solver_iter_ % scatupd_interval_ >= scatupd_postpone_ &&
        solver_iter_ % scatupd_interval_ <  scatupd_postpone_ + scatupd_duration_) {
      ScatupdRegular_cpu(bottom); ScatupdBackward_cpu();
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HomoBiasLossLayer);
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusterForward_gpu(const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatterForward_gpu(const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusupdForward_gpu(const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatupdForward_gpu(const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OvalizeForward_gpu(const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OvalizeMatcher_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatupdMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OdotterMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OvalizeMeasure_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusterRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatterRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusupdRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatupdRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OdotterReshunt_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OdotterRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OvalizeRegular_gpu(const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::PredictMeasure_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatterBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::TopdiffBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::OvalizeBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ClusupdBackward_gpu() { NO_GPU; }
template <typename Dtype> void HomoBiasLossLayer<Dtype>::ScatupdBackward_gpu() { NO_GPU; }
#endif

INSTANTIATE_CLASS(HomoBiasLossLayer);
REGISTER_LAYER_CLASS(HomoBiasLoss);
} // namespace caffe