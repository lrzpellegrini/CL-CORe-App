#ifndef CAFFE_MULTINOMIAL_LOGISTIC_LOSS_TARGET_LAYER_HPP_
#define CAFFE_MULTINOMIAL_LOGISTIC_LOSS_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss (a.k.a. cross entropy) for a one-of-many
 *        classification task, by taking target vectors as input.
 *
 * This is an extension to default Caffe layers by Davide Maltoni.
 * With respect to the default version (multinomial logistic loss) here,
 * instead of a single class label (scalar) per pattern,
 * a full probability vector (target vector) is provided for each pattern.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$ the predictions.
 *   -# @f$ (N \times C \times H \times W) @f$ the targets.
  * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$ the computed multinomial logistic loss.
 */
template <typename Dtype>
class MultinomialLogisticLossTargetLayer : public LossLayer<Dtype> {
 public:
  explicit MultinomialLogisticLossTargetLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultinomialLogisticLossTarget"; }

 protected:
  /// @copydoc MultinomialLogisticLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_MULTINOMIAL_LOGISTIC_LOSS_TARGET_LAYER_HPP_
