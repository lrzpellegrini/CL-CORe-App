#ifndef CAFFE_BATCHRENORM_LAYER_HPP_
#define CAFFE_BATCHRENORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Implements a variant of Batch Norm (called Batch Renorm), which is OK also for small minibatch size and non iid cases.
 * This custom layer was implemented by: https://github.com/seokhoonboo/caffe-boo
 * based on [1]
 *
 * [1] S. Ioffe, "Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models",
 *     https://arxiv.org/abs/1702.03275
 *
 * Extra Params:
 *  - iter_size = [default = 1]
 *  - step_to_init [default = 1000] - da che iterazione BatchRenorm inizia a differire da Renorm
 *  - step_to_r_max [default = 10000] - r_max attuale linearmente aumentato da [1, r_max] fornito sotto nel range [step_to_init, step_to_r_max] 
 *  - step_to_d_max [default = 10000] - come precedente
 *  - r_max [default = 3]  come da paper
 *  - d_max [default = 5]  come da paper
 */

	template <typename Dtype>
	class BatchReNormLayer : public Layer<Dtype> {
	public:
		explicit BatchReNormLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BatchReNorm"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> mean_, variance_, temp_, x_norm_,r_,d_;
		bool use_global_stats_;
		Dtype moving_average_fraction_;
		int channels_;
		Dtype eps_;

		Dtype r_max_, d_max_;
		int step_to_init_,step_to_r_max_, step_to_d_max_,iter_size_;
		

		Blob<Dtype> batch_sum_multiplier_;
		Blob<Dtype> num_by_chans_;
		Blob<Dtype> spatial_sum_multiplier_;
	};

}  // namespace caffe

#endif  // CAFFE_BATCHRENORM_LAYER_HPP_