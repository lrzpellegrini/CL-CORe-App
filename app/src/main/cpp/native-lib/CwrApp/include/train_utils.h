#ifndef CAFFE_CWR_TO_CPP_TRAIN_UTILS_H
#define CAFFE_CWR_TO_CPP_TRAIN_UTILS_H

#include <memory>
#include <vector>
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>

void log_android_debug(std::stringstream &debug_stream);

void extract_minibatch_size_from_prototxt_with_input_layers(
        const boost::shared_ptr<caffe::Net<float>>& train_net,
        const boost::shared_ptr<caffe::Net<float>>& test_net,
        int* train_minibatch_size,
        int* test_minibatch_size);

bool net_use_target_vectors(const boost::shared_ptr<caffe::Net<float>>& net);

void get_data(const std::string& fpath,
              const std::string& bpath,
              std::vector<cv::Mat>& x,
              std::vector<int>& y,
              const cv::Mat &mean_image);

void get_data(const std::string &fpath,
              const std::string &bpath,
              std::vector<cv::Mat> &x,
              std::vector<int> &y,
              int resizeW, int resizeH,
              const cv::Mat &mean_image);

std::vector<std::string> read_lines(const std::string& file_path);
std::vector<std::pair<std::string, int>> read_file_list(const std::string& file_path);
int count_lines(const std::string& file_path);

void preprocess_image(cv::Mat& img, const cv::Mat &mean_image);

void pad_data(std::vector<cv::Mat>& x,
              std::vector<int>& y,
              int& it,
              int mb_size);

template <class T>
int pad_data_single(std::vector<T> &x, int mb_size) {
    // computing test_iters
    int n_missing = x.size() % mb_size;
    int surplus;
    if (n_missing > 0) {
        surplus = 1;
    } else {
        surplus = 0;

    }
    int it = (x.size() / mb_size) + surplus;

    // padding data to fix batch dimentions

    if (n_missing > 0) {
        int n_to_add = mb_size - n_missing;
        while(n_to_add > 0) {
            int addition = n_to_add % x.size();
            std::vector<T> x_insert(&x[0], &x[addition]);

            x.insert(x.begin(), x_insert.begin(), x_insert.end());
            n_to_add -= addition;
        }
    }

    return it;
}

std::vector<int> count_lines_in_batches(int batch_count, const std::string& fpath);

std::vector<std::vector<float>> compute_one_hot_vectors_from_class_count(const std::vector<int>& y, int class_count);
std::vector<std::vector<float>> compute_one_hot_vectors(
        const std::vector<int> &y,
        const boost::shared_ptr<caffe::Net<float>> &net,
        const std::string &prediction_layer);
std::vector<std::vector<float>> compute_one_hot_vectors(
        const std::vector<std::shared_ptr<std::vector<int>>> &y,
        const boost::shared_ptr<caffe::Net<float>> &net,
        const std::string &prediction_layer);

void shuffle_in_unison(std::vector<cv::Mat> &train_x,
                       std::vector<int> &train_y,
                       std::vector<std::vector<float>> &target_y,
                       unsigned long seed);

void shuffle_in_unison(std::vector<std::shared_ptr<cv::Mat>> &train_x,
                       std::vector<int> &train_y,
                       std::vector<std::vector<float>> &target_y,
                       unsigned long seed);

void shuffle_in_unison(std::vector<std::shared_ptr<std::vector<float>>> &train_x,
                       std::vector<int> &train_y,
                       std::vector<std::vector<float>> &target_y,
                       unsigned long seed);

void shuffle_in_unison(std::vector<std::shared_ptr<std::vector<float>>> &train_x,
                       std::vector<std::shared_ptr<std::vector<int>>> &train_y,
                       std::vector<std::vector<float>> &target_y,
                       unsigned long seed);

void feed_image_layer(const boost::shared_ptr<caffe::Net<float>>& net,
        const std::string& layer_name,
        const std::vector<cv::Mat>& images,
        int from,
        int to,
        int dest_from = 0);

/*
void feed_image_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                      const std::string& layer_name,
                      const std::vector<std::shared_ptr<cv::Mat>>& images,
                      int from,
                      int to,
                      int dest_from = 0);*/

void feed_feature_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                        const std::string& layer_name,
                        const std::vector<std::shared_ptr<std::vector<float>>>& data,
                        int from,
                        int to,
                        int dest_from = 0);

void feed_label_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                      const std::string& layer_name,
                      const std::vector<float>& labels,
                      int from,
                      int to,
                      int dest_from = 0);

void feed_label_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                      const std::string& layer_name,
                      const std::vector<int>& labels,
                      int from,
                      int to,
                      int dest_from = 0);

void feed_label_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                      const std::string& layer_name,
                      const std::vector<std::shared_ptr<std::vector<int>>>& labels,
                      int from,
                      int to,
                      int dest_from = 0);

void feed_target_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                      const std::string& layer_name,
                      const std::vector<std::vector<float>>& targets,
                      int from,
                      int to,
                      int dest_from = 0);



void test_network_with_accuracy_layer(caffe::Solver<float>* solver,
        const std::vector<cv::Mat>& test_x, const std::vector<int>& test_y,
        int test_iterat, int test_minibatch_size, const std::string& final_prediction_layer,
        double& test_acc, double& test_loss, std::vector<int>& pred_y, bool compute_test_loss = false,
        bool return_prediction = false);

extern cv::Mat core50_mean_image;

void partial_param_update(const boost::shared_ptr<caffe::Net<float>>& net, int from, int to=-1);
void partial_param_update(caffe::Net<float>* net, int from, int to=-1);

void partial_apply_update(caffe::SGDSolver<float>* solver, int from, int to=-1);


// Utils
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    if ( !v.empty() ) {
        out << '[';
        for(int i = 0; i < v.size(); i++) {
            out << v[i];
            if(i != v.size()-1) {
                out << ", ";
            }
        }

        out << "]";
    }
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::set<T>& s) {
    if ( !s.empty() ) {
        out << '[';
        int i = 0;
        for(auto& it : s) {
            out << it;
            if(i != s.size()-1) {
                out << ", ";
            }
            i++;
        }

        out << "]";
    }
    return out;
}

void choose_file_paths_config(const std::string &dataset_variation, const std::string &model, bool reduced_test_set,
        int nicv2_scenario, std::string &selected_configuration, std::string& train_filelists,
        std::string& test_filelist, std::string& db_path, std::string& solver_file_first_batch, std::string& solver_file, std::string& class_labels_txt,
        std::string& tmp_weights_file, std::string& init_weights_file, std::string& cwr_serialization_file,
        std::vector<std::string> &cwr_layers_Model, std::string &prediction_level_Model);

std::string hash_generic(void* pointer, size_t size_in_bytes);

std::string hash_vector(const std::vector<float> &vec);
std::string hash_vector(const std::vector<int> &vec);
std::string hash_vector_as_int64(const std::vector<int> &vec);

std::string hash_rehearsal_memory(const std::vector<std::shared_ptr<std::vector<float>>> &vec);

std::string hash_caffe_blob(const boost::shared_ptr<caffe::Blob<float>> &blob);
std::string hash_caffe_blob(const boost::shared_ptr<caffe::Net<float>> &net, const std::string &blob_name);

std::string hash_net_params(const boost::shared_ptr<caffe::Net<float>>& net);
std::string hash_net(const boost::shared_ptr<caffe::Net<float>>& net, bool include_lrs=true);

std::string hash_net_lrs(const boost::shared_ptr<caffe::Net<float>>& net);

std::vector<float> inference_on_single_pattern(caffe::Solver<float>* solver, const std::string &pattern,
                                          const std::string &final_prediction_layer);

std::unordered_map<std::string, int> generate_layers_name_to_id_mapping(caffe::Net<float>* net);
std::unordered_map<std::string, int> generate_layers_name_to_id_mapping(const boost::shared_ptr<caffe::Net<float>>& net);
std::unordered_map<std::string, int> generate_layers_name_to_param_id_mapping(caffe::Net<float>* net);
std::unordered_map<std::string, int> generate_layers_name_to_param_id_mapping(const boost::shared_ptr<caffe::Net<float>>& net);
int get_layer_id_from_name(const boost::shared_ptr<caffe::Net<float>>& net, const std::string &layer_name);

std::vector<std::shared_ptr<std::vector<float>>> get_layer_features(const boost::shared_ptr<caffe::Net<float>>& net,
                                                                    const std::string& layer_name,
                                                                    int from = 0, int to = -1);

std::vector<float> per_class_accuracy(int classes, const std::vector<int> &pred_y, const std::vector<int> &truth);
bool file_exists(const std::string& name);
std::vector<int> indexes_range(int min, int max, int n_to_generate);

#endif //CAFFE_CWR_TO_CPP_TRAIN_UTILS_H
