#ifndef CWR_APP_H
#define CWR_APP_H
#include <iostream>
#include <memory>
#include <visualization.h>
#include <train_utils.h>
#include <cwr.h>
#include <rehearsal_memory.h>
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>
#include <fstream>
#include <chrono>
#include <model_status_reporter.h>
#include <training_status_reporter.h>

// TODO: rearrange methods and fields

typedef std::shared_ptr<std::vector<float>> cwra_rehe_t;

#ifndef __ANDROID__
namespace std {
    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
    typename _Unique_if<T>::_Single_object
    make_unique(Args&&... args) {
        return unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template<class T>
    typename _Unique_if<T>::_Unknown_bound
    make_unique(size_t n) {
        typedef typename remove_extent<T>::type U;
        return unique_ptr<T>(new U[n]());
    }

    template<class T, class... Args>
    typename _Unique_if<T>::_Known_bound
    make_unique(Args&&...) = delete;
}

#endif

/**
 * The class managing the inference and training steps.
 *
 * For the Java-C++ connectors, please refer to "native-lib.cpp".
 */
class CwrApp {
public:
    /**
     * Create a mean image given the mean RGB values.
     * @param G The green channel mean value.
     * @param B The blue channel mean value.
     * @param R The red channel mean value.
     * @return The mean image.
     */
    static cv::Mat create_mean_image(float G, float B, float R);

    /**
     * Check if the Caffe C++ library was initialized.
     *
     * @return True if Caffe was initialized.
     */
    static bool is_caffe_initialized();
    static bool caffe_initialized;

    /**
     * Check if a previous CwrApp state exist at the given folder.
     * @param folder The search folder.
     * @return True if a previous state exist.
     */
    static bool state_exists(const std::string &folder);

    /**
     * Reload the CwrApp from the given folder
     * @param folder The folder in which the app state was previously saved.
     * @return The loaded CwrApp instance.
     */
    static CwrApp read_from_disk(const std::string &folder);

    CwrApp();

    /*
     * Constructor used to create an inference-only version
     */
    CwrApp(const std::string &solver_path,
           const std::string &initial_weights_path,
           int classes_n,
           const std::string &initial_prediction_layer,
           const std::vector<std::string> &intial_labels,
           const cv::Mat& initial_mean_image);
    /*
     * Constructor used to create a training and inference version
     */
    CwrApp(const std::string &solver_path,
           const std::string &initial_weights_path,
           const std::vector<float> &initial_class_updates,
           int train_epochs,
           int initial_classes_n,
           const std::vector<std::string> &cwr_layers,
           const std::string &initial_prediction_layer,
           const std::vector<std::string> &intial_labels,
           const cv::Mat& initial_mean_image,
           const std::string &training_pre_extraction_layer = "",
           const std::string &backward_stop_layer = ""
           );


    /**
     * Get the input layer minibatch size.
     *
     * The size is obtained from the solver definition.
     * In the app this value is 1, as the training minibatch
     * is injected at the latent replay layer.
     *
     * @return The input layer minibatch size. This is 1 by default.
     */
    int get_minibatch_size();

    /**
     * Get the latent features minibatch size.
     *
     * The size is obtained from the solver definition.
     *
     * @return The training minibatch size.
     */
    int get_features_train_minibatch_size();

    /**
     * Get the inference minibatch size.
     *
     * @return The inference minibatch size.
     */
    int get_inference_minibatch_size();

    /**
     * If true, training is disabled.
     *
     * @return True if training is disabled (inference-only)
     */
    bool is_training_disabled() const;

    /**
     * Obtain the list of labels.
     *
     * @return The list of category labels.
     */
    const vector<string> &get_labels() const;

    /**
     * Obtain the mean image used when preprocessing input images.
     *
     * @return The mean image to be subtracted from input images.
     */
    const Mat &get_mean_image() const;

    /**
     * Get the name of the prediction layer.
     *
     * @return The name of the prediction layer.
     */
    const string &get_prediction_layer() const;

    /**
     * Get the amount of training epochs.
     *
     * @return The amount of training epochs.
     */
    int get_epochs() const;



    /**
     * Subtract the mean image from the input one.
     *
     * @param image The input image.
     * @return An image from which the mean image has been subtracted.
     */
    cv::Mat subtract_mean_image(const cv::Mat& image);

    /**
     * Add an image to the training batch.
     *
     * This will also pre-extract the latent activations.
     *
     * @param new_image The image to add.
     */
    void add_batch_image(const cv::Mat& new_image);

    /**
     * Get the Caffe solver.
     *
     * @return The Caffe solver.
     */
    caffe::Solver<float>* get_solver();

    /**
     * Get the Caffe training model.
     *
     * @return The Caffe training model.
     */
    boost::shared_ptr<caffe::Net<float>> get_net();

    /**
     * Get the Caffe inference model.
     *
     * @return The Caffe inference model.
     */
    boost::shared_ptr<caffe::Net<float>> get_test_net();

    /**
     * Remove all instances from the training batch.
     */
    void reset_batch();

    /**
     * Executes the training step.
     *
     * This will use the images added through "add_batch_image".
     *
     * @param label The category label
     * @param reporter The model status reporter. Can be NULL.
     * @param callback The training status reported. Can be NULL.
     */
    void cwr_execute_step(int label,
            ModelStatusReporter* reporter = nullptr,
            TrainingProgressListener* callback = nullptr);

    /**
     * Add a new category.
     *
     * @param category_label The category label.
     * @return The new category ID.
     */
    int add_category(const std::string &category_label);

    /**
     * Get the amount of currently registered categories.
     *
     * @return How many categories have been registered.
     */
    int get_current_categories_count();

    /**
     * Get the maximum supported amount of categories.
     *
     * @return The maximum amount of supported categories.
     */
    int get_max_categories();

    /**
     * Run inference on an image.
     * @param image The input image.
     * @return The prediction scores.
     */
    std::vector<float> inference(const cv::Mat &image);

    /**
     * Run inference on an image.
     *
     * This will ignore the default prediction layer and use the previded one instead.
     * @param image The input image.
     * @param output_layer_name The output layer name.
     * @return The prediction scores.
     */
    std::vector<float> inference(const cv::Mat &image, const std::string &output_layer_name);

    /**
     * Save the current model to disk.
     *
     * This will overwrite the existing model.
     */
    void overwrite_model_on_disk();
    /**
     * Save the current model to disk.
     *
     * @param path The save path.
     */
    void serialize_model_on_disk(const std::string &path);

    /**
     * Saves this app status, including the CWR one.
     *
     * Current model status is saved by using overwrite_model_on_disk(void)
     *
     * @param folder The save folder.
     */
    void save_everything_to_disk(const std::string &folder);

    /**
     * Set the amount of training threads.
     * @param threads The amount of training threads.
     */
    void set_training_threads(int threads);

    /**
     * Set the amount of threads used when extracting features.
     * @param threads The amount of threads used to extract features.
     */
    void set_feature_extraction_threads(int threads);

    /**
     * Set the replay memory buffer.
     *
     * @param new_rehearsal_memory The replay buffer.
     */
    void set_rehearsal_memory(const std::shared_ptr<RehearsalMemory<cwra_rehe_t>> &new_rehearsal_memory);

    /**
     * Get the current replay buffer.
     *
     * @return The current replay buffer.
     */
    std::shared_ptr<RehearsalMemory<cwra_rehe_t>> get_rehearsal_memory();

    /**
     * Set the backward stop layer.
     *
     * @param new_backward_stop_layer The name of the new backward stop layer.
     */
    void set_backward_stop_layer(const std::string &new_backward_stop_layer);

    /**
     * Get the backward stop layer.
     *
     * @return The name of the backward stop layer.
     */
    std::string get_backward_stop_layer();

    /**
     * Set the latent replay layer.
     *
     * @param new_rehearsal_layer The name of the latent replay layer.
     */
    void set_rehearsal_layer(const std::string &new_rehearsal_layer);

    /**
     * Get the latent replay layer.
     *
     * @return The name of the latent replay layer.
     */
    std::string get_rehearsal_layer();

private:
    void preallocate_memory();
    void set_brn_past_weight(float weight);
    bool must_pre_extract_batch_features();
    void cwr_execute_step_with_images(int label,
            ModelStatusReporter* reporter = nullptr,
            TrainingProgressListener* callback = nullptr);
    void cwr_execute_step_with_features(int label,
            ModelStatusReporter* reporter = nullptr,
            TrainingProgressListener* callback = nullptr);

    bool training_disabled = true; // Always false after deserialization (transient)

    std::string solver_path; // Persistent
    std::string weights_path; // Persistent
    caffe::Solver<float>* solver; // Transient
    cv::Mat mean_image; // Persistent
    std::string prediction_layer; // Persistent
    std::string feature_extraction_layer; // Persistent
    int epochs; // Persistent
    std::vector<string> labels; // Persistent
    std::string rehearsal_layer; // Persistent
    std::string backward_stop_layer; // Persistent
    std::shared_ptr<RehearsalMemory<cwra_rehe_t>> rehearsal_memory; // Persistent

    // Transient fields that are regenerated from persistent data
    boost::shared_ptr<caffe::Net<float>> net; // Transient
    boost::shared_ptr<caffe::Net<float>> test_net; // Transient
    Cwr cwr; // Transient
    int minibatch_size; // Transient
    int inference_minibatch_size; // Transient
    std::unordered_map<std::string, int> train_layer_name_to_id; // Transient
    std::unordered_map<std::string, int> test_layer_name_to_id; // Transient
    std::unordered_map<std::string, int> train_layer_names_to_param_id; // Transient

    // Transient field that are not regenerated from persistent data
    std::vector<Mat> training_batch_images; // Transient
    std::vector<std::shared_ptr<std::vector<float>>> training_batch_features; // Transient
    std::unique_ptr<std::recursive_mutex> mtx = std::make_unique<std::recursive_mutex>(); // Transient
    int training_threads = -1; // Transient, default = -1 (doesn't change existing blas values)
    int feature_extraction_threads = -1; // Transient, default = -1 (doesn't change existing blas values)


};

void serialize_string(const std::string &str,
        std::basic_fstream<char, std::char_traits<char>> &out_file);
std::string deserialize_string(std::basic_fstream<char, std::char_traits<char>> &in_file);
void serialize_float_vector(const std::vector<float> &float_vec,
        std::basic_fstream<char, std::char_traits<char>> &out_file);
std::vector<float> deserialize_float_vector(
        std::basic_fstream<char,std::char_traits<char>> &in_file);

void serialize_rehearsal_blob(const std::vector<cwra_rehe_t> &rehe_vec,
                            std::basic_fstream<char, std::char_traits<char>> &out_file);
std::vector<cwra_rehe_t> deserialize_rehearsal_blob(
        std::basic_fstream<char,std::char_traits<char>> &in_file);

void serialize_int_vector(const std::vector<int> &int_vec,
                          std::basic_fstream<char, std::char_traits<char>> &out_file);
std::vector<int> deserialize_int_vector(std::basic_fstream<char, std::char_traits<char>> &in_file);
void serialize_string_vector(const std::vector<std::string> &str_vec,
                             std::basic_fstream<char, std::char_traits<char>> &out_file);
std::vector<std::string> deserialize_string_vector(
        std::basic_fstream<char, std::char_traits<char>> &in_file);
void serialize_Mat(const cv::Mat &image,
        std::basic_fstream<char, std::char_traits<char>> &out_file);
cv::Mat deserialize_Mat(std::basic_fstream<char, std::char_traits<char>> &in_file);

void serialize_uint32t(uint32_t value, std::basic_fstream<char, std::char_traits<char>> &out_file);
uint32_t deserialize_uint32t(std::basic_fstream<char, std::char_traits<char>> &out_file);

std::shared_ptr<RehearsalMemory<cwra_rehe_t>> load_rehe_from_snapshot(std::basic_fstream<char, std::char_traits<char>> &in_stream);
#endif