#include <train_utils.h>
#include <visualization.h>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>
#include <limits>
#include <math.h>
#include <picosha2.h>
#ifdef __ANDROID__
#include <android/log.h>
#include <omp.h>
#else
#include <randomkit.h>
#endif

using namespace std;
using namespace caffe;
using namespace cv;

Mat mean_init();

Mat core50_mean_image = mean_init(); // 128*128*3 BGR (8 bit per canale)

Mat mean_init() {
    Mat mean(128, 128, CV_32FC3);
    auto mean_data = (float*) mean.data;

    for (int size = 0; size < 3 * 128 * 128; size += 3) { // 3 = channels
        mean_data[size + 0] = 104;
        mean_data[size + 1] = 117;
        mean_data[size + 2] = 123;
    }

    return mean;
}

void extract_minibatch_size_from_prototxt_with_input_layers(
        const boost::shared_ptr<Net<float>> &train_net,
        const boost::shared_ptr<Net<float>> &test_net,
        int *train_minibatch_size,
        int *test_minibatch_size) {
    if (train_minibatch_size != nullptr) {
        *train_minibatch_size = train_net->input_blobs()[0]->shape()[0];
    }

    if (test_minibatch_size != nullptr) {
        *test_minibatch_size = test_net->input_blobs()[0]->shape()[0];
    }
}

bool net_use_target_vectors(const boost::shared_ptr<Net<float>> &net) {
    return std::find(net->layer_names().begin(), net->layer_names().end(), "target") != net->layer_names().end();
}

void get_data(const string &fpath,
              const string &bpath,
              vector<Mat> &x,
              vector<int> &y,
              const Mat &mean_image) {
    path fp{fpath};
    path bp{bpath};

    x.clear();
    y.clear();
    auto file_relative_paths = read_file_list(fp.make_preferred().string());
    x.reserve(file_relative_paths.size());
    y.reserve(file_relative_paths.size());

    int loaded_n = 0;

    for (auto &line : file_relative_paths) {
        path image_path = bp / line.first;
        Mat image;
        if(!file_exists(image_path.make_preferred().string())) {
            throw "Image not found";
        }
        Mat imagex = imread(image_path.make_preferred().string(), IMREAD_COLOR); // BGR, 8-bit per canale
        imagex.convertTo(image, CV_32FC3); // BGR, 32 bit (float) per canale
        preprocess_image(image, mean_image);
        x.push_back(image);
        y.push_back(line.second);
        loaded_n++;
    }
}

void get_data(const string &fpath,
              const string &bpath,
              vector<Mat> &x,
              vector<int> &y,
              int resizeW, int resizeH,
              const Mat &mean_image) {
    path fp{fpath};
    path bp{bpath};

    x.clear();
    y.clear();
    auto file_relative_paths = read_file_list(fp.make_preferred().string());
    x.reserve(file_relative_paths.size());
    y.reserve(file_relative_paths.size());

    int loaded_n = 0;

    for (auto &line : file_relative_paths) {
        path image_path = bp / line.first;
        //cout << "Image path:" << image_path.make_preferred().string() << endl;
        if(!file_exists(image_path.make_preferred().string())) {
            throw "Image not found";
        }
        Mat image, resized;
        Mat imagex = imread(image_path.make_preferred().string(), IMREAD_COLOR); // BGR, 8-bit per canale
        imagex.convertTo(image, CV_32FC3); // BGR, 32 bit (float) per canale
        resize(image, resized, Size(resizeW, resizeH));

        //cout << resized.rows << " " << resized.cols << " vs " << mean_image.rows << " " << mean_image.cols << endl;
        preprocess_image(resized, mean_image);
        x.push_back(resized);
        y.push_back(line.second);
        loaded_n++;
    }
}

vector<string> read_lines(const string &file_path) {
    vector<string> result;
    ifstream infile(file_path);
    string line;

    while (getline(infile, line)) {
        result.push_back(line);
    }

    return result;
}

vector<pair<string, int>> read_file_list(const string& file_path) {
    vector<string> lines = read_lines(file_path);
    vector<pair<string, int>> result;
    result.reserve(lines.size());

    for(auto & line : lines) {
        string path, label;
        istringstream f(line);
        getline(f, path, ' ');
        getline(f, label, ' ');

        string platform_path = path;
        std::replace(platform_path.begin(), platform_path.end(), '\\', '/');

        result.emplace_back(platform_path, atoi(label.c_str()));
    }

    return result;
}

int count_lines(const std::string &file_path) {
    int result = 0;
    std::ifstream infile(file_path);
    std::string line;
    while (std::getline(infile, line)) {
        result++;
    }

    return result;
}


void preprocess_image(Mat &img, const Mat &mean_image) {
    //Input = immagine BGR, 3 canali, 32 bit per canale, range 0-255
    //# scale each pixel to 255 img = img * 255 -> No
    //# Swap RGB to BRG img = img[:, :, ::-1] -> No
    img -= mean_image;

    //# Swap channel dimension to fit the caffe format (c, h, w)
    //img = np.transpose(img, (2, 0, 1)) -> Non si può fare!
}

void pad_data(std::vector<cv::Mat> &x, std::vector<int> &y, int &it, int mb_size) {
    // computing test_iters
    int n_missing = x.size() % mb_size;
    int surplus;
    if (n_missing > 0) {
        surplus = 1;
    } else {
        surplus = 0;

    }
    it = ((int) x.size() / mb_size) + surplus;

    // # padding data to fix batch dimentions
    if (n_missing > 0) {
        int n_to_add = mb_size - n_missing;
        while(n_to_add > 0) {
            int addition = n_to_add % x.size();
            vector<Mat> x_insert(&x[0], &x[addition]);
            vector<int> y_insert(&y[0], &y[addition]);

            x.insert(x.begin(), x_insert.begin(), x_insert.end());
            y.insert(y.begin(), y_insert.begin(), y_insert.end());
            n_to_add -= addition;
        }
    }
}

vector<int> count_lines_in_batches(int batch_count, const std::string &fpath) {
    vector<int> lines(batch_count);
    for (int batch = 0; batch < batch_count; batch++) {
        string batch_filelist = fpath;
        string replacement = std::to_string(batch);
        if (replacement.size() < 2) {
            replacement = "0" + replacement;
        }

        boost::replace_all(batch_filelist, "XX", replacement);
        lines[batch] = count_lines(batch_filelist);
    }

    return lines;
}

std::vector<std::vector<float>> compute_one_hot_vectors(
        const std::vector<int> &y,
        const boost::shared_ptr<caffe::Net<float>> &net,
        const std::string &prediction_layer) {
    boost::shared_ptr<Blob<float>> prediction_blob = net->blob_by_name(prediction_layer);
    int class_count = prediction_blob->shape()[1];
    return compute_one_hot_vectors_from_class_count(y, class_count);
}

std::vector<std::vector<float>> compute_one_hot_vectors(
        const std::vector<std::shared_ptr<std::vector<int>>> &y,
        const boost::shared_ptr<caffe::Net<float>> &net,
        const std::string &prediction_layer) {
    std::vector<int> y_data;
    for(int i = 0; i < y.size(); i++) {
        y_data.push_back(y[i]->at(0));
    }

    boost::shared_ptr<Blob<float>> prediction_blob = net->blob_by_name(prediction_layer);
    int class_count = prediction_blob->shape()[1];
    return compute_one_hot_vectors_from_class_count(y_data, class_count);
}

vector<vector<float>> compute_one_hot_vectors_from_class_count(const std::vector<int> &y, int class_count) {
    vector<vector<float>> target_y(y.size(), vector<float>(class_count, 0.0f));
    for (int i = 0; i < y.size(); i++) {
        target_y[i][y[i]] = 1.0f;
    }

    return target_y;
}

#ifdef __ANDROID__
void shuffle_in_unison(vector<std::shared_ptr<vector<float>>> &train_x,
                       vector<int> &train_y,
                       vector<vector<float>> &target_y,
                       unsigned long seed) {
    std::mt19937 generator(seed);

    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = (int) (generator() % i);
        if (i == j) {
            continue;
        }

        std::shared_ptr<vector<float>> tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        int tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        vector<float> tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }
}

void shuffle_in_unison(vector<Mat> &train_x,
                       vector<int> &train_y,
                       vector<vector<float>> &target_y,
                       unsigned long seed) {
    std::mt19937 generator(seed);


    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = (int) (generator() % i);
        if (i == j) {
            continue;
        }

        auto tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        int tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        vector<float> tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }
}
#else
void shuffle_in_unison(vector<Mat> &train_x,
                       vector<int> &train_y,
                       vector<vector<float>> &target_y,
                       unsigned long seed) {
    auto state = (rk_state*) malloc(sizeof(rk_state));
    rk_seed(seed, state);
    /*rk_state* rng_state = (rk_state*) malloc(sizeof(rk_state));
    memcpy(rng_state, state, sizeof(rk_state));*/


    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = rk_interval(i, state);
        if (i == j) {
            continue;
        }

        Mat tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        int tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        vector<float> tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }

    free(state);
}

void shuffle_in_unison(std::vector<std::shared_ptr<cv::Mat>> &train_x,
                       std::vector<int> &train_y,
                       std::vector<std::vector<float>> &target_y,
                       unsigned long seed) {
    auto state = (rk_state*) malloc(sizeof(rk_state));
    rk_seed(seed, state);
    /*rk_state* rng_state = (rk_state*) malloc(sizeof(rk_state));
    memcpy(rng_state, state, sizeof(rk_state));*/


    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = rk_interval(i, state);
        if (i == j) {
            continue;
        }

        auto tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        int tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        vector<float> tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }

    free(state);
}

void shuffle_in_unison(vector<std::shared_ptr<vector<float>>> &train_x,
                       vector<int> &train_y,
                       vector<vector<float>> &target_y,
                       unsigned long seed) {
    auto state = (rk_state*) malloc(sizeof(rk_state));
    rk_seed(seed, state);
    /*rk_state* rng_state = (rk_state*) malloc(sizeof(rk_state));
    memcpy(rng_state, state, sizeof(rk_state));*/


    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = rk_interval(i, state);
        if (i == j) {
            continue;
        }

        auto tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        int tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        vector<float> tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }

    free(state);
}

void shuffle_in_unison(std::vector<std::shared_ptr<std::vector<float>>> &train_x,
                       std::vector<std::shared_ptr<std::vector<int>>> &train_y,
                       std::vector<std::vector<float>> &target_y, unsigned long seed) {
    auto state = (rk_state*) malloc(sizeof(rk_state));
    rk_seed(seed, state);
    /*rk_state* rng_state = (rk_state*) malloc(sizeof(rk_state));
    memcpy(rng_state, state, sizeof(rk_state));*/


    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = train_x.size()-1; i >= 1; i--) {
        int j = rk_interval(i, state);
        if (i == j) {
            continue;
        }

        auto tmpTrainX = train_x[j];
        train_x[j] = train_x[i];
        train_x[i] = tmpTrainX;

        auto tmpTrainY = train_y[j];
        train_y[j] = train_y[i];
        train_y[i] = tmpTrainY;

        auto tmpTargetY = target_y[j];
        target_y[j] = target_y[i];
        target_y[i] = tmpTargetY;
    }

    free(state);
}
#endif

void feed_image_layer(const boost::shared_ptr<caffe::Net<float>> &net, const std::string &layer_name,
                      const std::vector<cv::Mat> &images, int from, int to, int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);
    int width = input_layer->width();
    int height = input_layer->height();
    int offset_range = to - from;

    #pragma omp parallel for
    for (int input_offset = 0; input_offset < offset_range; input_offset++) {
        int i = input_offset + from;

        vector<Mat> input_channels;
        int inputImageOffset = input_layer->offset(input_offset + dest_from, 0, 0, 0);
        float *input_data = &input_layer->mutable_cpu_data()[inputImageOffset];
        for (int j = 0; j < input_layer->channels(); ++j) {
            Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }

        split(images[i], input_channels);
    }
}

void feed_label_layer(const boost::shared_ptr<Net<float>> &net, const string &layer_name,
                      const std::vector<int> &labels, int from, int to, int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);

    for (int i = from, input_offset = 0; i < to; i++, input_offset++) {
        input_layer->mutable_cpu_data()[input_offset + dest_from] = (float) labels[i];
    }
}

void feed_label_layer(const boost::shared_ptr<Net<float>> &net, const string &layer_name,
                      const std::vector<std::shared_ptr<std::vector<int>>> &labels, int from, int to, int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);

    for (int i = from, input_offset = 0; i < to; i++, input_offset++) {
        input_layer->mutable_cpu_data()[input_offset + dest_from] = (float) labels[i]->at(0);
    }
}

void feed_feature_layer(const boost::shared_ptr<caffe::Net<float>>& net,
                        const std::string& layer_name,
                        const std::vector<std::shared_ptr<std::vector<float>>>& data,
                        int from,
                        int to,
                        int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);
    int offset_range = to - from;

    /*#pragma omp parallel for
    for (int input_offset = 0; input_offset < offset_range; input_offset++) {
        int i = input_offset + from;
        std::shared_ptr<std::vector<float>> features = data[i];
        for (int j = 0; j < features->size(); j++) {
            int input_target_offset = input_layer->offset({dest_from + input_offset, j});
            input_layer->mutable_cpu_data()[input_target_offset] = features.get()[0][j];
        }
    }*/

    #pragma omp parallel for
    for (int input_offset = 0; input_offset < offset_range; input_offset++) {
        int i = input_offset + from;
        std::shared_ptr<std::vector<float>> features = data[i];
        float* features_ptr = features.get()->data();
        int input_target_initial_offset = input_layer->offset({dest_from + input_offset, 0});
        float* layer_data = &input_layer->mutable_cpu_data()[input_target_initial_offset];
        memcpy(layer_data, features_ptr, features->size() * sizeof(float));
    }
}

void feed_label_layer(const boost::shared_ptr<Net<float>> &net, const string &layer_name,
                      const vector<float> &labels, int from, int to, int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);

    for (int i = from, input_offset = 0; i < to; i++, input_offset++) {
        input_layer->mutable_cpu_data()[input_offset + dest_from] = labels[i];
    }
}

void feed_target_layer(const boost::shared_ptr<Net<float>> &net, const string &layer_name,
                       const vector<vector<float>> &targets, int from, int to, int dest_from) {
    boost::shared_ptr<Blob<float>> input_layer = net->blob_by_name(layer_name);

    for (int i = from, input_offset = 0; i < to; i++, input_offset++) {
        for (int j = 0; j < targets[i].size(); j++) {
            int input_target_offset = input_layer->offset({input_offset + dest_from, j});
            input_layer->mutable_cpu_data()[input_target_offset] = targets[i][j];
        }
    }
}

void test_network_with_accuracy_layer(Solver<float>* solver,
                                      const vector<Mat> &test_x, const vector<int> &test_y,
                                      int test_iterat, int test_minibatch_size,
                                      const string &final_prediction_layer, double &test_acc, double &test_loss,
                                      vector<int> &pred_y, bool compute_test_loss, bool return_prediction) {
    test_acc = test_loss = 0;
    pred_y = vector<int>(test_y.size(), 0);
    boost::shared_ptr<Net<float>> test_net = solver->test_nets()[0];

    for (int test_it = 0; test_it < test_iterat; test_it++) {
        int start = test_it * test_minibatch_size;
        int end = (test_it + 1) * test_minibatch_size;

        feed_image_layer(test_net, "data", test_x, start, end);
        feed_label_layer(test_net, "label", test_y, start, end);
        test_net->Forward();

        float acc = test_net->blob_by_name("accuracy")->data_at({});
        test_acc += (double) acc;

        if (compute_test_loss) {
            float loss = test_net->blob_by_name("loss")->data_at({});
            test_loss += loss;
        }

        if (return_prediction) {
            for (int i = start; i < end; i++) {
                boost::shared_ptr<Blob<float>> pred_layer = test_net->blob_by_name(final_prediction_layer);

                int max_pred_class = 0;
                float max_pred_val = std::numeric_limits<float>::min();

                int row = i - start;
                for (int column = 0; column < pred_layer->shape(1); column++) {
                    float current_pred_val = pred_layer->data_at({row, column});
                    if (current_pred_val > max_pred_val) {
                        max_pred_val = current_pred_val;
                        max_pred_class = column;
                    }
                }

                pred_y[i] = max_pred_class;
            }
        }
    }

    test_acc /= (double) test_iterat;
    test_loss /= (double) test_iterat;
}

void partial_param_update(const boost::shared_ptr<caffe::Net<float>>& net, int from, int to) {
    partial_param_update(net.get(), from, to);
}

void partial_param_update(caffe::Net<float>* net, int from, int to) {
    if(to < 0) {
        to = net->learnable_params().size();
    }

    for (int i = from; i < to; ++i) {
        net->learnable_params_[i]->Update();
    }
}

void partial_apply_update(caffe::SGDSolver<float>* solver, int from, int to) {
    if(to < 0) {
        to = solver->net_->learnable_params().size();
    }
    float rate = solver->GetLearningRate();

    solver->ClipGradients();
    for (int param_id = from; param_id < to; ++param_id) {
        solver->Normalize(param_id);
        solver->Regularize(param_id);
        solver->ComputeUpdateValue(param_id, rate);
    }
    partial_param_update(solver->net_, from, to);
    //solver->net_->Update();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++solver->iter_;
}

void caffenet_config(const string &dataset_variation, bool reduced_test_set,
                     string &selected_configuration, string &train_filelists, string &test_filelist, string &db_path,
                     string &solver_file_first_batch, string &solver_file,
                     string &class_labels_txt, string &tmp_weights_file,
                     string &init_weights_file, string &cwr_serialization_file,
                     vector<string> &cwr_layers_Model, string &prediction_level_Model) {
    string project_root;
    cwr_layers_Model = {"mid_fc8"};
    prediction_level_Model = "mid_fc8";

    // Mio PC uni
    project_root = "/home/lorenzo/Desktop/caffe_cwr_to_cpp";
    selected_configuration = "PC Uni Lorenzo";
    train_filelists = project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lorenzo/Desktop/preparazione raptor01/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/NIC/NIC1/NIC_solver_CaffeNet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/NIC/NIC1/NIC_solver_CaffeNet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lorenzo/Desktop/preparazione raptor01/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/CaffeNet.caffemodel";
    init_weights_file = project_root + "/models/CaffeNet.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Mio PC casa
    project_root = "/home/lorenzo/Desktop/caffe_cwr_to_cpp";
    selected_configuration = "PC casa Lorenzo";
    train_filelists = project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lorenzo/Desktop/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/NIC/NIC1/NIC_solver_CaffeNet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/NIC/NIC1/NIC_solver_CaffeNet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lorenzo/Desktop/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/CaffeNet.caffemodel";
    init_weights_file = project_root + "/models/CaffeNet.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Raptor01
    project_root = "/home/lpellegrini/remote_mirror/clion_projects/caffe_cwr_to_cpp";
    selected_configuration = "Raptor01 Lorenzo";
    train_filelists = project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lpellegrini/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/NIC/NIC1/NIC_solver_CaffeNet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/NIC/NIC1/NIC_solver_CaffeNet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lpellegrini/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/CaffeNet.caffemodel";
    init_weights_file = project_root + "/models/CaffeNet.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Docker
    project_root = "/opt/project";
    selected_configuration = "Immagine docker";
    train_filelists = project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/NIC_inc1" + dataset_variation + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/NIC/NIC1/NIC_solver_CaffeNet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/NIC/NIC1/NIC_solver_CaffeNet.prototxt"; // conf['solver_file']
    class_labels_txt = "/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/CaffeNet.caffemodel";
    init_weights_file = project_root + "/models/CaffeNet.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    throw std::runtime_error("[caffenet] Can't find paths configuration. This PC is not Lorenzo's PC(s), Raptor01 or Docker image!");
}

void mobilenet_config(const string &dataset_variation, bool reduced_test_set, int nicv2_scenario,
                     string &selected_configuration, string &train_filelists, string &test_filelist, string &db_path,
                     string &solver_file_first_batch, string &solver_file,
                     string &class_labels_txt, string &tmp_weights_file,
                     string &init_weights_file, string &cwr_serialization_file,
                      vector<string> &cwr_layers_Model, string &prediction_level_Model) {
    string project_root;
    cwr_layers_Model = {"mid_fc7"};
    prediction_level_Model = "mid_fc7";
    string filelist_folder_name("sIII_v2_");
    filelist_folder_name += std::to_string(nicv2_scenario);
    filelist_folder_name += dataset_variation;

            // Mio PC uni
    project_root = "/home/lorenzo/Desktop/caffe_cwr_to_cpp";
    selected_configuration = "PC Uni Lorenzo";
    train_filelists = project_root + "/batch_filelists/" + filelist_folder_name + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/" + filelist_folder_name + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lorenzo/Desktop/preparazione raptor01/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/mobilenet/nic_solver_mobilenet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/mobilenet/nic_solver_mobilenet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lorenzo/Desktop/preparazione raptor01/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/MobileNetV1.caffemodel";
    init_weights_file = project_root + "/mobilenet/MobileNetV1_50_classes.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/MobileNetV1cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Mio PC casa
    project_root = "/home/lorenzo/Desktop/caffe_cwr_to_cpp";
    selected_configuration = "PC casa Lorenzo";
    train_filelists = project_root + "/batch_filelists/" + filelist_folder_name + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/" + filelist_folder_name + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lorenzo/Desktop/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/mobilenet/nic_solver_mobilenet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/mobilenet/nic_solver_mobilenet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lorenzo/Desktop/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/MobileNetV1.caffemodel";
    init_weights_file = project_root + "/mobilenet/MobileNetV1_50_classes.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/MobileNetV1cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Raptor01
    project_root = "/home/lpellegrini/remote_mirror/clion_projects/caffe_cwr_to_cpp";
    selected_configuration = "Raptor01 Lorenzo";
    train_filelists = project_root + "/batch_filelists/" + filelist_folder_name + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/" + filelist_folder_name + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/home/lpellegrini/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/mobilenet/nic_solver_mobilenet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/mobilenet/nic_solver_mobilenet.prototxt"; // conf['solver_file']
    class_labels_txt = "/home/lpellegrini/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/MobileNetV1.caffemodel";
    init_weights_file = project_root + "/mobilenet/MobileNetV1_50_classes.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/MobileNetV1cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    // Docker
    project_root = "/opt/project";
    selected_configuration = "Immagine docker";
    train_filelists = project_root + "/batch_filelists/" + filelist_folder_name + "/run0/train_batch_XX_filelist.txt"; // conf['train_filelists']
    test_filelist =  project_root + "/batch_filelists/" + filelist_folder_name + "/run0/test_filelist.txt"; // conf['test_filelist']
    db_path = "/datasets/core50/core50_128x128"; // conf['db_path']
    solver_file_first_batch = project_root + "/mobilenet/nic_solver_mobilenet_first_batch.prototxt"; // conf['solver_file_first_batch']
    solver_file = project_root + "/mobilenet/nic_solver_mobilenet.prototxt"; // conf['solver_file']
    class_labels_txt = "/datasets/core50/core50_labels.txt"; //conf['class_labels']
    tmp_weights_file = project_root + "/tmp_exp_data/MobileNetV1.caffemodel";
    init_weights_file = project_root + "/mobilenet/MobileNetV1_50_classes.caffemodel"; // conf['init_weights_file']
    cwr_serialization_file = project_root + "/tmp_exp_data/MobileNetV1cwrdata";
    if(reduced_test_set) {
        test_filelist =  project_root + "/batch_filelists/test_filelist_20.txt"; // conf['test_filelist']
    }

    if(boost::filesystem::exists(test_filelist) && boost::filesystem::exists(class_labels_txt)) {
        return;
    }

    throw std::runtime_error("[mobilenet] Can't find paths configuration. This PC is not Lorenzo's PC(s), Raptor01 or Docker image!");
}

void choose_file_paths_config(const string &dataset_variation, const string &model, bool reduced_test_set,
                              int nicv2_scenario, string &selected_configuration, string &train_filelists, string &test_filelist, string &db_path,
                              string &solver_file_first_batch, string &solver_file,
                              string &class_labels_txt, string &tmp_weights_file,
                              string &init_weights_file, string &cwr_serialization_file,
                              vector<string> &cwr_layers_Model, string &prediction_level_Model) {
    if(model == "caffenet") {
        cout << "Caffenet is deprecated. This configuration may not work..." << endl;
        caffenet_config(dataset_variation, reduced_test_set, selected_configuration, train_filelists, test_filelist, db_path,
                solver_file_first_batch, solver_file,
                class_labels_txt, tmp_weights_file,
                init_weights_file, cwr_serialization_file,
                cwr_layers_Model, prediction_level_Model);
    } else if(model == "mobilenet") {
        mobilenet_config(dataset_variation, reduced_test_set, nicv2_scenario,
                selected_configuration, train_filelists, test_filelist, db_path,
                solver_file_first_batch, solver_file,
                class_labels_txt, tmp_weights_file,
                init_weights_file, cwr_serialization_file,
                cwr_layers_Model, prediction_level_Model);
    } else {
        throw std::runtime_error("Invalid model, must be caffenet or mobilenet!");
    }
}


string hash_generic(void* pointer, size_t size_in_bytes) {
    return picosha2::hash256_hex_string((uchar*)pointer, ((uchar*)pointer)+size_in_bytes);
}

string hash_caffe_blob(const boost::shared_ptr<caffe::Blob<float>> &blob) {
    const float* data = blob->cpu_data();
    int byte_size = blob->count() * sizeof(float);
    return hash_generic((void*)data, byte_size);
}

string hash_caffe_blob(const boost::shared_ptr<caffe::Net<float>> &net, const std::string &blob_name) {
    auto blob = net->blob_by_name(blob_name);
    return hash_caffe_blob(blob);
}

string hash_net_params(const boost::shared_ptr<Net<float>> &net) {
    string result;
    vector<string> names = net->layer_names();
    std::sort(names.begin(), names.end());

    for(auto& name : names) {
        auto blobs = net->layer_by_name(name)->blobs();
        if(!blobs.empty()) {
            if(blobs.size() <= 2) {
                result += name;
                result += "\t";
                result += hash_caffe_blob(blobs[0]);
                if (blobs.size() == 2) {
                    result += "\nBias: ";
                    result += hash_caffe_blob(blobs[1]);
                }
                result += "\n";
            } else {
                result += name;
                result += ":\n";
                for(int i = 0; i < blobs.size(); i++) {
                    result += "\t";
                    result += hash_caffe_blob(blobs[i]);
                    result += "\n";
                }
            }
        }
    }

    return result;
}

string hash_net(const boost::shared_ptr<Net<float>> &net, bool include_lrs) {
    picosha2::hash256_one_by_one hasher;

    vector<string> names = net->layer_names();
    std::sort(names.begin(), names.end());

    for(auto& name : names) {
        auto blobs = net->layer_by_name(name)->blobs();
        if(!blobs.empty()) {
            for(int i = 0; i < blobs.size(); i++) {
                const float* blob_data = blobs[i]->cpu_data();
                int byte_size = blobs[i]->count() * sizeof(float);
                hasher.process((uchar*)blob_data, ((uchar*)blob_data)+byte_size);
            }
        }
    }

    if(include_lrs) {
        size_t size_in_bytes = net->params_lr_.size() * sizeof(float);
        hasher.process((uchar*) net->params_lr_.data(), ((uchar*) net->params_lr_.data()) + size_in_bytes);
    }

    hasher.finish();

    std::string hex_str = picosha2::get_hash_hex_string(hasher);

    return hex_str;
}

std::string hash_vector(const std::vector<float> &vec) {
    size_t size_in_bytes = vec.size() * sizeof(float);
    return hash_generic((void*) vec.data(), size_in_bytes);
}

std::string hash_vector(const std::vector<int> &vec) {
    size_t size_in_bytes = vec.size() * sizeof(int);
    return hash_generic((void*) vec.data(), size_in_bytes);
}

std::string hash_vector_as_int64(const std::vector<int> &vec) {
    std::vector<int64_t> vec_64(vec.size());
    for(int i = 0; i < vec.size(); i++) {
        vec_64[i] = vec[i];
    }

    size_t size_in_bytes = vec_64.size() * sizeof(int64_t);
    return hash_generic((void*) vec_64.data(), size_in_bytes);
}

std::string hash_rehearsal_memory(const std::vector<std::shared_ptr<std::vector<float>>> &vec) {
    //picosha2::hash256_one_by_one hasher;
    std::vector<float> unique_byte_array;

    for(int i = 0; i < vec.size(); i++) {
        //std::vector<float> my_vec = *vec[i].get();

        //size_t byte_size = my_vec.size() * sizeof(float);
        //hasher.process((uchar*)my_vec.data(), ((uchar*)my_vec.data())+byte_size);

        unique_byte_array.insert( unique_byte_array.end(), vec[i]->begin(), vec[i]->end() );
    }

    return hash_vector(unique_byte_array);

    /*hasher.finish();

    std::string hex_str = picosha2::get_hash_hex_string(hasher);

    return hex_str;*/
}

vector<float> inference_on_single_pattern(Solver<float>* solver, const string &pattern,
        const string &final_prediction_layer) {

    boost::shared_ptr<Net<float>> test_net = solver->test_nets()[0];

    Mat image;
    image = imread(pattern, cv::IMREAD_COLOR);

    cv::Mat img2;
    image.convertTo(img2, CV_32FC3);
    cv::Mat sample_normalized;
    sample_normalized = img2 - core50_mean_image;

    vector<float> label_input(1, 0.0f);
    feed_image_layer(test_net, "data", {sample_normalized}, 0, 1);
    //feed_label_layer(test_net, "label", label_input, 0, 1);
    test_net->Forward();

    vector<float> result;
    boost::shared_ptr<Blob<float>> pred_layer = test_net->blob_by_name(final_prediction_layer);

    for (int column = 0; column < pred_layer->shape(1); column++) {
        result.push_back(pred_layer->data_at({0, column}));
    }

    return result;
}

void log_android_debug(std::stringstream &debug_stream) {
    int buffer_len;
    streamsize read;
    buffer_len = 500;
    std::string prev_remaining = "", actual_remaining = "";

    char buffer[buffer_len];

    debug_stream.read(buffer, buffer_len);
    read = debug_stream.gcount();
    while(!(debug_stream.eof() && read <= 0)) {
        std::string out_str(buffer, read);

        size_t lf = out_str.find_last_of('\n');
        if(lf != std::string::npos && lf != (read-1) && lf != 0) {
            actual_remaining = std::string(&out_str.c_str()[lf+1]);
            out_str = out_str.substr(0, lf);
        } else {
            actual_remaining = "";
        }

#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s%s", prev_remaining.c_str(), out_str.c_str());
#else
        cout << "[NativePrint]" << prev_remaining << out_str << endl;
#endif

        debug_stream.read(buffer, buffer_len - actual_remaining.length());
        read = debug_stream.gcount();
        prev_remaining = actual_remaining;
    }

    debug_stream.str("");
    debug_stream.clear();
}

std::unordered_map<std::string, int> generate_layers_name_to_id_mapping(caffe::Net<float>* net) {
    std::unordered_map<std::string, int> layer_names_to_id;

    const auto &layers = net->layers();

    for (int layer_id = 0; layer_id < net->layer_names_.size(); ++layer_id) {
        layer_names_to_id.insert(std::pair<string, int>(net->layer_names_[layer_id], layer_id));
        //layer_names_index_[layer_names_[layer_id]] = layer_id;
    }

    /*for(int layer_index = 0; layer_index < layers.size(); layer_index++) {
        layer_names_to_id.insert(std::pair<string, int>(layers[layer_index]->layer_param().name(), layer_index));
    }*/

    return layer_names_to_id;
}

//TODO: remove duplicate in cwr.cpp
std::unordered_map<std::string, int> generate_layers_name_to_param_id_mapping(const boost::shared_ptr<caffe::Net<float>> &net) {
    return generate_layers_name_to_param_id_mapping(net.get());
}

std::unordered_map<std::string, int> generate_layers_name_to_param_id_mapping(caffe::Net<float>* net) {
    std::unordered_map<std::string, int> layer_names_to_param_id;

    const auto &layers = net->layers();

    //int layer_index = 0;
    int param_entry = 0;
    for(int i = 0; i < layers.size(); i++) {
        auto lr = layers[i];
        int blobs_sz = lr->blobs().size();

        if(blobs_sz > 0) {
            layer_names_to_param_id.insert(std::pair<std::string, int>(net->layer_names()[i], param_entry));
        }

        param_entry += blobs_sz;
    }

    return layer_names_to_param_id;
}

std::unordered_map<std::string, int> generate_layers_name_to_id_mapping(const boost::shared_ptr<caffe::Net<float>> &net) {
    return generate_layers_name_to_id_mapping(net.get());
}

int get_layer_id_from_name(const boost::shared_ptr<caffe::Net<float>> &net, const std::string &layer_name) {
    const auto &layers = net->layers();

    for (int layer_id = 0; layer_id < net->layer_names_.size(); ++layer_id) {
        if(layer_name == net->layer_names_[layer_id]) {
            return layer_id;
        }
    }

    return -1;
}

std::vector<std::shared_ptr<std::vector<float>>>
get_layer_features(const boost::shared_ptr<caffe::Net<float>> &net, const std::string &layer_name, int from, int to) {
    std::vector<std::shared_ptr<std::vector<float>>> result;

    boost::shared_ptr<Blob<float>> blob = net->blob_by_name(layer_name);
    int pattern_feature_size = 1;
    for(int i = 1; i < blob->shape().size(); i++) {
        pattern_feature_size *= blob->shape()[i];
    }

    if(to < 0) {
        to = blob->shape()[0];
    }

    for (int i = from; i < to; i++) {
        const float* pattern_data_start = &blob->cpu_data()[i * pattern_feature_size];
        const float* pattern_data_end = &blob->cpu_data()[(i+1) * pattern_feature_size];

        result.push_back(std::make_shared<std::vector<float>>(pattern_data_start, pattern_data_end));
    }

    return result;
}

std::vector<float> per_class_accuracy(int classes, const std::vector<int> &pred_y, const std::vector<int> &truth) {
    vector<float> result(classes, 0.0f);
    vector<int> correct(classes, 0);
    vector<int> all(classes, 0);
    int i, prediction, correct_label;

    for(i = 0; i < pred_y.size(); i++) {
        prediction = pred_y[i];
        correct_label = truth[i];

        all[correct_label]++;

        if(prediction == correct_label) {
            correct[prediction]++;
        }
    }

    for(i = 0; i < classes; i++) {
        if(all[i] == 0) {
            result[i] = NAN;
        } else {
            result[i] = (double) correct[i] /  (double) all[i];
        }
        //cout << "Class " << i << " " << correct[i] << " / " << all[i] << " = " << result[i] << endl;
    }

    return result;
}

std::string hash_net_lrs(const boost::shared_ptr<caffe::Net<float>> &net) {
    return hash_vector(net->params_lr_);
}

bool file_exists(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

vector<int> indexes_range(int min, int max, int n_to_generate) {
    vector<int> result;

    if((max - min) + 1 < n_to_generate) {
        for(int i = min; i <= max; i++) {
            result.push_back(i);
        }
    } else {
        double delta = ((double)(max - min)) / (double)(n_to_generate - 1);
        for (int i = 0; i < n_to_generate; i++) {
            result.push_back((int) std::lround((double) min + i * delta));
        }
    }

    return result;
}


