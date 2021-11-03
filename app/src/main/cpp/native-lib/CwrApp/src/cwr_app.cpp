#include <cwr_app.h>
#include <boost/filesystem.hpp>
#include <cblas.h>


using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace boost::filesystem;


bool CwrApp::caffe_initialized = false;
CwrApp::CwrApp() = default;

CwrApp::CwrApp(const std::string &solver_path,
               const std::string &initial_weights_path,
               int classes_n,
               const std::string &initial_prediction_layer,
               const std::vector<std::string> &intial_labels,
               const cv::Mat& initial_mean_image) {
    std::stringstream debug_str;

    this->training_disabled = true;
    if(!caffe_initialized) {
        caffe_initialized = true;
        google::InitGoogleLogging("");
        google::InstallFailureSignalHandler();
        Caffe::set_mode(Caffe::CPU);

        debug_str << "OpenBLAS threads: " << openblas_get_num_threads();
        log_android_debug(debug_str);
    }

    this->solver_path = solver_path;
    this->weights_path = initial_weights_path;
    {
        SolverParameter solver_param;
        ReadSolverParamsFromTextFileOrDie(solver_path, &solver_param);

        solver = caffe::SolverRegistry<float>::CreateSolver(solver_param);
    }

    net = solver->net();
    net->CopyTrainedLayersFrom(initial_weights_path);

    test_net = solver->test_nets()[0];
    test_net->ShareTrainedLayersWith(net.get());

    this->train_layer_name_to_id = generate_layers_name_to_id_mapping(this->net);
    this->test_layer_name_to_id = generate_layers_name_to_id_mapping(this->test_net);
    this->train_layer_names_to_param_id = generate_layers_name_to_param_id_mapping(this->net);

    this->preallocate_memory();

    //test_net->CopyTrainedLayersFrom(initial_test_weights_path);

    extract_minibatch_size_from_prototxt_with_input_layers(net, test_net,
                                                           &minibatch_size,
                                                           &inference_minibatch_size);

    labels = intial_labels;
    training_batch_images = {};
    training_batch_features = {};
    mean_image = initial_mean_image;
    prediction_layer = initial_prediction_layer;
    this->epochs = 0;
}

CwrApp::CwrApp(
        const std::string &solver_path,
        const std::string &initial_weights_path,
        const std::vector<float> &initial_class_updates,
        int train_epochs,
        int initial_classes_n,
        const std::vector<std::string> &cwr_layers,
        const std::string &initial_prediction_layer,
        const std::vector<std::string> &intial_labels,
        const Mat& initial_mean_image,
        const std::string &training_pre_extraction_layer,
        const std::string &backward_stop_layer) {
    std::stringstream debug_str;

    //TODO: add parameters checks
    this->training_disabled = false;
    this->feature_extraction_layer = training_pre_extraction_layer;
    this->backward_stop_layer = backward_stop_layer;
    if(!caffe_initialized) {
        caffe_initialized = true;
        google::InitGoogleLogging("");
        google::InstallFailureSignalHandler();
        Caffe::set_mode(Caffe::CPU);
        debug_str << "OpenBLAS threads: " << openblas_get_num_threads();

        log_android_debug(debug_str);
    }

    this->solver_path = solver_path;
    this->weights_path = initial_weights_path;
    {
        SolverParameter solver_param;
        ReadSolverParamsFromTextFileOrDie(solver_path, &solver_param);

        solver = caffe::SolverRegistry<float>::CreateSolver(solver_param);
    }

    net = solver->net();
    net->CopyTrainedLayersFrom(initial_weights_path);

    test_net = solver->test_nets()[0];
    test_net->ShareTrainedLayersWith(net.get());
    //test_net->CopyTrainedLayersFrom(initial_test_weights_path);

    this->train_layer_name_to_id = generate_layers_name_to_id_mapping(this->net);
    this->test_layer_name_to_id = generate_layers_name_to_id_mapping(this->test_net);
    this->train_layer_names_to_param_id = generate_layers_name_to_param_id_mapping(this->net);

    this->preallocate_memory();

    cwr = Cwr(initial_classes_n, initial_class_updates, net, cwr_layers, true);

    cwr.zeros_non_cwr_layers_lr();
    cwr.zeros_cwr_layer_bias_lr(-1.0f);
    this->set_brn_past_weight(20000.0);

    extract_minibatch_size_from_prototxt_with_input_layers(net, test_net,
                                                           &minibatch_size,
                                                           &inference_minibatch_size);

    bool need_target = net_use_target_vectors(net);
    if(!need_target) {
        throw std::runtime_error("Net has no target layer!");
    }

    labels = intial_labels;
    training_batch_images = {};
    training_batch_features = {};
    mean_image = initial_mean_image;
    prediction_layer = initial_prediction_layer;
    this->epochs = train_epochs;
}

Mat CwrApp::create_mean_image(float B, float G, float R) {
    Mat mean(128, 128, CV_32FC3);
    auto mean_data = (float*) mean.data;

    for (int size = 0; size < 3 * 128 * 128; size += 3) { // 3 = channels
        mean_data[size + 0] = B;
        mean_data[size + 1] = G;
        mean_data[size + 2] = R;
    }

    return mean;
}


int CwrApp::get_minibatch_size() {
    return minibatch_size;
}

void CwrApp::add_batch_image(const Mat& new_image) {
    if(this->training_disabled) {
        throw std::runtime_error("Training disabled!");
    }

    stringstream debug_str;

    this->mtx->lock();
    if(this->must_pre_extract_batch_features()) {  // True
        debug_str << "Pre-extracting features from layer " << this->feature_extraction_layer << endl;
        std::chrono::high_resolution_clock::time_point t1, t2;
        auto duration = duration_cast<milliseconds>( t2 - t1 ).count(); // Trick used for duration "auto" definition
        t1 = std::chrono::high_resolution_clock::now();

        this->training_batch_features.push_back(
                std::make_shared<vector<float>>(this->inference(new_image, this->feature_extraction_layer)));

        t2 = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        debug_str << "Features extraction took: " << duration << " ms" << endl;
    } else {
        debug_str << "Adding pattern without features pre-extraction" << endl;
        this->training_batch_images.push_back(new_image);
    }

    this->mtx->unlock();

    log_android_debug(debug_str);
}

caffe::Solver<float>* CwrApp::get_solver() {
    return this->solver;
}

void CwrApp::reset_batch() {
    if(this->training_disabled) {
        throw std::runtime_error("Training disabled!");
    }

    this->mtx->lock();
    this->training_batch_features.clear();
    this->training_batch_images.clear();
    this->mtx->unlock();
}

boost::shared_ptr<Net<float>> CwrApp::get_net() {
    return this->net;
}

void CwrApp::cwr_execute_step(int label,
        ModelStatusReporter* reporter,
        TrainingProgressListener* callback) {
    if(this->training_disabled) {
        throw std::runtime_error("Training disabled!");
    }

    this->mtx->lock();
    int train_minibatch_size = this->get_minibatch_size();
    if(this->training_threads > 0) {
        int max_th = min(this->training_threads, train_minibatch_size);
        cout << "Using " << max_th << " training threads" << endl;
        openblas_set_num_threads(max_th);
    }

    cout << "Using " << openblas_get_num_threads() << " training threads" << endl;

    if(this->must_pre_extract_batch_features()) {  // True, as we're running latent replay
        this->cwr_execute_step_with_features(label, reporter, callback);
    } else {
        this->cwr_execute_step_with_images(label, reporter, callback);
    }

    this->mtx->unlock();
}

cv::Mat CwrApp::subtract_mean_image(const cv::Mat &image) {
    return image - this->mean_image;
}

std::vector<float> CwrApp::inference(const cv::Mat &image) {
    return this->inference(image, this->prediction_layer);
}

std::vector<float> CwrApp::inference(const cv::Mat &image,
        const std::string &output_layer_name) {
    mtx->lock();

    if(this->feature_extraction_threads > 0) {
        int max_th = min(this->feature_extraction_threads, this->get_inference_minibatch_size());
        openblas_set_num_threads(max_th);
    }

    feed_image_layer(test_net, "data", {image}, 0, 1);
    test_net->ForwardTo(test_layer_name_to_id.at(output_layer_name));

    boost::shared_ptr<Blob<float>> pred_layer = test_net->blob_by_name(output_layer_name);
    vector<float> result(pred_layer->shape(1));

    int start_offset = pred_layer->offset({0, 0, 0, 0});
    memcpy(result.data(), &pred_layer->cpu_data()[start_offset], pred_layer->shape(1)*sizeof(float));

    mtx->unlock();

    return result;
}

void CwrApp::save_everything_to_disk(const std::string &folder) {
    mtx->lock();
    std::stringstream debug_str;
    debug_str << "save_everything_to_disk:" << endl;

    // CWR status
    std::vector<std::string> cwr_layers_names = cwr.get_cwr_layers_name();
    uint32_t max_classes = cwr.get_max_classes();
    uint32_t current_classes_n = cwr.get_current_classes_n();
    std::vector<float> class_updates = cwr.get_class_updates();

    // App status
    const std::string& solver_path = this->solver_path;
    const std::string& weights_path = this->weights_path;
    const std::vector<string>& labels = this->labels;
    const cv::Mat& mean_image = this->mean_image;
    const std::string& prediction_layer = this->prediction_layer;
    const std::string& feature_extraction_layer = this->feature_extraction_layer;
    const uint32_t epochs = this->epochs;
    const std::string& rehearsal_layer = this->rehearsal_layer;
    const std::string& backward_stop_layer = this->backward_stop_layer;
    const std::shared_ptr<RehearsalMemory<cwra_rehe_t>> rehearsal_memory = this->rehearsal_memory;

    //Serialization
    boost::filesystem::path serialization_folder{folder};
    boost::filesystem::create_directory(serialization_folder);
    boost::filesystem::path serialization_file = serialization_folder / "cwr_app_status";
    std::string path = serialization_file.make_preferred().string();

    debug_str << "saving to: " << path << endl;

    auto out_file = std::fstream(path, std::ios::out | std::ios::binary);

    serialize_string_vector(cwr_layers_names, out_file);
    debug_str << cwr_layers_names << endl;

    serialize_uint32t(current_classes_n, out_file);
    debug_str << current_classes_n << ", ";

    serialize_float_vector(class_updates, out_file);
    debug_str << class_updates << endl;

    serialize_string(solver_path, out_file);
    debug_str << solver_path << ", ";

    serialize_string(weights_path, out_file);
    debug_str << weights_path << endl;

    serialize_string_vector(labels, out_file);
    debug_str << labels << ", " << endl;

    serialize_Mat(mean_image, out_file);
    debug_str << hash_generic((void*) mean_image.data, mean_image.total() * mean_image.elemSize()) << ", " << endl;

    serialize_string(prediction_layer, out_file);
    debug_str << prediction_layer << ", ";

    serialize_string(feature_extraction_layer, out_file);
    debug_str << feature_extraction_layer << ", ";

    serialize_uint32t(epochs, out_file);
    debug_str << epochs << ", ";

    serialize_string(rehearsal_layer, out_file);
    debug_str << rehearsal_layer << ", ";

    serialize_string(backward_stop_layer, out_file);
    debug_str << backward_stop_layer << ", ";

    if(!rehearsal_layer.empty()) {
        rehearsal_memory->save_everything(out_file);
        debug_str << "ReheMem: " << hash_rehearsal_memory(rehearsal_memory->getSamplesX()) << endl;
    }

    out_file.close();

    if(out_file.fail()) {
        debug_str << "Failed to write state to disk" << endl;
    } else {
        this->overwrite_model_on_disk();
    }

    /*log_android_debug( debug_str );

    debug_str << "Params hash (net):" << endl << hash_net_params(this->net) << endl;
    log_android_debug( debug_str );

    debug_str << "Params hash (test_net):" << endl << hash_net_params(this->test_net) << endl;*/


    log_android_debug( debug_str );

    mtx->unlock();
}

void CwrApp::overwrite_model_on_disk() {
    mtx->lock();
    std::string net_path = this->weights_path;
    this->serialize_model_on_disk(net_path);
    mtx->unlock();
}

void CwrApp::serialize_model_on_disk(const std::string &path) {
    mtx->lock();
    //this->test_net->Forward();
    //this->net->Forward();

    //__android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "serialize_model_on_disk net %s", hash_net(this->net).c_str());
    //__android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "serialize_model_on_disk test net %s", hash_net(this->test_net).c_str());

    string net_path = path;
    //string test_net_path = path + "_test";
    NetParameter net_param;
    this->net->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, net_path);

    /*NetParameter test_net_param;
    this->test_net->ToProto(&test_net_param, false);
    WriteProtoToBinaryFile(test_net_param, test_net_path);*/
    mtx->unlock();
}

bool CwrApp::state_exists(const std::string &folder) {
    boost::filesystem::path serialization_folder{folder};
    boost::filesystem::path serialization_file = serialization_folder / "cwr_app_status";
    std::string path = serialization_file.make_preferred().string();

    if (FILE *file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

CwrApp CwrApp::read_from_disk(const std::string &folder) {
    std::stringstream debug_str;
    debug_str << "read_from_disk:" << endl;

    // CWR status
    std::vector<std::string> cwr_layers_names;
    uint32_t max_classes;
    uint32_t current_classes_n;
    std::vector<float> class_updates;

    // App status
    std::string solver_path;
    std::string weights_path;
    std::vector<string> labels;
    cv::Mat mean_image;
    std::string prediction_layer;
    std::string feature_extraction_layer;
    uint32_t epochs;
    std::string rehearsal_layer;
    std::string backward_stop_layer;
    std::shared_ptr<RehearsalMemory<cwra_rehe_t>> rehearsal_memory;

    //Serialization
    boost::filesystem::path serialization_folder{folder};
    boost::filesystem::path serialization_file = serialization_folder / "cwr_app_status";
    std::string path = serialization_file.make_preferred().string();

    debug_str << "loading from: " << path << endl;

    auto in_file = std::fstream(path, std::ios::in | std::ios::binary);

    log_android_debug( debug_str );

    cwr_layers_names = deserialize_string_vector(in_file); // OK
    debug_str << cwr_layers_names << endl;

    current_classes_n = deserialize_uint32t(in_file);
    debug_str << current_classes_n << ", ";

    class_updates = deserialize_float_vector(in_file); // OK
    debug_str << class_updates << endl;

    solver_path = deserialize_string(in_file); // OK
    debug_str << solver_path << ", ";

    weights_path = deserialize_string(in_file); // OK
    debug_str << weights_path << endl;

    labels = deserialize_string_vector(in_file); // OK
    debug_str << labels << ", " << endl;

    log_android_debug( debug_str );

    mean_image = deserialize_Mat(in_file); // OK
    debug_str << hash_generic((void*) mean_image.data, mean_image.total() * mean_image.elemSize()) << ", " << endl;

    prediction_layer = deserialize_string(in_file); // OK
    debug_str << prediction_layer << ", ";

    feature_extraction_layer = deserialize_string(in_file); // OK
    debug_str << feature_extraction_layer << ", ";

    epochs = deserialize_uint32t(in_file); // OK
    debug_str << epochs << ", ";

    rehearsal_layer = deserialize_string(in_file);
    debug_str << rehearsal_layer << ", ";

    backward_stop_layer = deserialize_string(in_file);
    debug_str << backward_stop_layer << ", ";

    if(!rehearsal_layer.empty()) {
        rehearsal_memory = load_rehe_from_snapshot(in_file);
        debug_str << "ReheMem: " << hash_rehearsal_memory(rehearsal_memory->getSamplesX()) << endl;
    }

    in_file.close(); // OK

    CwrApp result = CwrApp(solver_path, weights_path, class_updates, epochs,
                  current_classes_n, cwr_layers_names, prediction_layer, labels, mean_image,
                  feature_extraction_layer, backward_stop_layer);
    if(!rehearsal_layer.empty()) {
        result.set_rehearsal_layer(rehearsal_layer);
        result.set_rehearsal_memory(rehearsal_memory);
    }

    /*debug_str << "Net hash: " << hash_net(result.net) << endl;
    debug_str << "Test net hash: " << hash_net(result.test_net) << endl;

    debug_str << "Params hash (net):" << endl << hash_net_params(result.net) << endl;
    log_android_debug( debug_str );

    debug_str << "Params hash (test_net):" << endl << hash_net_params(result.test_net) << endl;*/

    log_android_debug( debug_str );

    return result;
}

void CwrApp::preallocate_memory() {
    bool has_forward_sensitive_params = false;

    for( auto it = this->train_layer_name_to_id.begin(); it != this->train_layer_name_to_id.end(); ++it )
    {
        string layer_name = it->first;
        int layer_id = it->second;
        boost::shared_ptr<caffe::Layer<float>> layer = this->net->layers()[layer_id];
        string layer_type_str(layer->type());

        if(layer_type_str == "BatchReNorm") {
            has_forward_sensitive_params = true;
            break;
        }
    }

    if(!has_forward_sensitive_params) {
        this->net->Forward();
        this->test_net->Forward();
    }
}

bool CwrApp::is_training_disabled() const {
    return training_disabled;
}

const vector<string> &CwrApp::get_labels() const {
    return labels;
}

const Mat &CwrApp::get_mean_image() const {
    return mean_image;
}

const string &CwrApp::get_prediction_layer() const {
    return prediction_layer;
}

int CwrApp::get_epochs() const {
    return epochs;
}

bool CwrApp::is_caffe_initialized() {
    return caffe_initialized;
}

int CwrApp::add_category(const std::string &category_label) {
    mtx->lock();
    //TODO: add checks (max class number)
    this->labels.push_back(category_label);
    int result = this->cwr.increment_class_number();
    mtx->unlock();
    return result;
}

int CwrApp::get_current_categories_count() {
    return this->cwr.get_current_classes_n();
}

int CwrApp::get_max_categories() {
    return this->cwr.get_max_classes();
}

void CwrApp::set_brn_past_weight(float weight) {
    cwr.set_brn_past_weight(weight);
}

bool CwrApp::must_pre_extract_batch_features() {
    return !this->feature_extraction_layer.empty();
}

void CwrApp::cwr_execute_step_with_images(
        int label,
        ModelStatusReporter* reporter,
        TrainingProgressListener* callback) {
    std::stringstream debug_stream;

    std::chrono::high_resolution_clock::time_point t1, t2;
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count(); // Trick used for duration "auto" definition
    int forward_ms = 0, backward_ms = 0, update_ms = 0;

    std::string backward_stop = this->feature_extraction_layer;
    if(!this->backward_stop_layer.empty()) {
        backward_stop = this->backward_stop_layer;
    }

    int train_minibatch_size = this->get_minibatch_size();
    int train_iterations_per_epoch = (int) ceil((double)training_batch_images.size() / (double)train_minibatch_size);
    int train_iterations = train_iterations_per_epoch * this->epochs;
    std::vector<cv::Mat> train_x = this->training_batch_images;

    std::vector<int> train_y(this->training_batch_images.size(), label);
    std::vector<std::vector<float>> target_y = compute_one_hot_vectors(train_y, this->net, this->prediction_layer);
    int iteration;

    pad_data_single(train_x, train_minibatch_size);
    pad_data_single(train_y, train_minibatch_size);
    pad_data_single(target_y, train_minibatch_size);

    std::set<int> unique_y_set(train_y.begin(), train_y.end());

    std::vector<cwra_rehe_t> rehe_x;
    std::vector<int> rehe_y;
    std::vector<std::vector<float>> rehe_t;
    int reha_iters_per_epoch = 0;
    int reha_in_minibatch = 0;

    if(this->rehearsal_memory != nullptr) {
        rehe_x = this->rehearsal_memory->getSamplesX();
        rehe_y = this->rehearsal_memory->getSamplesY();
        rehe_t = compute_one_hot_vectors(rehe_y, this->net, this->prediction_layer);
        cout << "Will use rehearsal memory: " << rehe_x.size() << " patterns";
        unique_y_set.insert(rehe_y.begin(), rehe_y.end());

        reha_in_minibatch = this->get_net()->blob_by_name("data_reha")->shape()[0];
        reha_iters_per_epoch = pad_data_single(rehe_x, reha_in_minibatch);
        pad_data_single(rehe_y, reha_in_minibatch);
        pad_data_single(rehe_t, reha_in_minibatch);
        cout << " (" << rehe_x.size() << " after padding)" << endl;
    }

    std::vector<int> unique_y(unique_y_set.begin(), unique_y_set.end());
    std::sort(unique_y.begin(), unique_y.end());
    std::vector<int> y_freq(unique_y.size(), 0);
    for(auto &train_label : train_y) {
        for(int i = 0; i < unique_y.size(); i++) {
            if(unique_y[i] == train_label) {
                y_freq[i]++;
            }
        }
    }

    for(auto &train_label : rehe_y) {
        for(int i = 0; i < unique_y.size(); i++) {
            if(unique_y[i] == train_label) {
                y_freq[i]++;
            }
        }
    }

    if(reporter) reporter->startChangePhase("Reset weights");
    cwr.reset_weights();
    if(reporter) reporter->onPhaseFinished("Reset weights");
    if(reporter) reporter->startChangePhase("Load weights nic");
    cwr.load_weights_nic(unique_y);
    if(reporter) reporter->onPhaseFinished("Load weights nic");
    //shuffle_in_unison(train_x, train_y, target_y, 0);

    if(rehe_x.size() > 1) {
        //shuffle_in_unison(rehe_x, rehe_t, target_y, 0);
    }

    iteration = 0;

    int backward_stop_layer_id = train_layer_name_to_id.at(backward_stop);

    while (iteration < train_iterations) {
        debug_stream << "Iteration " << iteration << endl;
        log_android_debug(debug_stream);

        if(callback != nullptr) {
            callback->updateProgress((float) iteration / train_iterations);
        }

        int it_mod = iteration % train_iterations_per_epoch;
        int start = it_mod * train_minibatch_size;
        int end = (it_mod + 1) * train_minibatch_size;

        feed_image_layer(this->get_net(), "data", train_x, start, end);
        feed_label_layer(this->get_net(), "label", train_y, start, end);
        feed_target_layer(this->get_net(), "target", target_y, start, end);

        if(!rehe_x.empty()) {
            int it_mod_reha = iteration % reha_iters_per_epoch;
            int reha_start = it_mod_reha * reha_in_minibatch;
            int reha_end = (it_mod_reha + 1) * reha_in_minibatch;

            feed_feature_layer(this->get_net(), "data_reha", rehe_x, reha_start, reha_end);
            feed_label_layer(this->get_net(), "label", rehe_y, reha_start, reha_end, train_minibatch_size);
            feed_target_layer(this->get_net(), "target", rehe_t, reha_start, reha_end, train_minibatch_size);
        }

        /*debug_stream << "Data hash: " << hash_caffe_blob(this->get_net(), "data") << endl;
        if(!rehe_x.empty()) {
            debug_stream << "Rehe data hash: " << hash_caffe_blob(this->get_net(), "data_reha") << endl;
        }
        debug_stream << "Labels hash: " << hash_caffe_blob(this->get_net(), "label") << endl;
        debug_stream << "Target hash: " << hash_caffe_blob(this->get_net(), "target") << endl;*/

        if(reporter) reporter->startChangePhase("Forward");
        t1 = std::chrono::high_resolution_clock::now();
        net->ClearParamDiffs();
        net->Forward();
        t2 = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        forward_ms += duration;
        if(reporter) reporter->onPhaseFinished("Forward");

        if(reporter) reporter->startChangePhase("Backward");
        t1 = std::chrono::high_resolution_clock::now();
        net->BackwardTo(backward_stop_layer_id);
        t2 = high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        backward_ms += duration;
        if(reporter) reporter->onPhaseFinished("Backward");

        if(reporter) reporter->startChangePhase("ApplyUpdate");
        t1 = std::chrono::high_resolution_clock::now();
        solver->ApplyUpdate();
        t2 = high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        update_ms += duration;
        if(reporter) reporter->onPhaseFinished("ApplyUpdate");

        if (iteration == (train_iterations - 1)) {
            if(reporter) reporter->startChangePhase("Cwr consolidation");
            cout << unique_y << endl;
            cout << y_freq << endl;

            if(reporter) reporter->startChangePhase("Consolidate weights CWR+");
            cwr.consolidate_weights_cwr_plus(unique_y, y_freq);
            if(reporter) reporter->onPhaseFinished("Consolidate weights CWR+");

            // increase weights of trained classes
            vector<float> class_updates = cwr.get_class_updates();

            for(int i = 0; i < unique_y.size(); i++) {
                class_updates[unique_y[i]] = class_updates[unique_y[i]] + y_freq[i];
            }

            cwr.set_class_updates(class_updates);

            if(reporter) reporter->startChangePhase("Load weights");
            cwr.load_weights();
            if(reporter) reporter->onPhaseFinished("Load weights");
            if(reporter) reporter->onPhaseFinished("Cwr consolidation");
        }

        iteration++;
    }

    debug_stream << "Forward time: " << forward_ms << endl
                 << "Backward time: " << backward_ms << endl
                 << "Update time: " << update_ms << endl;

    if(this->rehearsal_memory != nullptr) {
        debug_stream << "Updating rehearsal memory" << endl;

        /* This is the non-balanced version in which 10 randomly chosen instances are
         * inserted in the replay buffer by removing 10 randomly chosen replay instances.

        vector<int> to_be_added_idx = indexes_range(0, train_x.size()-1, 10);
        debug_stream << "Using patterns at indexes: " << to_be_added_idx << endl;
        vector<cwra_rehe_t> to_be_added_x;
        vector<int> to_be_added_y;
        for(int i = 0; i < to_be_added_idx.size(); i++) {
            vector<float> extracted_features = this->inference(train_x[to_be_added_idx[i]], this->rehearsal_layer);

            to_be_added_x.push_back(std::make_shared<vector<float>>(extracted_features));
            to_be_added_y.push_back(train_y[to_be_added_idx[i]]);
        }
        this->rehearsal_memory->update_memory(to_be_added_x, to_be_added_y, 1, 10);
         */

        /* This is the balanced version in which all instances are considered for insertion.
         * The replay buffer will be class-balanced.
         *
         * The selection is based on random choice over the union of old+new instances.
         */
        debug_stream << "Using all patterns for balanced replay: " << endl;
        vector<cwra_rehe_t> to_be_added_x;
        vector<int> to_be_added_y;
        for(int i = 0; i < train_x.size(); i++) {
            vector<float> extracted_features = this->inference(train_x[i], this->rehearsal_layer);

            to_be_added_x.push_back(std::make_shared<vector<float>>(extracted_features));
            to_be_added_y.push_back(train_y[i]);
        }
        this->rehearsal_memory->update_memory_balanced(to_be_added_x, to_be_added_y, 1, -1);
    }

    if(callback != nullptr) {
        callback->updateProgress(1.0f);
    }

    log_android_debug(debug_stream);
}

void CwrApp::cwr_execute_step_with_features(
        int label,
        ModelStatusReporter* reporter,
        TrainingProgressListener* callback) {
    // Define variables used for timing and debugging
    std::stringstream debug_stream;

    std::chrono::high_resolution_clock::time_point t1, t2;
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count(); // Trick used for duration "auto" definition
    int preparation_ms = 0, data_feed_ms = 0, forward_ms = 0, backward_ms = 0, update_ms = 0, consolidation_ms = 0;

    t1 = std::chrono::high_resolution_clock::now();

    // Get the backward stop layer
    // We want to keep the part of the model before the latent replay layer frozen, which means
    // that we can stop the backward pass early.
    std::string backward_stop = this->feature_extraction_layer;
    if(!this->backward_stop_layer.empty()) {
        backward_stop = this->backward_stop_layer;
    }

    // Compute the amount of train iterations
    int train_minibatch_size = this->get_features_train_minibatch_size();
    int train_iterations_per_epoch = (int) ceil((double)training_batch_features.size() / (double)train_minibatch_size);
    int train_iterations = train_iterations_per_epoch * this->epochs;
    std::vector<std::shared_ptr<std::vector<float>>> train_x = this->training_batch_features;

    // Compute one-hot vectors
    std::vector<int> train_y(this->training_batch_features.size(), label);
    std::vector<std::vector<float>> target_y = compute_one_hot_vectors(train_y, this->net, this->prediction_layer);
    int iteration;

    // If (n_train_instances % mb_size) != 0, then pad the batch by duplicating some instances
    pad_data_single(train_x, train_minibatch_size);
    pad_data_single(train_y, train_minibatch_size);
    pad_data_single(target_y, train_minibatch_size);

    std::set<int> unique_y_set(train_y.begin(), train_y.end());

    // Manage the replay buffer
    std::vector<cwra_rehe_t> rehe_x;
    std::vector<int> rehe_y;
    std::vector<std::vector<float>> rehe_t;
    int reha_iters_per_epoch = 0;
    int reha_in_minibatch = 0;

    if(this->rehearsal_memory != nullptr) {
        rehe_x = this->rehearsal_memory->getSamplesX();
        rehe_y = this->rehearsal_memory->getSamplesY();
        rehe_t = compute_one_hot_vectors(rehe_y, this->net, this->prediction_layer);
        cout << "Will use rehearsal memory: " << rehe_x.size() << " patterns";
        unique_y_set.insert(rehe_y.begin(), rehe_y.end());

        reha_in_minibatch = this->get_net()->blob_by_name("data_reha")->shape()[0];
        reha_iters_per_epoch = pad_data_single(rehe_x, reha_in_minibatch);
        pad_data_single(rehe_y, reha_in_minibatch);
        pad_data_single(rehe_t, reha_in_minibatch);
        cout << " (" << rehe_x.size() << " after padding)" << endl;
    }

    // Compute the amount of instances per category (to be used in the CWR consolidation procedure)
    std::vector<int> unique_y(unique_y_set.begin(), unique_y_set.end());
    std::sort(unique_y.begin(), unique_y.end());
    std::vector<int> y_freq(unique_y.size(), 0);
    for(auto &train_label : train_y) {
        for(int i = 0; i < unique_y.size(); i++) {
            if(unique_y[i] == train_label) {
                 y_freq[i]++;
            }
        }
    }

    for(auto &train_label : rehe_y) {
        for(int i = 0; i < unique_y.size(); i++) {
            if(unique_y[i] == train_label) {
                y_freq[i]++;
            }
        }
    }

    // CWR initialization steps
    debug_stream << "Resetting weights" << endl;
    log_android_debug(debug_stream);

    if(reporter) reporter->startChangePhase("Reset weights");
    cwr.reset_weights();
    if(reporter) reporter->onPhaseFinished("Reset weights");

    debug_stream << "Loading nic ones" << endl;
    log_android_debug(debug_stream);
    if(reporter) reporter->startChangePhase("Load weights nic");
    cwr.load_weights_nic(unique_y);
    if(reporter) reporter->onPhaseFinished("Load weights nic");

    //shuffle_in_unison(train_x, train_y, target_y, 0);
    if(rehe_x.size() > 1) {
        //shuffle_in_unison(rehe_x, rehe_t, target_y, 0);
    }

    iteration = 0;

    // Get the ID of the feature extraction layer
    debug_stream << "Getting id of feature extraction layer " << this->feature_extraction_layer << endl;
    log_android_debug(debug_stream);
    int features_layer_id = train_layer_name_to_id.at(this->feature_extraction_layer);

    // Get the ID of the backward stop layer
    debug_stream << "Getting id of backward stop layer  " << backward_stop << endl;
    log_android_debug(debug_stream);
    int backward_stop_layer_id = train_layer_name_to_id.at(backward_stop);
    int backward_stop_layer_param_id = train_layer_names_to_param_id.at(backward_stop); // Used for partial apply update

    t2 = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    preparation_ms += duration;

    // Training loop
    while (iteration < train_iterations) {
        debug_stream << "Iteration " << iteration << endl;
        log_android_debug(debug_stream);

        if(callback != nullptr) {
            callback->updateProgress((float) iteration / train_iterations);
        }

        int it_mod = iteration % train_iterations_per_epoch;
        int start = it_mod * train_minibatch_size;
        int end = (it_mod + 1) * train_minibatch_size;

        t1 = std::chrono::high_resolution_clock::now();

        feed_feature_layer(this->get_net(), this->feature_extraction_layer, train_x, start, end);
        feed_label_layer(this->get_net(), "label", train_y, start, end);
        feed_target_layer(this->get_net(), "target", target_y, start, end);

        if(!rehe_x.empty()) {
            int it_mod_reha = iteration % reha_iters_per_epoch;
            int reha_start = it_mod_reha * reha_in_minibatch;
            int reha_end = (it_mod_reha + 1) * reha_in_minibatch;

            feed_feature_layer(this->get_net(), "data_reha", rehe_x, reha_start, reha_end);
            feed_label_layer(this->get_net(), "label", rehe_y, reha_start, reha_end, train_minibatch_size);
            feed_target_layer(this->get_net(), "target", rehe_t, reha_start, reha_end, train_minibatch_size);
        }

        t2 = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        data_feed_ms += duration;

        if(reporter) reporter->startChangePhase("Forward");
        t1 = std::chrono::high_resolution_clock::now();
        net->ClearParamDiffs();
        net->ForwardFrom(features_layer_id);
        t2 = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        forward_ms += duration;
        if(reporter) reporter->onPhaseFinished("Forward");

        if(reporter) reporter->startChangePhase("Backward");
        t1 = std::chrono::high_resolution_clock::now();
        net->BackwardTo(backward_stop_layer_id);
        t2 = high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        backward_ms += duration;
        if(reporter) reporter->onPhaseFinished("Backward");

        if(reporter) reporter->startChangePhase("ApplyUpdate");
        t1 = std::chrono::high_resolution_clock::now();
        //solver->ApplyUpdate();
        partial_apply_update(dynamic_cast<caffe::SGDSolver<float>*>(solver), backward_stop_layer_param_id, -1);
        t2 = high_resolution_clock::now();
        duration = duration_cast<milliseconds>( t2 - t1 ).count();
        update_ms += duration;
        if(reporter) reporter->onPhaseFinished("ApplyUpdate");

        // CWR consolidation step
        if (iteration == (train_iterations - 1)) {
            t1 = std::chrono::high_resolution_clock::now();

            if(reporter) reporter->startChangePhase("Cwr consolidation");
            cout << unique_y << endl;
            cout << y_freq << endl;

            if(reporter) reporter->startChangePhase("Consolidate weights CWR+");
            cwr.consolidate_weights_cwr_plus(unique_y, y_freq);
            if(reporter) reporter->onPhaseFinished("Consolidate weights CWR+");

            // increase weights of trained classes
            vector<float> class_updates = cwr.get_class_updates();

            for(int i = 0; i < unique_y.size(); i++) {
                class_updates[unique_y[i]] = class_updates[unique_y[i]] + y_freq[i];
            }

            cwr.set_class_updates(class_updates);

            if(reporter) reporter->startChangePhase("Load weights");
            cwr.load_weights(); // This prepares the net for inference
            if(reporter) reporter->onPhaseFinished("Load weights");
            if(reporter) reporter->onPhaseFinished("Cwr consolidation");

            t2 = high_resolution_clock::now();
            duration = duration_cast<milliseconds>( t2 - t1 ).count();
            consolidation_ms += duration;
        }

        iteration++;
    }

    debug_stream << "Preparation: " << preparation_ms << endl
                 << "Data feed: " << data_feed_ms << endl
                 << "Forward time: " << forward_ms << endl
                 << "Backward time: " << backward_ms << endl
                 << "Update time: " << update_ms << endl
                 << "Consolidation: " << consolidation_ms << endl;

    // Update the replay buffer
    if(this->rehearsal_memory != nullptr) {
        debug_stream << "Updating rehearsal memory" << endl;

        /* This is the non-balanced version in which 10 randomly chosen instances are
         * inserted in the replay buffer by removing 10 randomly chosen replay instances.

        vector<int> to_be_added_idx = indexes_range(0, train_x.size()-1, 10);
        debug_stream << "Using patterns at indexes: " << to_be_added_idx << endl;
        vector<cwra_rehe_t> to_be_added_x;
        vector<int> to_be_added_y;
        for(int i = 0; i < to_be_added_idx.size(); i++) {
            to_be_added_x.push_back(train_x[to_be_added_idx[i]]);
            to_be_added_y.push_back(train_y[to_be_added_idx[i]]);
        }
        this->rehearsal_memory->update_memory_balanced(to_be_added_x, to_be_added_y, 1, 10);
        */

        /* This is the balanced version in which all instances are considered for insertion.
         * The replay buffer will be class-balanced.
         *
         * The selection is based on random choice over the union of old+new instances.
         */
        debug_stream << "Using all patterns for balanced replay: " << endl;
        vector<cwra_rehe_t> to_be_added_x;
        vector<int> to_be_added_y;
        for(int i = 0; i < train_x.size(); i++) {
            to_be_added_x.push_back(train_x[i]);
            to_be_added_y.push_back(train_y[i]);
        }
        this->rehearsal_memory->update_memory_balanced(to_be_added_x, to_be_added_y, 1, -1);
    }

    if(callback != nullptr) {
        callback->updateProgress(1.0f);
    }

    log_android_debug(debug_stream);
}

void CwrApp::set_training_threads(int threads) {
    this->training_threads = threads;
}

void CwrApp::set_feature_extraction_threads(int threads) {
    this->feature_extraction_threads = threads;
}

int CwrApp::get_inference_minibatch_size() {
    return this->inference_minibatch_size;
}

void CwrApp::set_rehearsal_memory(const std::shared_ptr<RehearsalMemory<cwra_rehe_t>> &new_rehearsal_memory) {
    this->rehearsal_memory = new_rehearsal_memory;
}

std::shared_ptr<RehearsalMemory<cwra_rehe_t>> CwrApp::get_rehearsal_memory() {
    return this->rehearsal_memory;
}

void CwrApp::set_backward_stop_layer(const std::string &new_backward_stop_layer) {
    this->backward_stop_layer = new_backward_stop_layer;
}

void CwrApp::set_rehearsal_layer(const std::string &new_rehearsal_layer) {
    this->rehearsal_layer = new_rehearsal_layer;
}

std::string CwrApp::get_backward_stop_layer() {
    return this->backward_stop_layer;
}

std::string CwrApp::get_rehearsal_layer() {
    return this->rehearsal_layer;
}

boost::shared_ptr<caffe::Net<float>> CwrApp::get_test_net() {
    return this->test_net;
}

int CwrApp::get_features_train_minibatch_size() {
    if(this->feature_extraction_layer.empty()) {
        return 0;
    }

    return this->net->blob_by_name(this->feature_extraction_layer)->shape()[0];
}

void serialize_string(const std::string &str,
                             std::basic_fstream<char, std::char_traits<char>> &out_file) {
    serialize_uint32t(str.length(), out_file);
    out_file.write(str.c_str(), str.length());
}

std::string deserialize_string(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t str_len = deserialize_uint32t(in_file);;

    std::vector<char> buffer(str_len);

    in_file.read(&buffer[0], str_len);
    return std::string(&buffer[0], str_len);
}

void serialize_float_vector(const std::vector<float> &float_vec,
                             std::basic_fstream<char, std::char_traits<char>> &out_file) {
    serialize_uint32t(float_vec.size(), out_file);
    for(auto& fl : float_vec) {
        out_file.write((const char*) &fl, sizeof(float));
    }
}

std::vector<float> deserialize_float_vector(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t result_size = deserialize_uint32t(in_file);
    std::vector<float> result;
    std::vector<char> buffer(sizeof(float));

    for(uint32_t i = 0; i < result_size; i++) {
        in_file.read(&buffer[0], sizeof(float));

        result.push_back(*(float*)(&buffer[0]));
    }

    return result;
}

void serialize_rehearsal_blob(const std::vector<cwra_rehe_t> &rehe_vec,
                              std::basic_fstream<char, std::char_traits<char>> &out_file) {
    serialize_uint32t(rehe_vec.size(), out_file);
    for(auto& r_vec : rehe_vec) {
        serialize_float_vector(*r_vec, out_file);
    }
}

std::vector<cwra_rehe_t> deserialize_rehearsal_blob(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t result_size = deserialize_uint32t(in_file);
    std::vector<cwra_rehe_t> result;

    for(uint32_t i = 0; i < result_size; i++) {
        result.push_back(std::make_shared<vector<float>>(deserialize_float_vector(in_file)));
    }

    return result;
}

void serialize_int_vector(const std::vector<int> &int_vec,
                            std::basic_fstream<char, std::char_traits<char>> &out_file) {
    serialize_uint32t(int_vec.size(), out_file);
    for(auto& fl : int_vec) {
        int64_t int64val = fl;
        out_file.write((const char*) &int64val, sizeof(int64_t));
    }
}

std::vector<int> deserialize_int_vector(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t result_size = deserialize_uint32t(in_file);
    std::vector<int> result;
    std::vector<char> buffer(sizeof(int64_t));

    for(uint32_t i = 0; i < result_size; i++) {
        in_file.read(&buffer[0], sizeof(int64_t));
        result.push_back((int) *(int64_t*)(&buffer[0]));
    }

    return result;
}

void serialize_string_vector(const std::vector<std::string> &str_vec,
                             std::basic_fstream<char, std::char_traits<char>> &out_file) {
    serialize_uint32t(str_vec.size(), out_file);
    for(auto& str : str_vec) {
        serialize_string(str, out_file);
    }
}

std::vector<std::string> deserialize_string_vector(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t result_size = deserialize_uint32t(in_file);
    std::vector<std::string> result;

    for(uint32_t i = 0; i < result_size; i++) {
        result.push_back(deserialize_string(in_file));
    }

    return result;
}

void serialize_Mat(const cv::Mat &image,
                             std::basic_fstream<char, std::char_traits<char>> &out_file) {
    // Implicit: CV_32FC3
    serialize_uint32t(image.cols, out_file);
    serialize_uint32t(image.rows, out_file);
    serialize_uint32t(image.total() * image.elemSize(), out_file);
    out_file.write((const char*) image.data, image.total() * image.elemSize());
}

cv::Mat deserialize_Mat(std::basic_fstream<char, std::char_traits<char>> &in_file) {
    uint32_t cols, rows, byte_size;
    // Implicit: CV_32FC3

    cols = deserialize_uint32t(in_file);
    rows = deserialize_uint32t(in_file);
    byte_size = deserialize_uint32t(in_file);
    std::vector<char> buffer;
    buffer.reserve(byte_size);
    in_file.read(&buffer[0], byte_size);
    cv::Mat result(rows, cols, CV_32FC3);
    memcpy(result.data, &buffer[0], byte_size);
    return result;
}

void serialize_uint32t(uint32_t value, std::basic_fstream<char, std::char_traits<char>> &out_file) {
    char* value_ptr = (char*) &value;
    for(int i = 0; i < 4; i++) {
        out_file.write(&value_ptr[i], 1);
    }
}

uint32_t deserialize_uint32t(std::basic_fstream<char, std::char_traits<char>> &out_file) {
    uint32_t result = 0;
    char* value_ptr = (char*) &result;
    for(int i = 0; i < 4; i++) {
        out_file.read(&value_ptr[i], 1);
    }

    return result;
}

std::shared_ptr<RehearsalMemory<cwra_rehe_t>> load_rehe_from_snapshot(std::basic_fstream<char, std::char_traits<char>> &in_stream) {
    std::stringstream debug_str;
    debug_str << "rehe memory, load_everything:" << endl;

    uint32_t current_num_patterns;
    uint32_t max_num_patterns;
    vector<cwra_rehe_t> external_memory_x;
    vector<int> external_memory_y;

    current_num_patterns = deserialize_uint32t(in_stream);
    debug_str << current_num_patterns << ", ";

    max_num_patterns = deserialize_uint32t(in_stream);
    debug_str << max_num_patterns << ", ";

    external_memory_x = deserialize_rehearsal_blob(in_stream);
    debug_str << "External memory x hash: " << hash_rehearsal_memory(external_memory_x) << "," << endl;

    external_memory_y = deserialize_int_vector(in_stream);
    debug_str << "External memory y hash: " << hash_vector(external_memory_y) << "," << endl;

    std::shared_ptr<RehearsalMemory<cwra_rehe_t>> result =
            std::make_shared<RehearsalMemory<cwra_rehe_t>>(max_num_patterns);
    result->load_memory(external_memory_x, external_memory_y);

    log_android_debug(debug_str);
    return result;
}


