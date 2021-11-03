//
// Created by lorenzo on 05/04/19.
//

#include <cwr.h>
#include <train_utils.h>
#include <iomanip>

//TODO: better constructors
Cwr::~Cwr() {

}

Cwr::Cwr() {
    /*this->class_updates = {};
    this->max_classes = 0;
    this->current_classes_n = 0;
    this->net = nullptr;
    this->cwr_layers_name = {};
    this->create_layers_name_to_id_mapping();
    this->consolidated_class_weights = {};*/
}

Cwr::Cwr(int initial_classes_n, const vector<float> &initial_class_updates,
         const boost::shared_ptr<Net<float>> &net, const vector<string> &cwr_layers_name,
         bool load_weights_from_net) {
    //Checks: len(class_updates) == max_classes
    this->class_updates = initial_class_updates;
    this->max_classes = get_net_max_classes(net, cwr_layers_name[0]);
    this->current_classes_n = initial_classes_n;
    this->net = net;
    this->cwr_layers_name = cwr_layers_name;
    this->create_layers_name_to_id_mapping();
    this->create_layers_name_to_param_lr_id_mapping();
    vector<int> unused_classes;
    for(int i = current_classes_n; i < max_classes; i++) {
        unused_classes.push_back(i);
    }
    this->partial_reset_weights(unused_classes);
    this->init_consolidated_weights(load_weights_from_net); // Consolidated weights = [0, 0, 0, ...]
}


Cwr::Cwr(const Cwr &from, const boost::shared_ptr<Net<float>> &new_net) {
    this->class_updates = from.class_updates;
    this->max_classes = from.max_classes;
    this->current_classes_n = from.current_classes_n;
    this->net = new_net;
    this->cwr_layers_name = from.cwr_layers_name;
    this->create_layers_name_to_id_mapping();
    this->create_layers_name_to_param_lr_id_mapping();
    this->init_consolidated_weights(true); // Load consolidated weights from net
}

void Cwr::init_consolidated_weights(bool from_net) {
    for (auto &it : this->cwr_layers_name) {
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        const boost::shared_ptr<Blob<float>> bias = layer->blobs()[1];

        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cpp
        int k = weights->shape()[1];

        for (int class_index = 0; class_index < this->max_classes; class_index++) {
            if (from_net) {
                const float *weights_data_start = &weights->cpu_data()[class_index * k];
                const float *weights_data_end = &weights->cpu_data()[(class_index + 1) * k];

                this->consolidated_class_weights.insert(
                        std::pair<std::pair<string, int>, std::pair<vector<float>, float>>(
                                std::pair<string, int>(
                                        it,
                                        class_index),
                                std::pair<vector<float>, float>(
                                        std::vector<float>(weights_data_start, weights_data_end), // Weights
                                        bias->cpu_data()[class_index]) // Bias
                        ));
            } else {
                this->consolidated_class_weights.insert(
                        std::pair<std::pair<string, int>, std::pair<vector<float>, float>>(
                                std::pair<string, int>(
                                        it,
                                        class_index),
                                std::pair<vector<float>, float>(
                                        std::vector<float>(k, 0.0f), // Weights
                                        0.0f) // Bias
                        ));
            }
        }
    }
}

int Cwr::layer_id_by_name(const string &name) {
    return this->layer_names_to_id[name];
}

void Cwr::zeros_cwr_layer_bias_lr(float multiplier) {
    for (auto &it : this->cwr_layers_name) {
        this->set_layer_b_lr(it, 0);
        if(multiplier > 0.0f) {
            this->set_layer_w_lr(it, multiplier);
        }
        //int cwr_layer_id = layer_id_by_name(it);
    }
}

void Cwr::reset_weights() {
    for (auto &it : this->cwr_layers_name) {
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        boost::shared_ptr<Blob<float>> bias;
        bool has_bias = strcmp(layer->type(), "Convolution") != 0 || layer->layer_param().convolution_param().bias_term();
        if(has_bias) {
            bias = layer->blobs()[1];
        }

        int k = weights->shape()[1];

        for (int class_index = 0; class_index < this->max_classes; class_index++) {
            float *weights_data_start = &weights->mutable_cpu_data()[class_index * k];
            //float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];

            memset((void *) weights_data_start, 0, k * sizeof(float));

            if(has_bias) {
                float *bias_data = &bias->mutable_cpu_data()[class_index];
                *bias_data = 1.0f;
            }
        }
    }
}

void Cwr::partial_reset_weights(const std::vector<int> &specific_classes, bool zero_lr) {
    float bias_value = 1.0f;
    if(zero_lr) {
        bias_value = 0.0f;
    }
    for (auto &it : this->cwr_layers_name) {
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        boost::shared_ptr<Blob<float>> bias;
        bool has_bias = strcmp(layer->type(), "Convolution") != 0 || layer->layer_param().convolution_param().bias_term();
        if(has_bias) {
            bias = layer->blobs()[1];
        }

        int k = weights->shape()[1];

        for (auto& class_index : specific_classes) {
            float *weights_data_start = &weights->mutable_cpu_data()[class_index * k];
            //float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];

            memset((void *) weights_data_start, 0, k * sizeof(float));

            if(has_bias) {
                float *bias_data = &bias->mutable_cpu_data()[class_index];
                *bias_data = bias_value;
            }
        }
    }
}

void Cwr::zeros_non_cwr_layers_lr() {
    for (auto &it : this->net->layer_names()) {
        if (std::find(this->cwr_layers_name.begin(), this->cwr_layers_name.end(), it) != this->cwr_layers_name.end()) {
            continue;
        }

        this->set_layer_lr(it, 0);
    }
}

void Cwr::load_weights_nic(const std::vector<int> &train_y) {
    std::stringstream debug_stream;
    std::set<int> class_to_load(train_y.begin(), train_y.end());
    debug_stream << "Loading classes: " << class_to_load << endl;
    log_android_debug(debug_stream);

    for (auto &it : this->cwr_layers_name) {
        debug_stream << "Cwr layer: " << it << endl;
        log_android_debug(debug_stream);
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        const boost::shared_ptr<Blob<float>> bias = layer->blobs()[1];

        int k = weights->shape()[1];

        for (const int& class_index : class_to_load) {
            debug_stream << "Class index: " << class_index << endl;
            log_android_debug(debug_stream);
            float *weights_data_start = &weights->mutable_cpu_data()[class_index * k];
            float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];
            float *bias_data = &bias->mutable_cpu_data()[class_index];

            auto consolidated = this->consolidated_class_weights.at(std::pair<string, int>(it, class_index));

            memcpy(weights_data_start, &consolidated.first[0], k * sizeof(float));
            *bias_data = consolidated.second;
        }
    }
}

void Cwr::set_layer_lr(const std::string &layer_name, float w_lr, float b_lr) {
    if(this->layer_names_to_param_lr_id.find(layer_name) == this->layer_names_to_param_lr_id.end()) { // Layer without params
        return;
    }

    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    this->net->params_lr_[param_offset] = w_lr;
    this->net->params_lr_[param_offset + 1] = b_lr;
}

void Cwr::set_layer_lr(const std::string &layer_name, float fill_r) {
    if(this->layer_names_to_param_lr_id.find(layer_name) == this->layer_names_to_param_lr_id.end()) { // Layer without params
        return;
    }

    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    int n_params = this->net->layer_by_name(layer_name)->blobs().size();

    for (int i = 0; i < n_params; i++) {
        this->net->params_lr_[param_offset + i] = fill_r;
    }
}

void Cwr::set_layer_w_lr(const std::string &layer_name, float w_lr) {
    if(this->layer_names_to_param_lr_id.find(layer_name) == this->layer_names_to_param_lr_id.end()) { // Layer without params
        return;
    }

    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    this->net->params_lr_[param_offset] = w_lr;
}

void Cwr::set_layer_b_lr(const std::string &layer_name, float b_lr) {
    if(this->layer_names_to_param_lr_id.find(layer_name) == this->layer_names_to_param_lr_id.end()) { // Layer without params
        return;
    }

    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    this->net->params_lr_[param_offset + 1] = b_lr;
}

float Cwr::get_layer_b_lr(const std::string &layer_name) {
    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    return this->net->params_lr_[param_offset + 1];
}

float Cwr::get_layer_w_lr(const std::string &layer_name) {
    int param_offset = this->layer_names_to_param_lr_id.at(layer_name);
    return this->net->params_lr_[param_offset];
}
/*
void Cwr::consolidate_weights_cwr(const std::vector<int> &train_y, float contribution) {
    std::set<int> class_to_consolidate(train_y.begin(), train_y.end());
    for (auto &it : this->cwr_layers_name) {
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        const boost::shared_ptr<Blob<float>> bias = layer->blobs()[1];

        int k = weights->shape()[1];

        for (auto &class_index : class_to_consolidate) {
            float *weights_data_start = &weights->mutable_cpu_data()[class_index * k];
            float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];
            float *bias_data = &bias->mutable_cpu_data()[class_index];

            auto consolidated = this->consolidated_class_weights.at(std::pair<string, int>(it, class_index));

            std::vector<float> updated(k);
            float w_lr = weights_data_start[class_index];
            for (int i = 0; i < k; i++) {
                updated[i] = (consolidated.first[i] * this->class_updates[class_index] + w_lr) * contribution /
                             (this->class_updates[class_index] + 1.0);
            }

            auto new_consolidated_weights = this->consolidated_class_weights.find(
                    std::pair<string, int>(it, class_index));
            new_consolidated_weights->second = std::pair<vector<float>, float>(updated, *bias_data);
        }
    }
}
*/

void Cwr::consolidate_weights_cwr_plus(
        const std::vector<int> &class_to_consolidate,
        const std::vector<int> &class_freq) {

    for (auto &it : this->cwr_layers_name) {
        float globavg = 0.0;
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        const boost::shared_ptr<Blob<float>> bias = layer->blobs()[1];

        int k = weights->shape()[1];

        auto sorted_class_to_consolidate = class_to_consolidate;
        std::sort(sorted_class_to_consolidate.begin(), sorted_class_to_consolidate.end());

        for (auto &class_index : sorted_class_to_consolidate) {
            const float *weights_data_start = &weights->cpu_data()[class_index * k];
            for (int weight_index = 0; weight_index < k; weight_index++) {
                globavg += weights_data_start[weight_index];
            }
        }

        globavg /= (float)(k*class_to_consolidate.size());
        //cout << "GlobalAvg ( " << it << " )  " << globavg << " hash: " << hash_generic(&globavg, sizeof(float)) << endl;

        int idx = 0;
        for (auto &class_index : class_to_consolidate) {
            const float *weights_data_start = &weights->cpu_data()[class_index * k];
            //float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];
            const float *bias_data = &bias->cpu_data()[class_index];

            std::pair<std::vector<float>, float> consolidated = this->consolidated_class_weights.at(std::pair<string, int>(it, class_index));

            float prev_weight = sqrt(this->class_updates[class_index] / (float) class_freq[idx]);
            // NO: float prev_weight = this->class_updates[class_index];

            /*cout << "Prev Consolidated: " << hash_vector(consolidated.first) << endl;
            cout << "Prev Weight: " << prev_weight << endl << "--> hash: "
            << hash_generic(&prev_weight, sizeof(float)) << endl;
            cout << "Net weights: " << hash_generic((void*) weights_data_start, sizeof(float) * k) << endl;*/

            std::vector<float> updated(k);
            for (int i = 0; i < k; i++) {
                updated[i] = (consolidated.first[i] * prev_weight + weights_data_start[i] - globavg) /
                        (prev_weight + 1.0f);
            }

            auto new_consolidated_weights = this->consolidated_class_weights.find(
                    std::pair<string, int>(it, class_index));
            new_consolidated_weights->second = std::pair<vector<float>, float>(updated, *bias_data);
            //cout << "Class " << class_index << " consolidated: " << hash_vector(updated) << " bias: " << *bias_data << endl;
            idx++;
        }
    }
}

vector<float> Cwr::get_class_updates() {
    return this->class_updates;
}

void Cwr::set_class_updates(const vector<float> &new_class_updates) {
    this->class_updates = new_class_updates;
}

void Cwr::load_weights() {
    for (auto &it : this->cwr_layers_name) {
        const boost::shared_ptr<Layer<float>> layer = this->net->layer_by_name(it);
        const boost::shared_ptr<Blob<float>> weights = layer->blobs()[0];
        const boost::shared_ptr<Blob<float>> bias = layer->blobs()[1];

        int k = weights->shape()[1];

        for (int class_index = 0; class_index < this->max_classes; class_index++) {
            float *weights_data_start = &weights->mutable_cpu_data()[class_index * k];
            //float *weights_data_end = &weights->mutable_cpu_data()[(class_index + 1) * k - 1];
            float *bias_data = &bias->mutable_cpu_data()[class_index];

            auto consolidated = this->consolidated_class_weights.at(std::pair<string, int>(it, class_index));

            memcpy(weights_data_start, &consolidated.first[0], k * sizeof(float));
            *bias_data = consolidated.second;
        }
    }
}

ostream &operator<<(ostream &os, const Cwr &m) {
    os << "Cwr[" << endl;
    os << "Max classes: " << m.max_classes << ";" << endl;
    os << "Current classes: " << m.current_classes_n << ";" << endl;

    os << "Cwr layers: ";
    for(int i = 0; i < m.cwr_layers_name.size(); i++) {
        os << m.cwr_layers_name[i];
        if(i != (m.cwr_layers_name.size() - 1)) {
            os << ", ";
        }
    }
    os << ";" << endl;

    int cw_size = m.consolidated_class_weights.size();
    os << "Consolidated weights: ";
    for( auto it = m.consolidated_class_weights.begin(); it != m.consolidated_class_weights.end(); ++it, cw_size-- )
    {
        auto key = it->first;
        auto value = it->second;

        os << "(" << key.first << ", " << setw(3) << right << key.second << ") -> [w: ";
        for(int i = 0; i < value.first.size(); i++) {
            os << setw(10) << fixed << setprecision(7) << right << value.first[i];
            if(i != (value.first.size() - 1)) {
                os << ", ";
            }
        }
        os << "; b: " << value.second << " ]";

        if(cw_size != 1) {
            os << endl;
        }
    }
    os << "]" << endl;

    return os;
}

void Cwr::create_layers_name_to_id_mapping() {
    this->layer_names_to_id = generate_layers_name_to_id_mapping(this->net);
}

void Cwr::create_layers_name_to_param_lr_id_mapping() {
    //this->layer_names_to_param_lr_id.clear();

    if(this->net.get() == nullptr) {
        return;
    }

    this->layer_names_to_param_lr_id.clear();

    auto layers = this->net->layers();
    //int layer_index = 0;
    int lr_entry = 0;
    for(int i = 0; i < layers.size(); i++) {
        auto lr = layers[i];
        int blobs_sz = lr->blobs().size();

        if(blobs_sz > 0) {
            this->layer_names_to_param_lr_id.insert(std::pair<std::string, int>(this->net->layer_names()[i], lr_entry));
        }

        lr_entry += blobs_sz;
    }
}

Cwr &Cwr::operator=(const Cwr &other) {
    if (this != &other) { // protect against invalid self-assignment
        this->net = other.net;
        this->layer_names_to_id = other.layer_names_to_id;
        this->layer_names_to_param_lr_id = other.layer_names_to_param_lr_id;
        this->cwr_layers_name = other.cwr_layers_name;
        this->max_classes = other.max_classes;
        this->current_classes_n = other.current_classes_n;
        this->class_updates = other.class_updates;
        this->consolidated_class_weights = other.consolidated_class_weights;
    }

    return *this;
}

Cwr::Cwr(const Cwr &other) {
    //Checks: len(class_updates) == max_classes
    this->class_updates = other.class_updates;
    this->max_classes = other.max_classes;
    this->current_classes_n = other.current_classes_n;
    this->net = other.net;
    this->cwr_layers_name = other.cwr_layers_name;
    this->layer_names_to_id = other.layer_names_to_id;
    this->layer_names_to_param_lr_id = other.layer_names_to_param_lr_id;
    this->consolidated_class_weights = other.consolidated_class_weights;
}

void Cwr::release_net_ref() {
    this->net.reset();
}

void Cwr::set_net_ref(boost::shared_ptr<Net<float>> &new_net) {
    this->net = new_net;
    this->create_layers_name_to_id_mapping();
    this->create_layers_name_to_param_lr_id_mapping();
}

int Cwr::get_max_classes() {
    return this->max_classes;
}

std::vector<std::string> Cwr::get_cwr_layers_name() {
    return this->cwr_layers_name;
}

int Cwr::get_current_classes_n() {
    return this->current_classes_n;
}

int Cwr::increment_class_number() {
    this->class_updates.push_back(0);
    this->current_classes_n++;
    return this->current_classes_n-1;
}

int Cwr::get_net_max_classes(const boost::shared_ptr<Net<float>> &net, const string &layer_name) {
    return net->blob_by_name(layer_name)->shape()[1];
}

void Cwr::set_brn_past_weight(float weight) {
    int n_brn_layers = 0;
    for( auto it = this->layer_names_to_id.begin(); it != this->layer_names_to_id.end(); ++it )
    {
        string layer_name = it->first;
        int layer_id = it->second;
        boost::shared_ptr<caffe::Layer<float>> layer = this->net->layers()[layer_id];
        string layer_type_str(layer->type());

        if(layer_type_str == "BatchReNorm") {
            n_brn_layers++;
            std::vector<boost::shared_ptr<caffe::Blob<float>>> params = layer->blobs();

            float scale = weight / params[2]->cpu_data()[0];

            int param_sz = params[0]->count();
            float* param_data = params[0]->mutable_cpu_data();
            for(int i = 0; i < param_sz; i ++) {
                param_data[i] = param_data[i] * scale;
            }

            param_sz = params[1]->count();
            param_data = params[1]->mutable_cpu_data();
            for(int i = 0; i < param_sz; i ++) {
                param_data[i] = param_data[i] * scale;
            }

            params[2]->mutable_cpu_data()[0] = weight;
        }

    }
}


