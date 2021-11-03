#include <rehearsal_memory.h>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cwr_app.h>

#ifndef __ANDROID__
#include <randomkit.h>
#endif

using namespace std;

template <class T>
RehearsalMemory<T>::RehearsalMemory(int max_num_patterns) {
    this->current_num_patterns = 0;
    this->max_num_patterns = max_num_patterns;
    this->adapt_memory();
    cout << "Rehearsal constructor: " << this->current_num_patterns << " " << this->max_num_patterns << endl;
}


template <class T>
void RehearsalMemory<T>::set_max_num_patterns(int new_max_num_patterns) {
    this->max_num_patterns = new_max_num_patterns;
    this->adapt_memory();
}


template <class T>
void RehearsalMemory<T>::adapt_memory() {
    if(this->external_memory_x.size() > this->max_num_patterns) {
        this->external_memory_x.resize(this->max_num_patterns);
        this->external_memory_y.resize(this->max_num_patterns);
    }

    if(this->current_num_patterns > this->max_num_patterns) {
        this->current_num_patterns = this->max_num_patterns;
    }
}


template <class T>
void RehearsalMemory<T>::update_memory(
        std::vector<T> x_data, std::vector<int> y_data,
        int batch, int fixed_size_replacement) {
    int n_cur_batch;
    if(fixed_size_replacement >= 0) {
        n_cur_batch = fixed_size_replacement;
    } else {
        n_cur_batch = max_num_patterns / (batch+1); //n_cur_batch = ExtMemSize // (batch + 1)
    }
    std::cout << n_cur_batch << std::endl;

    if(n_cur_batch > x_data.size()) { // if n_cur_batch > train_x.shape[0]:
        n_cur_batch = x_data.size(); // n_cur_batch = train_x.shape[0]
    }
    std::cout << n_cur_batch << std::endl;
    int n_ext_mem = max_num_patterns - n_cur_batch; // n_ext_mem = ExtMemSize - n_cur_batch
    std::cout << n_ext_mem << std::endl;

    std::vector<int> idxs_cur = // idxs_cur = np.random.choice(train_x.shape[0], n_cur_batch, replace=False)
            random_indexes_without_replacement(x_data.size(), n_cur_batch, 1);
    std::cout << n_cur_batch << std::endl;
    std::cout << n_ext_mem << std::endl;

    if(n_ext_mem == 0) {
        current_num_patterns = idxs_cur.size();
        external_memory_x.clear();
        external_memory_y.clear();
        for(int i = 0; i < idxs_cur.size(); i++) {
            external_memory_x.push_back(x_data[idxs_cur[i]]);
            external_memory_y.push_back(y_data[idxs_cur[i]]);
        }
    } else {
        std::vector<int> idxs_ext = // idxs_ext = np.random.choice(ExtMemSize, n_ext_mem, replace=False)
                random_indexes_without_replacement(max_num_patterns, n_ext_mem, 0);
        std::vector<T> new_external_memory_x;
        std::vector<int> new_external_memory_y;
        for(int i = 0; i < idxs_cur.size(); i++) {
            new_external_memory_x.push_back(x_data[idxs_cur[i]]);
            new_external_memory_y.push_back(y_data[idxs_cur[i]]);
        }

        for(int i = 0; i < idxs_ext.size(); i++) {
            new_external_memory_x.push_back(external_memory_x[idxs_ext[i]]);
            new_external_memory_y.push_back(external_memory_y[idxs_ext[i]]);
        }

        external_memory_x = new_external_memory_x;
        external_memory_y = new_external_memory_y;
    }
}

template <class T>
void RehearsalMemory<T>::update_memory_balanced(
        std::vector<T> x_data, std::vector<int> y_data,
        int batch, int fixed_size_replacement) {
    std::stringstream debug_str;
    std::map<int, std::vector<int>> cl_idxs;
    std::vector<T> new_x_data;
    std::vector<int> new_y_data;
    std::map<int, int> samples_per_cls;
    int new_inst_idx = 0;

    for(; new_inst_idx < this->current_num_patterns; new_inst_idx++) {
        new_x_data.push_back(this->external_memory_x[new_inst_idx]);
        int curr_y = this->external_memory_y[new_inst_idx];
        new_y_data.push_back(curr_y);

        if (cl_idxs.find(curr_y) == cl_idxs.end() ) {
            cl_idxs[curr_y] = std::vector<int>();
        }
        cl_idxs[curr_y].push_back(new_inst_idx);
    }

    for (int idx = 0; idx < y_data.size(); idx++, new_inst_idx++) {
        new_x_data.push_back(x_data[idx]);
        int curr_y = y_data[idx];
        new_y_data.push_back(curr_y);

        if (cl_idxs.find(curr_y) == cl_idxs.end() ) {
            // not found
            cl_idxs[curr_y] = std::vector<int>();
        }
        cl_idxs[curr_y].push_back(new_inst_idx);
    }

    int max_curr_cls = *std::max_element(
            new_y_data.begin(),
            new_y_data.end());

    int n_seen_classes = max_curr_cls + 1;
    int class_mem_size = this->max_num_patterns / n_seen_classes;

    // Similar to Avalanche "divide_remaining_samples"
    // Compute the amount of samples per cls
    debug_str << "Starting with the following replay buffer:" << std::endl;
    int rem_from_cls = 0;
    for (int cls_id = 0; cls_id < n_seen_classes; cls_id++) {
        debug_str << "Class " << cls_id << " has "
            << cl_idxs[cls_id].size() << "instances" << std::endl;
        samples_per_cls[cls_id] = cl_idxs[cls_id].size();
    }

    // Compute the amount of remaining slots
    for (int cls_id = 0; cls_id < n_seen_classes; cls_id++) {
        int this_class_remaining = class_mem_size - samples_per_cls[cls_id];
        if(this_class_remaining > 0) {
            rem_from_cls += this_class_remaining;
        }
    }

    int free_mem = rem_from_cls + this->max_num_patterns % n_seen_classes;

    // Compute per-class cutoff and remaining samples
    std::map<int, int> cutoff_per_cls;
    std::map<int, int> rem_samples_cls;
    for (int cls_id = 0; cls_id < n_seen_classes; cls_id++) {
        int this_cls_memsize = samples_per_cls[cls_id];
        cutoff_per_cls[cls_id] = std::min(
                class_mem_size,
                this_cls_memsize);

        int remaining_to_divide = this_cls_memsize - class_mem_size;
        if (remaining_to_divide > 0) {
            // If has more than "class_mem_size" (which is the per-class max)
            rem_samples_cls[cls_id] = remaining_to_divide;
        }
    }

    // Allocate free slots to other classes
    std::mt19937 generator(2);
    while((!rem_samples_cls.empty()) && (free_mem > 0)) {
        std::vector<int> available_keys = extract_keys(rem_samples_cls);
        int rnd_cls_idx = (int) (generator() % available_keys.size());
        int cls_id = available_keys[rnd_cls_idx];
        cutoff_per_cls[cls_id] += 1;
        free_mem -= 1;
        rem_samples_cls[cls_id] -= 1;
        if(rem_samples_cls[cls_id] <= 0) {
            rem_samples_cls.erase(cls_id);
        }
    }

    std::vector<T> final_x_data;
    std::vector<int> final_y_data;
    for(int cls_id = 0; cls_id < n_seen_classes; cls_id++) {
        debug_str << "Allocating " <<  cutoff_per_cls[cls_id] <<
            " slots to class " << cls_id << std::endl;
        std::vector<int> to_keep_indices = random_indexes_without_replacement(
                cl_idxs[cls_id].size(),
                cutoff_per_cls[cls_id],
                3);

        for(int instance_idx_idx : to_keep_indices) {
            int instance_idx = cl_idxs[cls_id][instance_idx_idx];
            T inst_x = new_x_data[instance_idx];
            int inst_y = new_y_data[instance_idx];
            final_x_data.push_back(inst_x);
            final_y_data.push_back(inst_y);

            if(inst_y != cls_id) {
                debug_str << "Something went wrong: " << inst_y
                    << " != " << cls_id << std::endl;
            }
        }
    }

    this->external_memory_x = final_x_data;
    this->external_memory_y = final_y_data;
    this->current_num_patterns = this->external_memory_x.size();
    log_android_debug(debug_str);
}

template <class T>
int RehearsalMemory<T>::n_classes() {
    int max_cls_id = *std::max_element(
            this->external_memory_y.begin(),
            this->external_memory_y.end());
    return max_cls_id + 1;
}

#ifdef __ANDROID__
template <class T>
std::vector<int> RehearsalMemory<T>::random_indexes_without_replacement(int array_size,
                                        int n_elements,
                                        long seed) {

    std::vector<int> indexes;
    for(int i = 0; i < array_size; i++) {
        indexes.push_back(i);
    }

    std::mt19937 generator(seed);

    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = indexes.size()-1; i >= 1; i--) {
        int j = (int) (generator() % i);
        if (i == j) {
            continue;
        }

        auto tmp = indexes[j];
        indexes[j] = indexes[i];
        indexes[i] = tmp;
    }

    indexes.resize(n_elements);
    return indexes;
}
#else
template <class T>
std::vector<int> RehearsalMemory<T>::random_indexes_without_replacement(
        int array_size,
        int n_elements,
        long seed) {

    std::vector<int> indexes;
    for(int i = 0; i < array_size; i++) {
        indexes.push_back(i);
    }

    auto state = (rk_state*) malloc(sizeof(rk_state));
    rk_seed(seed, state);

    //https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    for (int i = indexes.size()-1; i >= 1; i--) {
        int j = rk_interval(i, state);
        if (i == j) {
            continue;
        }

        auto tmp = indexes[j];
        indexes[j] = indexes[i];
        indexes[i] = tmp;
    }

    free(state);

    indexes.resize(n_elements);
    return indexes;
}
#endif

template <class T>
std::vector<T> RehearsalMemory<T>::getSamplesX() {
    return this->external_memory_x;
}

template <class T>
std::vector<int> RehearsalMemory<T>::getSamplesY() {
    return this->external_memory_y;
}

template<class T>
void RehearsalMemory<T>::load_memory(vector<T> x_data, std::vector<int> y_data) {
    current_num_patterns = x_data.size();
    external_memory_x.clear();
    external_memory_y.clear();

    int max_elements = std::min((int) x_data.size(), this->max_num_patterns);
    for(int i = 0; i < max_elements; i++) {
        external_memory_x.push_back(x_data[i]);
        external_memory_y.push_back(y_data[i]);
    }
}

template<> void RehearsalMemory<cwra_rehe_t>::save_everything(std::basic_fstream<char, std::char_traits<char>> &out_stream) {
    std::stringstream debug_str;
    debug_str << "rehe memory, save_everything:" << endl;

    const uint32_t current_num_patterns = this->current_num_patterns;
    const uint32_t max_num_patterns = this->max_num_patterns;
    const std::vector<cwra_rehe_t>& external_memory_x = this->external_memory_x;
    const std::vector<int>& external_memory_y = this->external_memory_y;

    serialize_uint32t(current_num_patterns, out_stream);
    debug_str << current_num_patterns << ", ";

    serialize_uint32t(max_num_patterns, out_stream);
    debug_str << max_num_patterns << ", ";

    serialize_rehearsal_blob(external_memory_x, out_stream);
    debug_str << "External memory x hash: " << hash_rehearsal_memory(external_memory_x) << "," << endl;

    serialize_int_vector(external_memory_y, out_stream);
    debug_str << "External memory y hash: " << hash_vector(external_memory_y) << "," << endl;

    log_android_debug( debug_str );
}

//The explicit instantiation part
template class RehearsalMemory<std::shared_ptr<std::vector<float>>>;
template class RehearsalMemory<cv::Mat>;
