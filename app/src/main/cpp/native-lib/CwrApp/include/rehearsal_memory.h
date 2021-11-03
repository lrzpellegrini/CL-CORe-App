#ifndef CAFFE_CWR_TO_CPP_REHEARSAL_MEMORY_H
#define CAFFE_CWR_TO_CPP_REHEARSAL_MEMORY_H

#include <memory>
#include <vector>
#include <map>

template<class T>
class RehearsalMemory {
public:
    RehearsalMemory() = default;
    explicit RehearsalMemory(int max_num_patterns);

    /**
     * Set the maximum replay buffer size.
     *
     * @param new_max_num_patterns The new replay buffer size.
     */
    void set_max_num_patterns(int new_max_num_patterns);

    /**
     * Update the replay buffer.
     *
     * @param x_data The X data from the current batch.
     * @param y_data The Y data from the current batch.
     * @param batch The batch number.
     * @param fixed_size_replacement If not -1, the amount of instances to replace.
     */
    void update_memory(std::vector<T> x_data,
            std::vector<int> y_data,
            int batch,
            int fixed_size_replacement=-1);

    /**
      * Update the replay buffer.
      *
      * This method tries to keep the replay buffer balanced.
      *
      * @param x_data The X data from the current batch.
      * @param y_data The Y data from the current batch.
      * @param batch The batch number.
      * @param fixed_size_replacement If not -1, the amount of instances to replace. If -1, then
      *        then both replay and new instances will be picked randomly.
      */
    void update_memory_balanced(std::vector<T> x_data,
                                std::vector<int> y_data,
                                int batch,
                                int fixed_size_replacement=-1);

    /**
     * Load the initial replay buffer.
     *
     * @param x_data The X data of replay instances.
     * @param y_data The Y data of replay instances.
     */
    void load_memory(std::vector<T> x_data, std::vector<int> y_data);

    /**
     * Get the X data of replay instances.
     *
     * @return The X data of replay instances.
     */
    std::vector<T> getSamplesX();

    /**
     * Get the labels of the replay instances.
     *
     * @return The category labels of replay instances.
     */
    std::vector<int> getSamplesY();

    /**
     * Serialize the replay buffer.
     *
     * @param out_stream The target output stream.
     */
    void save_everything(std::basic_fstream<char, std::char_traits<char>> &out_stream);

protected:
    int current_num_patterns; // Persistent
    int max_num_patterns; // Persistent
    std::vector<T> external_memory_x; // Persistent
    std::vector<int> external_memory_y; // Persistent

    void adapt_memory();

    static std::vector<int> random_indexes_without_replacement(int array_size, int n_elements, long seed);

    int n_classes();
};

template<typename TK, typename TV>
std::vector<TK> extract_keys(const std::map<TK, TV> &input_map) {
    std::vector<TK> retval;
    for (auto const& element : input_map) {
        retval.push_back(element.first);
    }
    return retval;
}

#endif //CAFFE_CWR_TO_CPP_REHEARSAL_MEMORY_H
