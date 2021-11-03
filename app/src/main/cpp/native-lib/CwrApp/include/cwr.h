//
// Created by lorenzo on 05/04/19.
//

#ifndef CAFFE_CWR_TO_CPP_CWR_H
#define CAFFE_CWR_TO_CPP_CWR_H

#include <set>
#include <vector>
#include <memory>
#include <map>
//#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <fstream>

// include headers that implement a archive in simple text format
#include <boost/serialization/map.hpp>
//#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

class Cwr {
public:
    Cwr(int initial_classes_n,
            const vector<float>& initial_class_updates,
            const boost::shared_ptr<Net<float>>& net,
            const vector<string>& cwr_layers_name,
            bool load_weights_from_net = false);

    Cwr(const Cwr& other);

    Cwr(const Cwr& from,
        const boost::shared_ptr<Net<float>>& new_net);

    Cwr();
    virtual ~Cwr();

    Cwr& operator=(const Cwr& other);

    void reset_weights();
    void partial_reset_weights(const std::vector<int>& specific_classes, bool zero_lr=false);
    void load_weights();
    void load_weights_nic(const std::vector<int>& train_y);
    void consolidate_weights_cwr_plus(const std::vector<int> &class_to_consolidate,
                                      const std::vector<int> &class_freq);
    //void consolidate_weights_cwr(const std::vector<int>& train_y, float contribution);
    void zeros_cwr_layer_bias_lr(float multiplier);
    void zeros_non_cwr_layers_lr();
    vector<float> get_class_updates();
    void set_class_updates(const vector<float>& new_class_updates);
    void release_net_ref();
    void set_net_ref(boost::shared_ptr<Net<float>>& new_net);
    int get_max_classes();
    int get_current_classes_n();
    int increment_class_number();
    std::vector<std::string> get_cwr_layers_name();

    void set_brn_past_weight(float weight);

    static int get_net_max_classes(const boost::shared_ptr<Net<float>> &net, const string &layer_name);

    friend std::ostream& operator<<(std::ostream&, const Cwr&);

private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        //TODO: fix serialization and deserialization (or just remove it)
        ar & cwr_layers_name;
        ar & max_classes;
        ar & current_classes_n;
        ar & class_updates;
        ar & consolidated_class_weights;
    }

    boost::shared_ptr<Net<float>> net; // transient
    unordered_map<string, int> layer_names_to_id; // transient
    unordered_map<string, int> layer_names_to_param_lr_id; // transient

    vector<string> cwr_layers_name;
    int max_classes;
    int current_classes_n;
    vector<float> class_updates;
    map<std::pair<string, int>, std::pair<vector<float>, float>> consolidated_class_weights;

    void init_consolidated_weights(bool from_net);

    int layer_id_by_name(const string &name);
    /*int find_layer_param_index(int layer_id);*/

    //void set_layer_lr(int layer_id, float w_lr, float b_lr);
    void set_layer_lr(const string &layer_name, float w_lr, float b_lr);
    void set_layer_lr(const string &layer_name, float fill_r);

    //void set_layer_w_lr(int layer_id, float w_lr);
    void set_layer_w_lr(const string &layer_name, float w_lr);

    //void set_layer_b_lr(int layer_id, float b_lr);
    void set_layer_b_lr(const string &layer_name, float b_lr);

    //float get_layer_b_lr(int layer_id);
    float get_layer_b_lr(const string &layer_name);

    //float get_layer_w_lr(int layer_id);
    float get_layer_w_lr(const string &layer_name);

    void create_layers_name_to_id_mapping();
    void create_layers_name_to_param_lr_id_mapping();

};

std::ostream &operator<<(std::ostream &os, const Cwr &m);

#endif //CAFFE_CWR_TO_CPP_CWR_H