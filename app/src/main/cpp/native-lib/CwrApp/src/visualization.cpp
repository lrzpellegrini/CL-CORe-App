#include <visualization.h>
using namespace std;
using namespace caffe;

void PrintShape(const std::vector<int> &shape, std::ostream &out_stream) {
    out_stream << '(';

    if(shape.size() > 1) {
        for (int i = 0; i < shape.size(); i++) {
            out_stream << shape[i];
            if (i != shape.size() - 1) {
                out_stream << ", ";
            }
        }
    } else if(shape.size() == 1) {
        out_stream << shape[0] << ',';
    }

    out_stream << ')';
}

void PrintLayerShapes(const boost::shared_ptr<Net<float>>& net, ostream &out_stream) {
    for(int i = 0; i < net->blobs().size(); i++) {
        out_stream << net->blob_names()[i] << '\t';
        PrintShape(net->blobs()[i]->shape(), out_stream);
        out_stream << endl;
    }
}

void PrintFiltersBias(const boost::shared_ptr<caffe::Net<float>> &net, std::ostream &out_stream) {
    auto layers = net->layers();
    for(int i = 0; i < layers.size(); i++) {
        auto lr = layers[i];
        if(lr->blobs().size() == 2) {
            out_stream << net->layer_names()[i] << '\t';
            PrintShape(lr->blobs()[0]->shape(), out_stream);
            out_stream << ' ';
            PrintShape(lr->blobs()[1]->shape(), out_stream);
        } else if(lr->blobs().size() == 1) {
            out_stream << net->layer_names()[i] << '\t';
            PrintShape(lr->blobs()[0]->shape(), out_stream);
        } else {
            continue;
        }

        out_stream << endl;
    }

}

void PrintNetworkArchitecture(const boost::shared_ptr<caffe::Net<float>> &net, std::ostream &out_stream) {
    out_stream << "NETWORK ARCHITECTURE" << endl;
    out_stream << "Layers" << endl;
    PrintLayerShapes(net, out_stream);
    out_stream << "Filter, Bias" << endl;
    PrintFiltersBias(net, out_stream);
    out_stream << endl;
}
