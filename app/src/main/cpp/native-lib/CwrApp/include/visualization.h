#ifndef CAFFE_CWR_TO_CPP_VISUALIZATION_H
#define CAFFE_CWR_TO_CPP_VISUALIZATION_H

#include <iostream>
#include <caffe/caffe.hpp>
#include <boost/shared_ptr.hpp>

void PrintShape(const std::vector<int>& shape, std::ostream& out_stream);

void PrintLayerShapes(const boost::shared_ptr<caffe::Net<float>>& net, std::ostream& out_stream);

void PrintFiltersBias(const boost::shared_ptr<caffe::Net<float>>& net, std::ostream& out_stream);

void PrintNetworkArchitecture(const boost::shared_ptr<caffe::Net<float>>& net, std::ostream& out_stream);

#endif //CAFFE_CWR_TO_CPP_VISUALIZATION_H
