#ifndef CAFFE_CWR_TO_CPP_MODEL_STATUS_REPORTER_H
#define CAFFE_CWR_TO_CPP_MODEL_STATUS_REPORTER_H
#include <caffe/caffe.hpp>

typedef void (*reporter_logging_function)(const std::string &toBeLogged);

void cout_reporter_function(const std::string &toBeLogged);

class ModelStatusReporter {
public:
    ModelStatusReporter(
            reporter_logging_function logFunction,
            const boost::shared_ptr<caffe::Net<float>>& initialNet,
            bool includeLrs, bool separateLrs, bool enabled = true);
    ModelStatusReporter(
            reporter_logging_function logFunction,
            bool includeLrs, bool separateLrs, bool enabled = true);
    boost::shared_ptr<caffe::Net<float>> getNet();

    void startChangePhase(const std::string &phaseName);
    void onPhaseFinished(const std::string &afterPhaseName);
    void changeNet(const boost::shared_ptr<caffe::Net<float>>& newNet);
private:
    bool enabled;
    bool includeLearningRates;
    bool printSeparateLearningRates;
    reporter_logging_function loggingFunction;
    boost::shared_ptr<caffe::Net<float>> net;
    std::vector<std::string> currentPhases;
    std::string lastNetHash;

    void popPhase(const std::string &phaseName);
    void pushPhase(const std::string &phaseName);
    std::string getCurrentPhaseTabs();
};

#endif //CAFFE_CWR_TO_CPP_MODEL_STATUS_REPORTER_H
