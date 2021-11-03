#include <iostream>
#include <train_utils.h>
#include <model_status_reporter.h>

ModelStatusReporter::ModelStatusReporter(
        reporter_logging_function logFunction,
        const boost::shared_ptr<caffe::Net<float>>& initialNet,
        bool includeLrs, bool separateLrs, bool enabled) {
    this->enabled = enabled;
    if(enabled) {
        this->loggingFunction = logFunction;
        this->net = initialNet;
        this->includeLearningRates = includeLrs;
        this->printSeparateLearningRates = separateLrs;
        this->lastNetHash = hash_net(initialNet, includeLrs);

        this->loggingFunction("Changing net: " + this->lastNetHash + "\n");
    }
}


ModelStatusReporter::ModelStatusReporter(
        reporter_logging_function logFunction,
        bool includeLrs, bool separateLrs, bool enabled) {
    this->enabled = enabled;
    if(enabled) {
        this->loggingFunction = logFunction;
        this->includeLearningRates = includeLrs;
        this->printSeparateLearningRates = separateLrs;
    }
}

boost::shared_ptr<caffe::Net<float>> ModelStatusReporter::getNet() {
    return this->net;
}

void ModelStatusReporter::onPhaseFinished(const std::string &afterPhaseName) {
    if(!enabled) {
        return;
    }

    std::string logData, netHash, phaseTabs;
    logData = phaseTabs = this->getCurrentPhaseTabs();
    this->popPhase(afterPhaseName);

    netHash = hash_net(this->getNet(), this->includeLearningRates);
    if(netHash == this->lastNetHash) {
        return;
    }

    this->lastNetHash = netHash;

    logData += "[" + afterPhaseName + "] after: ";
    logData += netHash;
    logData += "\n";

    if(printSeparateLearningRates) {
        logData += phaseTabs;
        logData += "[" + afterPhaseName + "] --> LR: ";
        logData += hash_net_lrs(this->getNet());
        logData += "\n";
    }

    this->loggingFunction(logData);
}

void ModelStatusReporter::startChangePhase(const std::string &phaseName) {
    if(!enabled) {
        return;
    }

    std::string logData, netHash;
    this->pushPhase(phaseName);
    logData = this->getCurrentPhaseTabs();

    netHash = hash_net(this->getNet(), this->includeLearningRates);
    if(netHash != lastNetHash) {
        this->lastNetHash = netHash;
        logData += "!! Unexpected net change !!\n";
        logData += this->getCurrentPhaseTabs();
    }

    logData += "[" + phaseName + "] before: ";
    logData += netHash;
    logData += "\n";

    if(printSeparateLearningRates) {
        logData += this->getCurrentPhaseTabs();
        logData += "[" + phaseName + "] --> LR: ";
        logData += hash_net_lrs(this->getNet());
        logData += "\n";
    }

    this->loggingFunction(logData);
}

void ModelStatusReporter::popPhase(const std::string &phaseName) {
    if(this->currentPhases.size() == 0) {
        this->loggingFunction("Unexpected phase (after): " + phaseName + "\n");
        return;
    }

    if(phaseName == this->currentPhases[this->currentPhases.size()-1]) {
        this->currentPhases.pop_back();
        return;
    }

    bool found = false;
    int n_elements = 0;
    for(int i = this->currentPhases.size()-1; i >= 0 && !found; i--) {
        n_elements++;
        if(phaseName == this->currentPhases[i]) {
            found = true;

            for(int j = 0; j < n_elements; j++) {
                this->currentPhases.pop_back();
            }
        }
    }

    if(!found) {
        this->loggingFunction("Unexpected phase (after): " + phaseName + "\n");
    } else {
        this->loggingFunction("Multiple closed phases! Skipping to " + phaseName + "\n");
    }
}

void ModelStatusReporter::pushPhase(const std::string &phaseName) {
    this->currentPhases.push_back(phaseName);
}

std::string ModelStatusReporter::getCurrentPhaseTabs() {
    std::string tabs;
    for(int i = 0; i < this->currentPhases.size()-1; i++) {
        tabs += "  ";
    }

    return tabs;
}

void ModelStatusReporter::changeNet(const boost::shared_ptr<caffe::Net<float>> &newNet) {
    if(!enabled) {
        return;
    }

    this->net = newNet;
    this->lastNetHash = hash_net(newNet, this->includeLearningRates);

    this->loggingFunction("Changing net: " + this->lastNetHash + "\n");
}

void cout_reporter_function(const std::string &toBeLogged) {
    std::cout << toBeLogged;
}
