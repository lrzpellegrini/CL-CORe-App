#ifndef NATIVE_CAFFE_APP_TRAINING_STATUS_REPORTER_H
#define NATIVE_CAFFE_APP_TRAINING_STATUS_REPORTER_H

class TrainingProgressListener {
public:
    virtual void updateProgress(float progress) = 0;
};
#endif //NATIVE_CAFFE_APP_TRAINING_STATUS_REPORTER_H
