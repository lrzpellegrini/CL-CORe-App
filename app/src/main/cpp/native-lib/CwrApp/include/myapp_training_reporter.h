#ifndef NATIVE_CAFFE_APP_MYAPP_TRAINING_REPORTER_H
#define NATIVE_CAFFE_APP_MYAPP_TRAINING_REPORTER_H
#include <training_status_reporter.h>
#include <jni.h>

class MyAppTrainingStatusReporter : public TrainingProgressListener {
public:
    MyAppTrainingStatusReporter() = default;
    MyAppTrainingStatusReporter(JNIEnv* env, jobject appObject, jmethodID callbackMethod);
    void updateProgress(float progress);
protected:
    JNIEnv* env;
    jobject appObject;
    jmethodID callbackMethod;
};

#endif //NATIVE_CAFFE_APP_MYAPP_TRAINING_REPORTER_H
