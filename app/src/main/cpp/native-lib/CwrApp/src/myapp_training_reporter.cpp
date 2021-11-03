#include <myapp_training_reporter.h>

MyAppTrainingStatusReporter::MyAppTrainingStatusReporter(JNIEnv *env, jobject appObject,
        jmethodID callbackMethod) {
    this->env = env;
    this->appObject = appObject;
    this->callbackMethod = callbackMethod;
}

void MyAppTrainingStatusReporter::updateProgress(float progress) {
    env->CallVoidMethod(this->appObject, this->callbackMethod, progress);
}
