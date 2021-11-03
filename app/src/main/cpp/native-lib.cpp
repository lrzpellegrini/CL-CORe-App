#include <jni.h>
#include <cstddef>
#include <string>
#include <snappy.h>
#include <lmdb.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <android/log.h>
#include <caffe/caffe.hpp>
#include <cblas.h>
#include <stdio.h>
#include <gflags/gflags.h>
#include <boost/math/common_factor.hpp>
#include <cwr_app.h>
#include <train_utils.h>
#include <cblas.h>
#include <myapp_training_reporter.h>

using namespace caffe;
using namespace snappy;
using namespace cv;
using namespace std;
using namespace std::chrono;

bool writeImageToDisk(const string& fileName, const cv::Mat& image);
Mat androidYUVToOpenCVMat(JNIEnv *env,
        const std::vector<char>& y_buffer,
        const std::vector<char>& u_buffer,
        const std::vector<char>& v_buffer,
        int width, int height,
        const Rect& cropRect);

Mat androidYUVToOpenCVMatEx(JNIEnv *env,
        const std::vector<char> &nv_21,
        int width, int height,
        const Rect& cropRect);

std::vector<char> YUV_420_888toNV21Ex(char* y_buffer,
                                      int y_size,
                                      char* u_buffer,
                                      int u_size,
                                      char* v_buffer,
                                      int v_size,
                                      int width, int height);
Mat rotateImage(const Mat& image, int rotation);
Mat adaptImageSizeForNetInput(
        const Mat &image,
        Net<float> *model,
        int input_blob_index,
        const Rect& crop_rect);


std::vector<string> java_string_array_to_std_vector(JNIEnv *env, jobjectArray stringArray);
std::string java_string_to_std_string(JNIEnv *env, jstring javaString);

void initializeJavaRefs(JNIEnv *env);

bool areJavaRefsInitialized = false;
jclass globalYuvImageClass, globalRectClass, globalBosClass, globalMyApplicationClass;
jmethodID globalYuvConstructor, globalRectConstructor, globalBosConstructor,
        globalCompress_to_jpeg, globalTo_byte_array, global_set_training_progress;

CwrApp cwr_app;

bool writeImageToDisk(const string& fileName, const cv::Mat& image) {
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    try {
        __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "Writing image to: %s", fileName.c_str());
        imwrite(fileName, image, compression_params);
    }
    catch (runtime_error& ex) {
        __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "Exception converting image to JPEG format: %s", ex.what());
        //fprintf(stderr, "Exception converting image to JPEG format: %s\n", ex.what());
        return false;
    }

    return true;
}

std::vector<char> YUV_420_888toNV21(const std::vector<char>& y_buffer,
                                    const std::vector<char>& u_buffer,
                                    const std::vector<char>& v_buffer,
                                    int width, int height) {
    std::vector<char> nv21;

    std::vector<char>::size_type y_size = y_buffer.size();
    std::vector<char>::size_type u_size = u_buffer.size();
    std::vector<char>::size_type v_size = v_buffer.size();

    nv21.resize(y_size + u_size + v_size);

    memcpy(&nv21[0], &y_buffer[0], y_size);
    memcpy(&nv21[y_size], &v_buffer[0], v_size);
    memcpy(&nv21[y_size + v_size], &u_buffer[0], u_size);

    return nv21;
}

std::vector<char> YUV_420_888toNV21Ex(char* y_buffer,
                                    int y_size,
                                    char* u_buffer,
                                    int u_size,
                                    char* v_buffer,
                                    int v_size,
                                    int width, int height) {
    std::vector<char> nv21;

    nv21.resize(y_size + u_size + v_size);

    memcpy(&nv21[0], y_buffer, y_size);
    memcpy(&nv21[y_size], v_buffer, v_size);
    memcpy(&nv21[y_size + v_size], u_buffer, u_size);

    return nv21;
}

Mat androidYUVToOpenCVMat(JNIEnv *env,
        const std::vector<char>& y_buffer,
                          const std::vector<char>& u_buffer,
                          const std::vector<char>& v_buffer,
                          int width, int height,
                          const Rect& cropRect) {
    std::stringstream debug_str;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Conversion
    std::vector<char> nv21 = YUV_420_888toNV21(y_buffer, u_buffer, v_buffer, width, height);

    //cv::Mat bgr;
    initializeJavaRefs(env);

    jobject rect_instance = env->NewObject(globalRectClass, globalRectConstructor,
            cropRect.x, cropRect.y, cropRect.x+cropRect.width, cropRect.y+cropRect.height);
    jobject bos_instance = env->NewObject(globalBosClass, globalBosConstructor);

    jbyteArray nv21_java_array = env->NewByteArray(nv21.size());
    env->SetByteArrayRegion(nv21_java_array, 0, nv21.size(), (jbyte*) &nv21[0]);

    jintArray intJavaArray = env->NewIntArray(2);
    int intCArray[2];
    intCArray[0] = width;
    intCArray[1] = width;
    env->SetIntArrayRegion(intJavaArray, 0, 2, intCArray);

    jobject yuv_image_instance = env->NewObject(globalYuvImageClass, globalYuvConstructor,
                                                nv21_java_array, 17, width, height, NULL);

    //yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);

    env->CallBooleanMethod(yuv_image_instance, globalCompress_to_jpeg,
            rect_instance, 100, bos_instance);

    //out.toByteArray();

    jbyteArray bgrArray = (jbyteArray) env->CallObjectMethod(bos_instance, globalTo_byte_array);
    jsize bgrLen = env->GetArrayLength(bgrArray);
    std::vector<char> bgrBuffer(bgrLen);
    env->GetByteArrayRegion(bgrArray, 0, bgrLen, reinterpret_cast<jbyte*>(&bgrBuffer[0]));
    Mat result = imdecode(bgrBuffer, IMREAD_COLOR);

    debug_str << "Decoded jpeg: W: " << result.cols << " H: " << result.rows << endl;

    /*env->DeleteLocalRef(intJavaArray);
    env->DeleteLocalRef(nv21_java_array);
    env->DeleteLocalRef(yuv_image_instance);
    env->DeleteLocalRef(rect_instance);
    env->DeleteLocalRef(bos_instance);*/

    //cv::cvtColor(nv21, bgr, COLOR_YUV2BGR_NV21); // Doesn't work

    //writeImageToDisk("/storage/emulated/0/CaffeAndroid/direct_yuv_out.jpg", result);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "androidYUVToOpenCVMat took " << duration << "ms";
    __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s", debug_str.str().c_str());

    return result;
}

Mat androidYUVToOpenCVMatEx(JNIEnv *env,
                            const std::vector<char> &nv21,
                          int width, int height,
                          const Rect& cropRect) {
    std::stringstream debug_str;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Conversion
    /*std::vector<char> nv21 = YUV_420_888toNV21Ex(
            y_buffer, ylen,
            u_buffer, ulen,
            v_buffer, vlen,
            width, height);*/

    //cv::Mat bgr;
    initializeJavaRefs(env);

    jobject rect_instance = env->NewObject(globalRectClass, globalRectConstructor, cropRect.x, cropRect.y, cropRect.x+cropRect.width, cropRect.y+cropRect.height);
    jobject bos_instance = env->NewObject(globalBosClass, globalBosConstructor);

    jbyteArray nv21_java_array = env->NewByteArray(nv21.size());
    env->SetByteArrayRegion(nv21_java_array, 0, nv21.size(), (jbyte*) &nv21[0]);

    jintArray intJavaArray = env->NewIntArray(2);
    int intCArray[2];
    intCArray[0] = width;
    intCArray[1] = width;
    env->SetIntArrayRegion(intJavaArray, 0, 2, intCArray);

    jobject yuv_image_instance = env->NewObject(globalYuvImageClass, globalYuvConstructor,
                                                nv21_java_array, 17, width, height, NULL);

    //yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);

    env->CallBooleanMethod(yuv_image_instance, globalCompress_to_jpeg, rect_instance, 100, bos_instance);

    //out.toByteArray();

    jbyteArray bgrArray = (jbyteArray) env->CallObjectMethod(bos_instance, globalTo_byte_array);
    jsize bgrLen = env->GetArrayLength(bgrArray);
    std::vector<char> bgrBuffer(bgrLen);
    env->GetByteArrayRegion(bgrArray, 0, bgrLen, reinterpret_cast<jbyte*>(&bgrBuffer[0]));
    Mat result = imdecode(bgrBuffer, IMREAD_COLOR);

    debug_str << "Decoded jpeg: W: " << result.cols << " H: " << result.rows << endl;

    /*env->DeleteLocalRef(intJavaArray);
    env->DeleteLocalRef(nv21_java_array);
    env->DeleteLocalRef(yuv_image_instance);
    env->DeleteLocalRef(rect_instance);
    env->DeleteLocalRef(bos_instance);*/

    //cv::cvtColor(nv21, bgr, COLOR_YUV2BGR_NV21); // Doesn't work

    //writeImageToDisk("/storage/emulated/0/CaffeAndroid/direct_yuv_out.jpg", result);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "androidYUVToOpenCVMatEx took " << duration << "ms";
    __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s", debug_str.str().c_str());

    return result;
}

Mat rotateImage(const Mat& image, int rotation) {
    std::stringstream debug_str;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((image.cols-1)/2.0, (image.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, rotation, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), image.size(), rotation).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - image.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - image.rows/2.0;

    cv::Mat dst;
    cv::warpAffine(image, dst, rot, bbox.size());

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "rotateImage took " << duration << "ms";
    __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s", debug_str.str().c_str());

    return dst;
}

Mat adaptImageSizeForNetInput(
        const Mat &image,
        Net<float> *model,
        int input_blob_index,
        const Rect& crop_rect) {
    Mat image_resized;
    Blob<float>* input_layer;
    Size input_geometry;

    std::stringstream debug_str;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    CHECK(input_blob_index < model->num_inputs()) << "Not enough input layers";

    input_layer = model->input_blobs()[input_blob_index];
    input_geometry = Size(input_layer->width(), input_layer->height());

    if(crop_rect.x == 0 && crop_rect.y == 0 &&
        crop_rect.width == image.cols && crop_rect.height == image.rows) {
        cv::resize(image, image_resized, input_geometry);
    } else {
        cv::resize(image(crop_rect), image_resized, input_geometry);
    }

    //writeImageToDisk("/storage/emulated/0/CaffeAndroid/opencv_image_resize_output.jpg", image_resized);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "adaptImageSizeForNetInput took " << duration << "ms";
    __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s", debug_str.str().c_str());

    return image_resized;
}

Mat adaptImageSizeForBlob(
        const Mat &image,
        Net<float> *model,
        Blob<float>* input_layer,
        const Rect& crop_rect) {
    Mat image_cropped, image_resized;
    Size input_geometry;

    std::stringstream debug_str;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    input_geometry = Size(input_layer->width(), input_layer->height());

    if(crop_rect.x == 0 && crop_rect.y == 0 &&
       crop_rect.width == image.cols && crop_rect.height == image.rows) {
        cv::resize(image, image_resized, input_geometry);
    } else {
        cv::resize(image(crop_rect), image_resized, input_geometry);
    }

    //writeImageToDisk("/storage/emulated/0/CaffeAndroid/opencv_image_resize_output.jpg", image_resized);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "adaptImageSizeForNetInput took " << duration << "ms";
    __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "%s", debug_str.str().c_str());

    return image_resized;
}

extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrReloadApp(
        JNIEnv *env,
        jobject,
        jstring savePath,
        jint trainThreads,
        jint featureExtractionThreads
) {
    std::string save_path = java_string_to_std_string(env, savePath);

    __android_log_print(ANDROID_LOG_VERBOSE, "NativePrint", "%s", "cwrReloadApp");

    //free(my_mem);
    cwr_app = CwrApp::read_from_disk(save_path);
    cwr_app.set_training_threads(trainThreads);
    cwr_app.set_feature_extraction_threads(featureExtractionThreads);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrPreviousStateExists(
        JNIEnv *env,
        jobject,
        jstring savePath
) {
    std::string save_path = java_string_to_std_string(env, savePath);
    return (jboolean) CwrApp::state_exists(save_path);
}

extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrSaveAppState(
        JNIEnv *env,
        jobject,
        jstring savePath) {
    std::string save_path = java_string_to_std_string(env, savePath);
    cwr_app.save_everything_to_disk(save_path);
}

/**
 * App initialization procedure. This is called the first time the app is started.
 *
 * If a previous state already exist (the app was already used in the past), then
 * Java_it_mibiolab_clapp_MyApplication_cwrReloadApp will be called instead.
 *
 * This will initialize the cwr_app global variable by setting the initial model,
 * the replay (rehearsal) memory, the per-channel mean, the initial list of categories,
 * the amount of epochs to use, ...
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrInitApp(
        JNIEnv *env,
        jobject,
        jstring solverPath,
        jstring initialWeightsPath,
        jfloatArray initialClassUpdates,
        jint trainEpochs,
        jint initialClassesNumber,
        jobjectArray cwrLayers,
        jstring predictionLayer,
        jstring preExtractLayer,
        jobjectArray labels,
        jfloat BMean,
        jfloat GMean,
        jfloat RMean,
        jint trainThreads,
        jint featureExtractionThreads,
        jstring rehearsalLayer,
        jstring initialRehearsalBlobPath) {
    std::stringstream debug_str;
    std::vector<string> cwr_layers = java_string_array_to_std_vector(env, cwrLayers);
    std::vector<string> labels_list = java_string_array_to_std_vector(env, labels);
    std::string solver_path = java_string_to_std_string(env, solverPath);
    std::string initial_weights_path = java_string_to_std_string(env, initialWeightsPath);
    std::string prediction_layer = java_string_to_std_string(env, predictionLayer);
    std::string extraction_layer;
    if(preExtractLayer != NULL) {
        extraction_layer = java_string_to_std_string(env, preExtractLayer);
    }

    std::string rehearsal_layer;
    if(rehearsalLayer != NULL) {
        rehearsal_layer = java_string_to_std_string(env, rehearsalLayer);
    }

    std::string rehearsal_blob_path;
    if(initialRehearsalBlobPath != NULL) {
        rehearsal_blob_path = java_string_to_std_string(env, initialRehearsalBlobPath);
    }

    std::vector<float> class_updates;
    {
        jfloat *java_class_updates = env->GetFloatArrayElements(initialClassUpdates, NULL);
        jsize class_updates_length = env->GetArrayLength(initialClassUpdates);

        class_updates = std::vector<float>(java_class_updates, java_class_updates+class_updates_length);
        env->ReleaseFloatArrayElements(initialClassUpdates, java_class_updates, 0);
    }

    //debug_str << "OpenBLAS threads: " << NUM_THREADS << endl;

    //free(my_mem);
    cwr_app = CwrApp(solver_path, initial_weights_path, /*initial_weights_path,*/ class_updates,
                     trainEpochs, initialClassesNumber, cwr_layers, prediction_layer,
                     labels_list, CwrApp::create_mean_image(BMean, GMean, RMean), extraction_layer,
                     cwr_layers[0]);
    cwr_app.set_training_threads(trainThreads);
    cwr_app.set_feature_extraction_threads(featureExtractionThreads);
    if(!rehearsal_layer.empty() && !rehearsal_blob_path.empty()) {
        auto in_file = std::fstream(rehearsal_blob_path, std::ios::in | std::ios::binary);
        std::shared_ptr<RehearsalMemory<cwra_rehe_t>> initial_rehe = load_rehe_from_snapshot(in_file);
        in_file.close();

        vector<cwra_rehe_t> rehe_x = initial_rehe->getSamplesX();
        vector<int> rehe_y = initial_rehe->getSamplesY();
        for(int i = 0; i < rehe_y.size(); i++) {
            rehe_y[i] /= 5;
        }

        initial_rehe->load_memory(rehe_x, rehe_y);

        cwr_app.set_rehearsal_layer(rehearsal_layer);
        cwr_app.set_rehearsal_memory(initial_rehe);
    }

    debug_str << indexes_range(0, 100-1, 10) << endl;

    //PrintNetworkArchitecture(cwr_app.get_net(), debug_str);
    log_android_debug(debug_str);
    // Inference only version (debug)
    /*cwr_app = CwrApp(solver_path, initial_weights_path, initialClassesNumber,
                    prediction_layer,labels_list, CwrApp::create_mean_image(BMean, GMean, RMean));*/
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrGetLabels(
        JNIEnv *env,
        jobject) {
    const vector<string> &labels = cwr_app.get_labels();
    jobjectArray ret = (jobjectArray)env->NewObjectArray(labels.size(),
            env->FindClass("java/lang/String"), nullptr);

    for(int i = 0; i < labels.size(); i++) {
        env->SetObjectArrayElement(ret, i, env->NewStringUTF(labels[i].c_str()));
    }

    return ret;
}

/**
 * Function used to add an image to the current training batch.
 *
 * This will also extract and internally store (in memory) the latent activations.
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrAddTrainingImageFromYUV(
        JNIEnv *env,
        jobject,
        jbyteArray Y,
        jbyteArray U,
        jbyteArray V,
        jint w,
        jint h,
        jint rotation,
        jint cropX, jint cropY, jint cropW, jint cropH) {
    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2;

    /*
     * In this first part of the function the image is converted to an OpenCV Mat.
     *
     * Data has to be extracted from the Java byte array, then converted, ...
     */
    jsize ylen = env->GetArrayLength(Y);
    //std::vector<char> y_buffer(ylen);
    //env->GetByteArrayRegion(Y, 0, ylen, reinterpret_cast<jbyte*>(&y_buffer[0]));
    jboolean isYCopy;
    char* y_buffer = (char*) env->GetPrimitiveArrayCritical(Y, &isYCopy);

    jsize ulen = env->GetArrayLength(U);
    //std::vector<char> u_buffer(ulen);
    //env->GetByteArrayRegion(U, 0, ulen, reinterpret_cast<jbyte*>(&u_buffer[0]));
    jboolean isUCopy;
    char* u_buffer = (char*) env->GetPrimitiveArrayCritical(U, &isUCopy);

    jsize vlen = env->GetArrayLength(V);
    //std::vector<char> v_buffer(vlen);
    //env->GetByteArrayRegion(V, 0, vlen, reinterpret_cast<jbyte*>(&v_buffer[0]));
    jboolean isVCopy;
    char* v_buffer = (char*) env->GetPrimitiveArrayCritical(V, &isVCopy);

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Getting yuv buffers took " << duration << "ms" << endl;
    t1 = t2;

    std::vector<char> nv21 = YUV_420_888toNV21Ex(
            y_buffer, ylen,
            u_buffer, ulen,
            v_buffer, vlen,
            w, h);

    env->ReleasePrimitiveArrayCritical(Y, y_buffer, isYCopy ? 0 : JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(U, u_buffer, isUCopy ? 0 : JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(V, v_buffer, isVCopy ? 0 : JNI_ABORT);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Copy and release buffers took " << duration << "ms" << endl;
    t1 = t2;

    Mat image = androidYUVToOpenCVMatEx(env, nv21, w, h, Rect(cropX, cropY, cropW, cropH));

    // Resize + rotate
    image = adaptImageSizeForBlob(image, cwr_app.get_net().get(),
                                  cwr_app.get_net()->blob_by_name("data").get(), Rect(0, 0, cropW, cropH));
    image = rotateImage(image, rotation);

    // To float32
    Mat img2;
    image.convertTo(img2, CV_32FC3);

    // Subtract mean
    cv::Mat sample_normalized = cwr_app.subtract_mean_image(img2);

    // Add the image. This will also extract and keep in memory its latent activations.
    cwr_app.add_batch_image(sample_normalized);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Data preprocessing took " << duration << "ms" << endl;

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "addTrainingImageFromYUV took " << duration << "ms";
    __android_log_print(ANDROID_LOG_VERBOSE, "NativePrint", "%s", debug_str.str().c_str());
}

/**
 * Function used to add an image to the current training batch.
 *
 * This will also extract and internally store (in memory) the latent activations.
 *
 * This is the JPEG version, which is not used by the app and is left here for
 * debugging purposes. Refer to Java_it_mibiolab_clapp_MyApplication_cwrAddTrainingImageFromYUV
 * for the actually used one.
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrAddTrainingImageFromJpeg(
        JNIEnv *env,
        jobject,
        jbyteArray jpegBytes,
        jint rotation,
        jint cropX, jint cropY, jint cropW, jint cropH) {
    // http://planet.jboss.org/post/jni_performance_the_saga_continues
    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = function_start;
    high_resolution_clock::time_point t2;
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    /*
     * In this first part of the function the image is converted to an OpenCV Mat.
     *
     * Data has to be extracted from the Java byte array, then converted, ...
     */
    jsize jpglen = env->GetArrayLength(jpegBytes);
    jboolean isJpegBufferCopy;
    void* jpeg_buffer = env->GetPrimitiveArrayCritical(jpegBytes, &isJpegBufferCopy);

    cv::Mat image = imdecode(cv::Mat(1, jpglen, CV_8UC1, jpeg_buffer), IMREAD_COLOR);
    env->ReleasePrimitiveArrayCritical(jpegBytes, jpeg_buffer, isJpegBufferCopy ? 0 : JNI_ABORT);
    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Image decode took " << duration << "ms" << endl;
    t1 = t2;

    // Resize + rotate
    image = adaptImageSizeForNetInput(image, cwr_app.get_net().get(), 0, cv::Rect(0, 0, cropW, cropH));
    image = rotateImage(image, rotation);

    // To float32
    cv::Mat img2;
    image.convertTo(img2, CV_32FC3);

    // Subtract mean
    cv::Mat sample_normalized = cwr_app.subtract_mean_image(img2);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Data preprocessing took " << duration << "ms" << endl;
    t1 = t2;

    // Add the image. This will also extract and keep in memory its latent activations.
    cwr_app.add_batch_image(sample_normalized);

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "addTrainingImageFromJpeg took " << duration << "ms";
    log_android_debug(debug_str);
}

extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrClearTrainingBatch(
        JNIEnv *env,
        jobject) {
    cwr_app.reset_batch();
}

/**
 * Runs the incremental training step.
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrTrainingStep(
        JNIEnv *env,
        jobject appObj,
        jint label) {
    initializeJavaRefs(env);
    MyAppTrainingStatusReporter reporter(env, appObj, global_set_training_progress);
    cwr_app.cwr_execute_step(label, nullptr, &reporter);
}

/**
 * Function used to run the inference pass on a single image.
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrInferenceFromJpegBytes(
        JNIEnv *env,
        jobject,
        jbyteArray jpegBytes,
        jint rotation,
        jint cropX, jint cropY, jint cropW, jint cropH,
        jfloatArray predictions) {
    /*
     * In this first part the image is converted to an OpenCV Mat.
     *
     * Data is in JPEG format, so that's easy.
     */
    // http://planet.jboss.org/post/jni_performance_the_saga_continues
    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = function_start;
    high_resolution_clock::time_point t2;
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    jfloat *c_array = env->GetFloatArrayElements(predictions, NULL);

    jsize jpglen = env->GetArrayLength(jpegBytes);
    jboolean isJpegBufferCopy;
    void* jpeg_buffer = env->GetPrimitiveArrayCritical(jpegBytes, &isJpegBufferCopy);

    // JPEG -> OpenCV Mat conversion
    cv::Mat image = imdecode(cv::Mat(1, jpglen, CV_8UC1, jpeg_buffer), IMREAD_COLOR);
    env->ReleasePrimitiveArrayCritical(jpegBytes, jpeg_buffer, isJpegBufferCopy ? 0 : JNI_ABORT);

    // Resize and rotate
    image = adaptImageSizeForNetInput(image, cwr_app.get_net().get(), 0, cv::Rect(0, 0, cropW, cropH));
    image = rotateImage(image, rotation);

    // Convert to float32
    cv::Mat img2;
    image.convertTo(img2, CV_32FC3);

    // Subtract mean
    cv::Mat sample_normalized = cwr_app.subtract_mean_image(img2);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "data preprocessing took " << duration << "ms" << endl;
    t1 = t2;

    // Inference
    std::vector<float> prediction_layer_output = cwr_app.inference(sample_normalized);

    int min_probs_size = env->GetArrayLength(predictions);
    if(prediction_layer_output.size() < min_probs_size) {
        min_probs_size = prediction_layer_output.size();
    }

    for (int i = 0; i < min_probs_size; i++)
    {
        c_array[i] = prediction_layer_output[i];
        debug_str << i << " " << c_array[i] << std::endl;
    }
    env->ReleaseFloatArrayElements(predictions, c_array, 0);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "inference took " << duration << "ms" << endl;
    t1 = t2;

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "cwrInferenceFromJpeg took " << duration << "ms";

    log_android_debug(debug_str);
}

/**
 * This is the function used to run the inference pass on a single image.
 *
 * This function is not used in the app and is left here for debugging purposes.
 * Refer to Java_it_mibiolab_clapp_MyApplication_cwrInferenceFromJpegBytes for the actually used one.
 */
extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrInferenceFromYUV(
        JNIEnv *env,
        jobject,
        jbyteArray Y,
        jbyteArray U,
        jbyteArray V,
        jint w,
        jint h,
        jint rotation,
        jint cropX, jint cropY, jint cropW, jint cropH,
        jfloatArray predictions) {
    /*
     * In this first part the image is converted to an OpenCV Mat.
     *
     * Data has to be extracted from the Java byte array, then converted, ...
     */
    // http://planet.jboss.org/post/jni_performance_the_saga_continues
    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2;

    jfloat *c_array = env->GetFloatArrayElements(predictions, NULL);

    jsize ylen = env->GetArrayLength(Y);
    //std::vector<char> y_buffer(ylen);
    //env->GetByteArrayRegion(Y, 0, ylen, reinterpret_cast<jbyte*>(&y_buffer[0]));
    jboolean isYCopy;
    char* y_buffer = (char*) env->GetPrimitiveArrayCritical(Y, &isYCopy);

    jsize ulen = env->GetArrayLength(U);
    //std::vector<char> u_buffer(ulen);
    //env->GetByteArrayRegion(U, 0, ulen, reinterpret_cast<jbyte*>(&u_buffer[0]));
    jboolean isUCopy;
    char* u_buffer = (char*) env->GetPrimitiveArrayCritical(U, &isUCopy);

    jsize vlen = env->GetArrayLength(V);
    //std::vector<char> v_buffer(vlen);
    //env->GetByteArrayRegion(V, 0, vlen, reinterpret_cast<jbyte*>(&v_buffer[0]));
    jboolean isVCopy;
    char* v_buffer = (char*) env->GetPrimitiveArrayCritical(V, &isVCopy);

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Getting yuv buffers took " << duration << "ms" << endl;
    t1 = t2;

    std::vector<char> nv21 = YUV_420_888toNV21Ex(
            y_buffer, ylen,
            u_buffer, ulen,
            v_buffer, vlen,
            w, h);

    env->ReleasePrimitiveArrayCritical(Y, y_buffer, isYCopy ? 0 : JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(U, u_buffer, isUCopy ? 0 : JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(V, v_buffer, isVCopy ? 0 : JNI_ABORT);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "Copy and release buffers took " << duration << "ms" << endl;
    t1 = t2;

    // Conversion to OpenCV Mat
    cv::Mat image = androidYUVToOpenCVMatEx(env, /*y_buffer, ylen, u_buffer, ulen, v_buffer, vlen,*/
                                            nv21,
            w, h, cv::Rect(cropX, cropY, cropW, cropH));

    // Resize image
    image = adaptImageSizeForNetInput(image, cwr_app.get_net().get(), 0, cv::Rect(0, 0, cropW, cropH));

    // Rotate the image
    image = rotateImage(image, rotation);

    // Convert to float32
    cv::Mat img2;
    image.convertTo(img2, CV_32FC3);

    // Subtract mean
    cv::Mat sample_normalized = cwr_app.subtract_mean_image(img2);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "data preprocessing took " << duration << "ms" << endl;
    t1 = t2;

    // Inference
    std::vector<float> prediction_layer_output = cwr_app.inference(sample_normalized);

    // Obtain the predictions
    int min_probs_size = env->GetArrayLength(predictions);
    if(prediction_layer_output.size() < min_probs_size) {
        min_probs_size = prediction_layer_output.size();
    }

    for (int i = 0; i < min_probs_size; i++)
    {
        c_array[i] = prediction_layer_output[i];
        debug_str << i << " " << c_array[i] << std::endl;
    }
    env->ReleaseFloatArrayElements(predictions, c_array, 0);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "inference took " << duration << "ms" << endl;
    t1 = t2;

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "cwrInferenceFromYUV took " << duration << "ms";

    __android_log_print(ANDROID_LOG_VERBOSE, "NativePrint", "%s", debug_str.str().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrInferenceFromFile(
        JNIEnv *env,
        jobject /* this */,
        jstring pattern,
        jint cropX, jint cropY, jint cropW, jint cropH,
        jfloatArray predictions) {

    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2;

    jfloat *c_array = env->GetFloatArrayElements(predictions, NULL);

    Mat image;
    image = imread(java_string_to_std_string(env, pattern), cv::IMREAD_COLOR);

    image = adaptImageSizeForNetInput(image, cwr_app.get_net().get(), 0, cv::Rect(0, 0, cropW, cropH));

    image = rotateImage(image, 0);

    cv::Mat img2;
    image.convertTo(img2, CV_32FC3);

    cv::Mat sample_normalized = cwr_app.subtract_mean_image(img2);

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "data preprocessing took " << duration << "ms" << endl;
    t1 = t2;

    std::vector<float> prediction_layer_output = cwr_app.inference(sample_normalized);
    /*std::vector<float> prediction_layer_output(10, 0.0f);
    executeInference(sample_normalized, cwr_app.get_net().get(), &prediction_layer_output[0], 10);*/

    int min_probs_size = env->GetArrayLength(predictions);
    if(prediction_layer_output.size() < min_probs_size) {
        min_probs_size = prediction_layer_output.size();
    }

    debug_str << "min_probs_size " << min_probs_size << endl;
    for (int i = 0; i < min_probs_size; i++)
    {
        c_array[i] = prediction_layer_output[i];
        debug_str << i << " " << c_array[i] << std::endl;
    }
    env->ReleaseFloatArrayElements(predictions, c_array, 0);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "inference took " << duration << "ms" << endl;
    t1 = t2;

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "cwrInferenceFromFile took " << duration << "ms";

    __android_log_print(ANDROID_LOG_VERBOSE, "NativePrint", "%s", debug_str.str().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrAddNewCategory(
        JNIEnv *env,
        jobject,
        jstring categoryName) {
    string category = java_string_to_std_string(env, categoryName);
    return cwr_app.add_category(category);
}

extern "C" JNIEXPORT jint JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrGetMaxCategories(
        JNIEnv *env,
        jobject) {
    return cwr_app.get_max_categories();
}

extern "C" JNIEXPORT void JNICALL
Java_it_mibiolab_clapp_MyApplication_cwrSaveThumbnailFromYUV(
        JNIEnv *env,
        jobject,
        jbyteArray Y,
        jbyteArray U,
        jbyteArray V,
        jint w,
        jint h,
        jint rotation,
        jint cropX, jint cropY, jint cropW, jint cropH,
        jint targetW, jint targetH,
        jstring savePath) {
    std::stringstream debug_str;
    high_resolution_clock::time_point function_start = high_resolution_clock::now();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2;

    jsize ylen = env->GetArrayLength(Y);
    std::vector<char> y_buffer(ylen);
    env->GetByteArrayRegion (Y, 0, ylen, reinterpret_cast<jbyte*>(&y_buffer[0]));

    jsize ulen = env->GetArrayLength(U);
    std::vector<char> u_buffer(ulen);
    env->GetByteArrayRegion (U, 0, ulen, reinterpret_cast<jbyte*>(&u_buffer[0]));

    jsize vlen = env->GetArrayLength(V);
    std::vector<char> v_buffer(vlen);
    env->GetByteArrayRegion (V, 0, vlen, reinterpret_cast<jbyte*>(&v_buffer[0]));

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "getting yuv buffers took " << duration << "ms" << endl;
    t1 = t2;

    cv::Mat image = androidYUVToOpenCVMat(env, y_buffer, u_buffer, v_buffer, w, h, cv::Rect(cropX, cropY, cropW, cropH));
    image = rotateImage(image, rotation);

    cv::Mat image_resized;
    Size target_size(targetW, targetH);
    cv::Rect crop_rect(0, 0, cropW, cropH);

    if(crop_rect.x == 0 && crop_rect.y == 0 &&
       crop_rect.width == image.cols && crop_rect.height == image.rows) {
        // Already cropped by androidYUVToOpenCVMat -> should never get inside this branch
        cv::resize(image, image_resized, target_size);
    } else {
        cv::resize(image(crop_rect), image_resized, target_size);
    }

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "data preprocessing took " << duration << "ms" << endl;
    t1 = t2;

    string save_path = java_string_to_std_string(env, savePath);
    if(!writeImageToDisk(save_path, image_resized)) {
        debug_str << "--- Save error! ---" << endl;
        jclass exClass = env->FindClass( "java/lang/RuntimeException");
        if (exClass == nullptr) {
            // That's not funny...
            debug_str << "--- Can't throw RuntimeException (no class) ---" << endl;
        } else {
            if(env->ThrowNew(exClass, "Can't write thumbnail to disk!") < 0) {
                // That's not funny x2
                debug_str << "--- Can't throw RuntimeException ---" << endl;
            }
        }
    }

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( t2 - t1 ).count();
    debug_str << "save took " << duration << "ms" << endl;
    t1 = t2;

    high_resolution_clock::time_point function_end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>( function_end - function_start ).count();
    debug_str << "cwrSaveThumbnailFromYUV took " << duration << "ms";

    __android_log_print(ANDROID_LOG_VERBOSE, "NativePrint", "%s", debug_str.str().c_str());
}

std::vector<std::string> java_string_array_to_std_vector(JNIEnv *env, jobjectArray stringArray) {
    jsize stringsCount = env->GetArrayLength(stringArray);
   // __android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "Strings count: %d", stringsCount);

    std::vector<std::string> stringsList;

    for (int i = 0; i < stringsCount; i++) {
        auto javaStr = (jstring) env->GetObjectArrayElement(stringArray, i);
        const char *rawString = env->GetStringUTFChars(javaStr, nullptr);
        std::string asStrString(rawString);
        stringsList.push_back(asStrString);
        env->ReleaseStringUTFChars(javaStr, rawString);
        //__android_log_print(ANDROID_LOG_DEBUG, "NativePrint", "Saved strings: %lu", stringsList.size());
    }

    return stringsList;
}

std::string java_string_to_std_string(JNIEnv *env, jstring javaString) {
    const char *byteString = env->GetStringUTFChars(javaString, nullptr);
    std::string result(byteString);
    env->ReleaseStringUTFChars(javaString, byteString);
    return result;
}

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return -1;
    }

    /*my_mem = malloc(1024 * 1024 * 1024);
    if(my_mem == nullptr) {
        __android_log_print(ANDROID_LOG_DEBUG, "NativePrintInit", "No memory!");
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "NativePrintInit", "A lot of memory!");
    }*/

    __android_log_print(ANDROID_LOG_DEBUG, "NativePrintInit", "On load");

    return JNI_VERSION_1_6;
}

void initializeJavaRefs(JNIEnv *env) {
    //https://stackoverflow.com/questions/16820209/use-javap-to-get-method-signatures-in-for-android-activity
    if(!areJavaRefsInitialized) {
        jclass jcls = env->FindClass("android/graphics/YuvImage");
        jclass jRectClass = env->FindClass("android/graphics/Rect");
        jclass bosClass = env->FindClass("java/io/ByteArrayOutputStream");
        jclass myAppClass = env->FindClass("it/mibiolab/clapp/MyApplication");

        jmethodID constructor = env->GetMethodID(jcls, "<init>", "([BIII[I)V");
        jmethodID rectConstructor = env->GetMethodID(jRectClass, "<init>", "(IIII)V");
        jmethodID bosConstructor = env->GetMethodID(bosClass, "<init>", "()V");
        jmethodID compress_to_jpeg = env->GetMethodID(jcls, "compressToJpeg", "(Landroid/graphics/Rect;ILjava/io/OutputStream;)Z");
        jmethodID to_byte_array = env->GetMethodID(bosClass, "toByteArray", "()[B");
        jmethodID set_training_progress = env->GetMethodID(myAppClass, "setTrainingProgress","(F)V");

        globalYuvImageClass = reinterpret_cast<jclass>(env->NewGlobalRef(jcls));
        globalRectClass = reinterpret_cast<jclass>(env->NewGlobalRef(jRectClass));
        globalBosClass = reinterpret_cast<jclass>(env->NewGlobalRef(bosClass));
        globalMyApplicationClass = reinterpret_cast<jclass>(env->NewGlobalRef(myAppClass));
        globalYuvConstructor = constructor;
        globalRectConstructor = rectConstructor;
        globalBosConstructor = bosConstructor;
        globalCompress_to_jpeg = compress_to_jpeg;
        globalTo_byte_array = to_byte_array;
        global_set_training_progress = set_training_progress;
        areJavaRefsInitialized = true;
    }
}
