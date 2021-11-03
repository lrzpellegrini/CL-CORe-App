package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;
import android.util.Log;

import java.lang.ref.WeakReference;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.MyApplication;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

import static it.mibiolab.clapp.utils.YUVImageUtils.getJpegBytesFromYUV;

public class PredictionTaskFromJpeg extends AsyncTask<AndroidCameraImageModel, Void, float[]> {

    private static final String TAG = "PredictionTaskJpeg";

    private final WeakReference<ClassifyCamera> activityReference;
    private final MyApplication application;
    private final long predictStartTime;

    public PredictionTaskFromJpeg(ClassifyCamera activity,
                                  long predictStartTime) {
        this.activityReference = new WeakReference<>(activity);
        this.application = activity.getMyApplication();
        this.predictStartTime = predictStartTime;
    }

    @Override
    protected void onPreExecute() {
        Log.d(TAG, "Starting prediction");
    }

    @Override
    protected void onPostExecute(float[] predicted) {
        long endTime = System.currentTimeMillis();
        long difference = endTime - predictStartTime;
        Log.d(TAG, "Ended prediction in " + difference + "ms");

        ClassifyCamera activity = activityReference.get();
        if (activity == null || activity.isFinishing()) return;

        activity.updateProcessingTimeData(difference);
        activity.onPredictionTaskCompleted(predicted, endTime);
    }

    @Override
    protected float[] doInBackground(AndroidCameraImageModel... predictionTaskData) {
        float[] predictions = new float[application.getCategoriesCount()];
        for (AndroidCameraImageModel data : predictionTaskData) {
            if (isCancelled()) {
                break;
            }

            application.cwrInferenceFromJpegBytes(
                    getJpegBytesFromYUV(data),
                    data.getRotation(),
                    data.getCropX(),
                    data.getCropY(),
                    data.getCropW(),
                    data.getCropH(),
                    predictions);
        }

        return predictions;
    }

    public static PredictionTaskFactory getFactory() {
        return new PredictionTaskFactory() {
            @Override
            public AsyncTask<AndroidCameraImageModel, Void, float[]> createNewTask(
                    ClassifyCamera activity, long startTime) {
                return new PredictionTaskFromJpeg(activity, startTime);
            }
        };
    }
}
