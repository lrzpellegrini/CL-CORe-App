package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;
import android.util.Log;

import java.lang.ref.WeakReference;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.MyApplication;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

public class PredictionTaskFromYUV extends AsyncTask<AndroidCameraImageModel, Void, float[]> {

    private static final String TAG = "PredictionTaskYUV";

    private final WeakReference<ClassifyCamera> activityReference;
    private final MyApplication application;
    private final long predictStartTime;

    public PredictionTaskFromYUV(ClassifyCamera activity,
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
        /*double averageProcessingTime = activity.processingTime.getMean();
        double averageFPS = activity.fps.getFps();

        String message = String.format(Locale.US,
                "%s\nProcessing time = %.2f ms\nFPS = %.2f",
                formatPredictions(predicted), averageProcessingTime, averageFPS);
        activity.debug_tv.setText(message);*/
        activity.onPredictionTaskCompleted(predicted, endTime);
    }

    @Override
    protected float[] doInBackground(AndroidCameraImageModel... predictionTaskData) {
        float[] predictions = new float[application.getCategoriesCount()];
        for (AndroidCameraImageModel predictionTaskDatum : predictionTaskData) {
            if (isCancelled()) {
                break;
            }
            // long startTime = System.currentTimeMillis();
            application.cwrInferenceFromYUV(
                    predictionTaskDatum.getYPlane(),
                    predictionTaskDatum.getUPlane(),
                    predictionTaskDatum.getVPlane(),
                    predictionTaskDatum.getImgW(),
                    predictionTaskDatum.getImgH(),
                    predictionTaskDatum.getRotation(),
                    predictionTaskDatum.getCropX(),
                    predictionTaskDatum.getCropY(),
                    predictionTaskDatum.getCropW(),
                    predictionTaskDatum.getCropH(),
                    predictions);

            /*Log.v(TAG, "Ended inference. It took " +
                    (System.currentTimeMillis() - startTime) + " ms");
            Log.v(TAG, "Inference results:\n" + Arrays.toString(predictions));*/
        }

        return predictions;
    }

    public static PredictionTaskFactory getFactory() {
        return new PredictionTaskFactory() {
            @Override
            public AsyncTask<AndroidCameraImageModel, Void, float[]> createNewTask(
                    ClassifyCamera activity, long startTime) {
                return new PredictionTaskFromYUV(activity, startTime);
            }
        };
    }
}
