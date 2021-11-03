package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;
import android.util.Log;

import java.lang.ref.WeakReference;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.Core50Constants;
import it.mibiolab.clapp.MyApplication;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

public class SaveThumbnailTaskFromYUV extends AsyncTask<AndroidCameraImageModel, Void, Void> {

    private static final String TAG = "SaveThumbTaskYUV";

    private final WeakReference<ClassifyCamera> activityReference;
    private final MyApplication application;
    private final long saveStartTime;

    public SaveThumbnailTaskFromYUV(ClassifyCamera activity,
                                    long saveStartTime) {
        this.activityReference = new WeakReference<>(activity);
        this.application = activity.getMyApplication();
        this.saveStartTime = saveStartTime;
    }

    @Override
    protected void onPreExecute() {
        Log.d(TAG, "Starting Save Thumbnail Task");
    }

    @Override
    protected void onPostExecute(Void ignored) {
        long endTime = System.currentTimeMillis();
        long difference = endTime - saveStartTime;
        Log.d(TAG, "Ended thumbnail image save in " + difference + "ms");

        ClassifyCamera activity = activityReference.get();
        if (activity == null || activity.isFinishing()) {
            return;
        }

        if (!activity.isFinishing()) {
            activity.onThumbnailSaved();
        }
    }

    @Override
    protected Void doInBackground(AndroidCameraImageModel... predictionTaskData) {
        ClassifyCamera activity = activityReference.get();
        if (activity == null || activity.isFinishing()) {
            return null;
        }

        if (activity.getTrainingPhase() != ClassifyCamera.TrainingPhase.GATHERING_IMAGES) {
            return null;
        }

        for (AndroidCameraImageModel predictionTaskDatum : predictionTaskData) {
            if (isCancelled()) {
                break;
            }

            long startTime = System.currentTimeMillis();

            application.cwrSaveThumbnailFromYUV(
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
                    Core50Constants.CORE50_IMAGES_SIZE[0],
                    Core50Constants.CORE50_IMAGES_SIZE[1],
                    application.getTmpThumbnailPath());

            Log.d(TAG, "Ended thumbnail save. It took " +
                    (System.currentTimeMillis() - startTime) + " ms");
        }


        return null;
    }
}
