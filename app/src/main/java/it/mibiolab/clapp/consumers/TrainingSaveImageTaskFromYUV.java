package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;
import android.util.Log;

import java.io.File;
import java.lang.ref.WeakReference;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.utils.FPSConstrainedDeque;
import it.mibiolab.clapp.MyApplication;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

public class TrainingSaveImageTaskFromYUV extends AsyncTask<AndroidCameraImageModel, Void, Void> {

    private static final String TAG = "TrainSaveImageTaskYUV";

    private final WeakReference<ClassifyCamera> activityReference;
    private final MyApplication application;
    private final long saveStartTime;
    private final boolean onlyImageGather;
    private final int imageNumber;
    private final FPSConstrainedDeque queue;

    public TrainingSaveImageTaskFromYUV(ClassifyCamera activity,
                                        long saveStartTime,
                                        boolean onlyImageGather,
                                        int imageNumber,
                                        FPSConstrainedDeque queue) {
        this.activityReference = new WeakReference<>(activity);
        this.application = activity.getMyApplication();
        this.saveStartTime = saveStartTime;
        this.onlyImageGather = onlyImageGather;
        this.imageNumber = imageNumber;
        this.queue = queue;
    }

    @Override
    protected void onPreExecute() {
        Log.d(TAG, "Starting Training Save Image Task");
    }

    @Override
    protected void onPostExecute(Void ignored) {
        long endTime = System.currentTimeMillis();
        long difference = endTime - saveStartTime;
        Log.d(TAG, "Ended training image save in " + difference + "ms");

        ClassifyCamera activity = activityReference.get();
        if (activity == null || activity.isFinishing()) {
            return;
        }

        queue.removeElement();
        activity.onTrainingImageSaveCompleted(endTime);
    }

    @Override
    protected Void doInBackground(AndroidCameraImageModel... predictionTaskData) {
        ClassifyCamera activity = activityReference.get();
        if (activity == null || activity.isFinishing()) return null;

        AndroidCameraImageModel predictionTaskDatum = predictionTaskData[0];

        //long startTime = System.currentTimeMillis();

        if (onlyImageGather) {
            String imageFileName = imageNumber + ".jpg";

            File pathSDCard = new File(application.getTrainingImagesPath());
            pathSDCard.mkdirs();
            pathSDCard = new File(pathSDCard, imageFileName);

            application.cwrSaveThumbnailFromYUV(predictionTaskDatum.getYPlane(),
                    predictionTaskDatum.getUPlane(),
                    predictionTaskDatum.getVPlane(),
                    predictionTaskDatum.getImgW(),
                    predictionTaskDatum.getImgH(),
                    predictionTaskDatum.getRotation(),
                    predictionTaskDatum.getCropX(),
                    predictionTaskDatum.getCropY(),
                    predictionTaskDatum.getCropW(),
                    predictionTaskDatum.getCropH(),
                    predictionTaskDatum.getCropW(),
                    predictionTaskDatum.getCropH(),
                    pathSDCard.getAbsolutePath());
        } else {
            application.cwrAddTrainingImageFromYUV(
                    predictionTaskDatum.getYPlane(),
                    predictionTaskDatum.getUPlane(),
                    predictionTaskDatum.getVPlane(),
                    predictionTaskDatum.getImgW(),
                    predictionTaskDatum.getImgH(),
                    predictionTaskDatum.getRotation(),
                    predictionTaskDatum.getCropX(),
                    predictionTaskDatum.getCropY(),
                    predictionTaskDatum.getCropW(),
                    predictionTaskDatum.getCropH());
        }

        /*Log.d(TAG, "Ended save. It took " +
                (System.currentTimeMillis() - startTime) + " ms");*/

        return null;
    }

    public static TrainingSaveImageTaskFactory getFactory() {
        return new TrainingSaveImageTaskFactory() {
            @Override
            public AsyncTask<AndroidCameraImageModel, Void, Void> createNewTask(
                    ClassifyCamera activity, long saveStartTime, boolean onlyImageGather,
                    int imageNumber, FPSConstrainedDeque queue) {
                return new TrainingSaveImageTaskFromYUV(activity, saveStartTime, onlyImageGather,
                        imageNumber, queue);
            }
        };
    }
}
