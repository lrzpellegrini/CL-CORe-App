package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;
import android.util.Log;

import java.io.File;
import java.lang.ref.WeakReference;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.MyApplication;
import it.mibiolab.clapp.model.AndroidCameraImageModel;
import it.mibiolab.clapp.utils.FPSConstrainedDeque;

import static it.mibiolab.clapp.utils.YUVImageUtils.getJpegBytesFromYUV;

public class TrainingSaveImageTaskFromJpeg extends AsyncTask<AndroidCameraImageModel, Void, Void> {

    private static final String TAG = "TrainSaveImageTaskJpeg";

    private final WeakReference<ClassifyCamera> activityReference;
    private final MyApplication application;
    private final long saveStartTime;
    private final boolean onlyImageGather;
    private final int imageNumber;
    private final FPSConstrainedDeque queue;

    public TrainingSaveImageTaskFromJpeg(ClassifyCamera activity,
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

        AndroidCameraImageModel data = predictionTaskData[0];

        //long startTime = System.currentTimeMillis();

        if (onlyImageGather) {
            String imageFileName = imageNumber + ".jpg";

            File pathSDCard = new File(application.getTrainingImagesPath());
            pathSDCard.mkdirs();
            pathSDCard = new File(pathSDCard, imageFileName);

            application.cwrSaveThumbnailFromYUV(data.getYPlane(),
                    data.getUPlane(),
                    data.getVPlane(),
                    data.getImgW(),
                    data.getImgH(),
                    data.getRotation(),
                    data.getCropX(),
                    data.getCropY(),
                    data.getCropW(),
                    data.getCropH(),
                    data.getCropW(),
                    data.getCropH(),
                    pathSDCard.getAbsolutePath());
        } else {
            application.cwrAddTrainingImageFromJpeg(
                    getJpegBytesFromYUV(data),
                    data.getRotation(),
                    data.getCropX(),
                    data.getCropY(),
                    data.getCropW(),
                    data.getCropH());
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
                return new TrainingSaveImageTaskFromJpeg(activity, saveStartTime, onlyImageGather,
                        imageNumber, queue);
            }
        };
    }
}
