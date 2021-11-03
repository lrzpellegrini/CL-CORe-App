package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

import static android.os.AsyncTask.Status.FINISHED;

/**
 * Consumer used to receive images to be classified.
 *
 * The actual procedure is handled in the {@link PredictionTaskFromJpeg} task, but the
 * {@link PredictionTaskFromYUV} is also available.
 */
public class InferenceConsumer extends AbstractClassifyCameraConsumer {

    private AsyncTask<AndroidCameraImageModel, Void, float[]> runningTask;
    private final PredictionTaskFactory taskFactory;

    public InferenceConsumer(ClassifyCamera cameraActivity, PredictionTaskFactory taskFactory) {
        super(cameraActivity);
        this.runningTask = null;
        this.taskFactory = taskFactory;
    }

    @Override
    public boolean canConsumeImage() {
        return runningTask == null || runningTask.getStatus() == FINISHED;
    }

    @Override
    public void onImageAvailable(AndroidCameraImageModel image) {
        if(!shouldConsumeNextImage()) {
            return;
        }

        ClassifyCamera activity = getActivity();
        runningTask = taskFactory.createNewTask(activity, System.currentTimeMillis());
        activity.executeConsumerTask(ClassifyCamera.TaskExecutors.INFERENCE, runningTask, image);
    }
}
