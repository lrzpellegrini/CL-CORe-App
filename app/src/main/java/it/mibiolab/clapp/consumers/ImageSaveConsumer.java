package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.model.AndroidCameraImageModel;
import it.mibiolab.clapp.utils.FPSConstrainedDeque;

/**
 * Consumer that is used to receive training images.
 *
 * This will create and run an async task that will manage the image conversion and latent vector
 * extraction process. The actual process is handles by the {@link TrainingSaveImageTaskFromYUV}
 * task, but the {@link TrainingSaveImageTaskFromJpeg} can also be used.
 */
public class ImageSaveConsumer extends AbstractClassifyCameraConsumer {

    private static final String TAG = "ImageSaveConsumer";

    private int imageNumber = 0;
    private final FPSConstrainedDeque trainingImageSaveQueue;
    private final boolean onlyImageGather;
    private final TrainingSaveImageTaskFactory taskFactory;

    public ImageSaveConsumer(ClassifyCamera cameraActivity,
                             FPSConstrainedDeque queue,
                             boolean onlyImageGather,
                             TrainingSaveImageTaskFactory taskFactory) {
        super(cameraActivity);
        this.trainingImageSaveQueue = queue;
        this.onlyImageGather = onlyImageGather;
        this.taskFactory = taskFactory;
    }

    @Override
    public boolean canConsumeImage() {
        return trainingImageSaveQueue.canAddNewElement(System.currentTimeMillis());
    }

    @Override
    public void onImageAvailable(AndroidCameraImageModel image) {
        if(!shouldConsumeNextImage()) {
            return;
        }

        //Log.d(TAG, "Adding new image element");

        trainingImageSaveQueue.addNewElement(true);

        ClassifyCamera activity = getActivity();
        AsyncTask<AndroidCameraImageModel, Void, Void> task = taskFactory.createNewTask(activity,
                System.currentTimeMillis(),
                onlyImageGather,
                imageNumber,
                trainingImageSaveQueue);

        activity.executeConsumerTask(ClassifyCamera.TaskExecutors.SAVE_TRAINING_IMAGES, task, image);
        imageNumber++;

        if(trainingImageSaveQueue.isSessionTerminated()) {
            activity.removeImageConsumer(this);
        }
    }
}
