package it.mibiolab.clapp.consumers;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

/**
 * Consumer used to receive the image to be stored as the category thumbnail.
 *
 * The actual procedure is handled in the {@link SaveThumbnailTaskFromYUV} task.
 */
public class ThumbnailSaveConsumer extends AbstractClassifyCameraConsumer {

    private int countDown;

    public ThumbnailSaveConsumer(ClassifyCamera cameraActivity, int frameNumber) {
        super(cameraActivity);
        if(frameNumber <= 0) {
            throw new IllegalArgumentException("Invalid frame number: " + frameNumber);
        }

        this.countDown = frameNumber;
    }

    @Override
    public boolean canConsumeImage() {
        if(countDown == 0) {
            return false;
        }

        countDown--;
        return countDown == 0;
    }

    @Override
    public void onImageAvailable(AndroidCameraImageModel image) {
        if(!shouldConsumeNextImage()) {
            return;
        }

        ClassifyCamera activity = getActivity();
        activity.removeImageConsumer(this);
        SaveThumbnailTaskFromYUV task = new SaveThumbnailTaskFromYUV(activity, System.currentTimeMillis());
        activity.executeConsumerTask(ClassifyCamera.TaskExecutors.SAVE_THUMBNAIL, task, image);
    }
}
