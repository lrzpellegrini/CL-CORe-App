package it.mibiolab.clapp.consumers;

import it.mibiolab.clapp.model.AndroidCameraImageModel;

/**
 * Interface for image consumers.
 *
 * Image consumers are registered in the {@link it.mibiolab.clapp.ClassifyCamera} activity based
 * on the current phase.
 *
 * All the consumers in this package extend from {@link AbstractClassifyCameraConsumer}, that implements
 * this interface.
 */
public interface ImageConsumer {

    /**
     * Returns true if the consumer needs/is ready to accept the next image.
     * @return True if the next image should be passed to {@link #onImageAvailable(AndroidCameraImageModel)},
     * false otherwise.
     */
    boolean needsNextImage();

    /**
     * Receives and processes the input image.
     *
     * @param image The structure containing the YUV raw data, the crop location, size, etcetera.
     */
    void onImageAvailable(AndroidCameraImageModel image);
}
