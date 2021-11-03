package it.mibiolab.clapp.consumers;

import java.lang.ref.WeakReference;
import java.security.Key;

import it.mibiolab.clapp.ClassifyCamera;

/**
 * Abstract implementation of an image consumer.
 */
public abstract class AbstractClassifyCameraConsumer implements ImageConsumer {

    private boolean shouldConsumeNext = false;
    private ClassifyCamera activity = null;
    private final WeakReference<ClassifyCamera> cameraActivity;

    public AbstractClassifyCameraConsumer(ClassifyCamera cameraActivity) {
        this.cameraActivity = new WeakReference<>(cameraActivity);
    }

    protected final ClassifyCamera getActivity() {
        ClassifyCamera result = activity;
        if(result == null) {
            result = cameraActivity.get();
        }

        activity = null;
        return result;
    }

    @Override
    public final boolean needsNextImage() {
        shouldConsumeNext = canConsumeImage();
        if(shouldConsumeNext) {
            activity = cameraActivity.get();
            if(activity == null) {
                shouldConsumeNext = false;
            }
        }
        return shouldConsumeNext;
    }

    protected final boolean shouldConsumeNextImage() {
        return shouldConsumeNext;
    }

    protected abstract boolean canConsumeImage();
}

