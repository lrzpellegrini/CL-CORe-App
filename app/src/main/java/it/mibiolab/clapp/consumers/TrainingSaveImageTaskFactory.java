package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.model.AndroidCameraImageModel;
import it.mibiolab.clapp.utils.FPSConstrainedDeque;

public interface TrainingSaveImageTaskFactory {

    AsyncTask<AndroidCameraImageModel, Void, Void> createNewTask(
            ClassifyCamera activity,
            long saveStartTime,
            boolean onlyImageGather,
            int imageNumber,
            FPSConstrainedDeque queue);
}
