package it.mibiolab.clapp.consumers;

import android.os.AsyncTask;

import it.mibiolab.clapp.ClassifyCamera;
import it.mibiolab.clapp.model.AndroidCameraImageModel;

public interface PredictionTaskFactory {

    AsyncTask<AndroidCameraImageModel, Void, float[]> createNewTask(
            ClassifyCamera activity,
            long startTime);
}
