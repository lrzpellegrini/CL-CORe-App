package it.mibiolab.clapp;

import android.app.Application;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is the Application class.
 *
 * This is more convoluted than your usual Application as it serves two purposes:
 * - It's a singleton containing paths, the thumbnails and categories list, etcetera.
 * - It's the class that defines the JNI bindings (the "native" methods).
 *
 * Most of the methods are self-explanatory bookkeeping utilities that
 * are used extensively through the app.
 *
 * Once loaded, the class will load the C++ library (called "native-lib") that
 * implements the inference and training procedures .
 */
public class MyApplication extends Application {

    private static final String TAG = "MyApplication";

    private File rootPath;
    private String statePath;
    private String tmpThumbnailPath;
    private String trainingImagesPath;
    private File trainingImagesZipDir;

    private List<String> categories;
    private List<String> thumbnails;
    private List<Bitmap> thumbnailsBitmap;

    private final AtomicInteger trainingProgress = new AtomicInteger();

    static {
        System.loadLibrary("native-lib");
    }

    @Override
    public void onCreate() {
        super.onCreate();

        rootPath = new File(Environment.getExternalStorageDirectory() +
                "/CaffeAndroid");
        statePath = new File(rootPath, "CwrAppState").getAbsolutePath();
        tmpThumbnailPath = new File(rootPath, "tmpThumbnail.jpg").getAbsolutePath();

        File trainingImagesPathFile = new File(rootPath, "TrainingFiles");

        trainingImagesPath = trainingImagesPathFile.getAbsolutePath();
        trainingImagesZipDir = new File(rootPath, "TrainingBatches");

        categories = null;
        thumbnails = null;
        thumbnailsBitmap = null;
    }

    public String getStatePath() {
        return statePath;
    }

    public String getTmpThumbnailPath() {
        return tmpThumbnailPath;
    }

    public String getTrainingImagesPath() {
        return trainingImagesPath;
    }

    public File getTrainingImagesZipDir() {
        return trainingImagesZipDir;
    }

    public File generateTrainingImagesZipPath(int categoryId) {
        // https://stackoverflow.com/a/7489888

        String categoryLabel = getCategoryLabel(categoryId);
        String fileName = new SimpleDateFormat(
                "'Cat-" + categoryLabel + "-'yyyyMMddHHmmss'.zip'",
                Locale.getDefault()).format(new Date());
        return new File(trainingImagesZipDir, fileName);
    }

    public int getCategoriesCount() {
        return categories.size();
    }

    public int addCategory(String category, String thumbnailPath) {
        categories.add(category);
        thumbnails.add(thumbnailPath);
        thumbnailsBitmap.add(loadThumbnail(thumbnailPath));
        return categories.size() - 1;
    }

    public List<String> getCategories() {
        return new ArrayList<>(categories);
    }

    public String getCategoryLabel(int categoryId) {
        return categories.get(categoryId);
    }

    public String getCategoryThumbnailPath(int categoryId) {
        return thumbnails.get(categoryId);
    }

    public Bitmap getCategoryThumbnail(int categoryId) {
        return thumbnailsBitmap.get(categoryId);
    }

    public String renameCategory(int categoryId, String newName) {
        return categories.set(categoryId, newName);
    }

    public String replaceCategoryThumbnail(int categoryId, String newThumbnailPath) {
        thumbnailsBitmap.set(categoryId, loadThumbnail(newThumbnailPath));
        return thumbnails.set(categoryId, newThumbnailPath);
    }

    public void loadInitialCategories(List<String> initialCategories,
                                      List<String> initialThumbnails) {
        if(initialCategories.size() != initialThumbnails.size()) {
            throw new IllegalArgumentException("Initial categories number differ from thumnails " +
                    "number: " + initialCategories.size() + " vs " + initialThumbnails.size());
        }

        categories = new ArrayList<>(initialCategories);
        thumbnails = new ArrayList<>(initialThumbnails);
        thumbnailsBitmap = new ArrayList<>(thumbnails.size());
        for (String thumbnailPath : thumbnails) {
            thumbnailsBitmap.add(loadThumbnail(thumbnailPath));
        }
    }

    public String getDefaultThumbnailPath(int categoryId) {
        File pathSDCard = new File(Environment.getExternalStorageDirectory() +
                "/CaffeAndroid/" + "cat" + categoryId + "_thumbnail.png");
        return pathSDCard.getAbsolutePath();
    }

    public void resetAppState(ResetProcedureListener listener) { ;
        try {
            FileUtils.deleteDirectory(rootPath);
        } catch (IOException e) {
            Log.e(TAG, "Error deleting status folder", e);
        }

        listener.onResetCompleted();
    }

    public float getTrainingProgress() {
        return Float.intBitsToFloat(trainingProgress.get());
    }

    public void setTrainingProgress(float trainProgress) {
        this.trainingProgress.set(Float.floatToIntBits(trainProgress));
    }

    public String[] cwrGetCurrentLabels() {
        return (String[]) cwrGetLabels();
    }

    private Bitmap loadThumbnail(String thumbnail) {
        return BitmapFactory.decodeFile(thumbnail);
    }

    /* New CwrApp functions */

    /**
     * Initializes the C++ library. This is invoked the first time the app is started (or after a
     * reset). This provides all the paths to initial resources.
     *
     * The app will save a new state in the internal memory after each train iteration. That state
     * will be loaded next time the app is opened. This means that the
     * {@link PermissionActivity} will call {@link #cwrReloadApp(String, int, int)} instead of this
     * function.
     *
     * @param solverPath The path to the caffe solver definition.
     * @param initialWeightsPath The path to the file containing the model weights.
     * @param initialClassUpdates The class updated balancing factor used by CWR.
     * @param trainEpochs The amount of training epoch to run.
     * @param initialClassesNumber The amount of registered categories.
     * @param cwrLayers The list of layers used by the CWR algorithm. This is usually an array with
     *                  a single string (the name of the last fully connected layer).
     * @param predictionLayer The name of the layer that will output the final prediction.
     * @param preExtractLayer The name of the layer to use as the latent replay layer. In this
     *                        version of the app the last pooling layer is used.
     * @param labels The list of category labels.
     * @param BMean The mean value for the blue channel.
     * @param GMean The mean value for the green channel.
     * @param RMean The mean value for the red channel.
     * @param trainThreads How many threads to use when training.
     * @param featureExtractionThreads How many threads to use when extracting the latent features.
     * @param rehearsalLayer The replay layer. Usually the same of preExtractLayer.
     * @param initialRehearsalBlobPath The path to the file containing the replay buffer.
     */
    public native void cwrInitApp(
            String solverPath,
            String initialWeightsPath,
            float[] initialClassUpdates,
            int trainEpochs,
            int initialClassesNumber,
            String[] cwrLayers,
            String predictionLayer,
            String preExtractLayer,
            String[] labels,
            float BMean,
            float GMean,
            float RMean,
            int trainThreads,
            int featureExtractionThreads,
            String rehearsalLayer,
            String initialRehearsalBlobPath);

    /**
     * Reload previous data for the C++ library.
     *
     * @param savePath The path to the folder in which data was saved.
     * @param trainThreads How many threads to use when training.
     * @param featureExtractionThreads How many threads to use when extracting the latent features.
     */
    public native void cwrReloadApp(String savePath,
                                    int trainThreads,
                                    int featureExtractionThreads);

    /**
     * Check if data already exist for the C++ library.
     *
     * @param savePath The path to the folder to check.
     * @return True if previous data exist, false otherwise.
     */
    public native boolean cwrPreviousStateExists(String savePath);

    /**
     * Store the C++ library state.
     *
     * @param savePath The path to the folder where to store the state.
     */
    public native void cwrSaveAppState(String savePath);

    /**
     * Add a training image to the current batch.
     *
     * During this process the latent activation for the image will be extracted and kept in memory
     * until the training procedure is triggered by calling {@link #cwrTrainingStep(int)}.
     *
     * @param Y The Y Data of the YUV image
     * @param U The U Data of the YUV image
     * @param V The V Data of the YUV image
     * @param w The width of the image
     * @param h The height of the image
     * @param rotation The rotation of the image
     * @param cropX The x position of the crop (greyed area in the UI)
     * @param cropY The y position of the crop (greyed area in the UI)
     * @param cropW The width of the crop (greyed area in the UI)
     * @param cropH The height of the crop (greyed area in the UI)
     */
    public native void cwrAddTrainingImageFromYUV(
            byte[] Y,
            byte[] U,
            byte[] V,
            int w,
            int h,
            int rotation,
            int cropX, int cropY, int cropW, int cropH);

    public native void cwrAddTrainingImageFromJpeg(
            byte[] jpegBytes,
            int rotation,
            int cropX, int cropY, int cropW, int cropH);

    /**
     * Discard the training batch.
     *
     * This will remove the previously prepared latent activations from the memory.
     */
    public native void cwrClearTrainingBatch();

    /**
     * Run a training step with the previously prepared latent activations.
     * @param label The category ID.
     */
    public native void cwrTrainingStep(int label);

    /**
     * Run inference for an image.
     * @param Y The Y Data of the YUV image
     * @param U The U Data of the YUV image
     * @param V The V Data of the YUV image
     * @param w The width of the image
     * @param h The height of the image
     * @param rotation The rotation of the image
     * @param cropX The x position of the crop (greyed area in the UI)
     * @param cropY The y position of the crop (greyed area in the UI)
     * @param cropW The width of the crop (greyed area in the UI)
     * @param cropH The height of the crop (greyed area in the UI)
     * @param predictions An empty float array to be used to store the confidence scores for each
     *                    class. Must be of size equal to the amount of registered categories.
     */
    public native void cwrInferenceFromYUV(
            byte[] Y,
            byte[] U,
            byte[] V,
            int w,
            int h,
            int rotation,
            int cropX, int cropY, int cropW, int cropH,
            float[] predictions);

    public native void cwrInferenceFromJpegBytes(
            byte[] jpegBytes,
            int rotation,
            int cropX, int cropY, int cropW, int cropH,
            float[] predictions);

    public native void cwrInferenceFromFile(
            String pattern,
            int cropX, int cropY, int cropW, int cropH,
            float[] predictions);

    /**
     * Get the category list.
     * @return The category list as an array of strings.
     */
    private native Object[] cwrGetLabels();

    /**
     * Add a new category.
     * @param categoryName The category name
     * @return The new category ID.
     */
    public native int cwrAddNewCategory(String categoryName);

    /**
     * Get the maximum number of categories (15).
     *
     * @return The maximum number of categories.
     */
    public native int cwrGetMaxCategories();

    /**
     * Saves an image from the YUV raw data.
     */
    public native void cwrSaveThumbnailFromYUV(
            byte[] Y,
            byte[] U,
            byte[] V,
            int w,
            int h,
            int rotation,
            int cropX, int cropY, int cropW, int cropH,
            int targetW, int targetH,
            String savePath) throws RuntimeException;

    public interface ResetProcedureListener {
        void onResetCompleted();
    }
}
