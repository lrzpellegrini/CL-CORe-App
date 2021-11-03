package it.mibiolab.clapp;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.Layout;
import android.text.StaticLayout;
import android.text.TextPaint;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.File;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import it.mibiolab.clapp.consumers.ImageConsumer;
import it.mibiolab.clapp.consumers.ImageSaveConsumer;
import it.mibiolab.clapp.consumers.InferenceConsumer;
import it.mibiolab.clapp.consumers.PredictionTaskFromJpeg;
import it.mibiolab.clapp.consumers.PredictionTaskFromYUV;
import it.mibiolab.clapp.consumers.ThumbnailSaveConsumer;
import it.mibiolab.clapp.consumers.TrainingSaveImageTaskFromYUV;
import it.mibiolab.clapp.model.AndroidCameraImageModel;
import it.mibiolab.clapp.model.LazyImageModel;
import it.mibiolab.clapp.utils.CloseableReentrantLock;
import it.mibiolab.clapp.utils.FPSConstrainedDeque;
import it.mibiolab.clapp.utils.FPSCounter;
import it.mibiolab.clapp.utils.ResourceLock;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

// It is recommended to jump to the first ">>  README  <<" comment!
// The readme sections will guide you through the code (which is quite complex).
/**
 * The main activity of the application.
 *
 * This activity handles the logic of the inference (idle), image gathering and training phases.
 *
 * It's quite a complex code that involves:
 * - Handling the graphic aspects of the upper part of the UI
 * - Interfacing with the Camera2 image acquisition component
 * - Managing of the state of the other parts of the application (image gathering, training, ...)
 *
 * Most of the lines of code are "spent" in Java-related overhead and Android-specific stuff.
 *
 * The activity doesn't directly manage the inference and image gathering aspects. Those aspects
 * are managed by image consumers implemented in {@link InferenceConsumer} and
 * {@link ImageSaveConsumer}.
 */
public class ClassifyCamera extends AppCompatActivity implements ClassSelectionFragment.OnFragmentInteractionListener {

    private static final String TAG = "ClassifyCamera";

    private static final boolean SHOW_DEBUG_TEXT_VIEW = false;
    //private static final boolean SAVE_TRAINING_IMAGES = false;
    private static final boolean INFERENCE_WHILE_TRAINING = false;
    //private static final boolean ONLY_IMAGE_GATHER = false;

    private static final int RANKING_SIZE = 3;

    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final int CAPTURE_W = 768;
    private static final int CAPTURE_H = 1024;
    private static final double MIN_ASPECT_RATIO = 0.74;
    private static final double MAX_ASPECT_RATIO = 0.76;
    private static final int CAPTURE_CROP_W = 128 * 4; //Core50 image side * 4
    private static final int CAPTURE_CROP_H = 128 * 4;
    private static final boolean DRAW_PREVIEW_FRAME = true;
    private static final int PREVIEW_CROP_SHADOW_COLOR = Color.argb(180, 40, 40, 40);
    private static final int PREVIEW_CROP_FRAME_COLOR = Color.argb(255, 0, 0, 0);
    private static final double TRAINING_FPS = 5.0;
    private static final int TRAINING_BATCH_SIZE = 100;
    private static final int THUMBNAIL_CAPTURE_FRAME_NUMBER = 20;
    private static final int INFERENCE_SMOOTHING_FACTOR = 6;

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private final FPSConstrainedDeque trainingImageSaveQueue =
            new FPSConstrainedDeque((int) Math.ceil(TRAINING_FPS * 2.0), TRAINING_FPS);
    private final CloseableReentrantLock trainingMutex = new CloseableReentrantLock(true); //TODO: use a service instead (requires major refactor)
    private CameraDevice cameraDevice;
    private ImageReader imageReader;
    private CameraCaptureSession cameraCaptureSessions;
    private CaptureRequest.Builder captureRequestBuilder;
    private TextureView textureView;
    private SurfaceView shadowView;
    private String cameraId;
    private Size imageDimension;
    private int imageRotation;
    private TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private TextView debug_tv;
    private TextView status_tv;
    private TextView info_tv;

    private boolean onlyImageGather = false;
    private DescriptiveStatistics processingTime = new DescriptiveStatistics(30);
    private FPSCounter fps = new FPSCounter(30, System.currentTimeMillis());
    private DescriptiveStatistics[] predictionsInTime = null;
    private double[] lastPredictions = null;
    private int previousPredictedCategory = -1;
    private volatile TrainingPhase trainingPhase = TrainingPhase.IDLE; // Background + UI thread + async task
    private volatile boolean thumbnailSaved = false; // Background + UI thread + async task
    private volatile int trainingCategoryId = -1; // Background + UI thread
    private volatile String trainingCategoryName = null; // Background + UI thread
    private volatile boolean trainingCategoryIsNew = false; // Background + UI thread
    private volatile boolean trainingReplaceThumbnail = false; // Background + UI thread
    private volatile int currentTrainingBatchSize = 0; // Background + UI thread
    private volatile long phaseStartTimeMs = System.currentTimeMillis();

    private List<ImageConsumer> imageConsumers = new CopyOnWriteArrayList<>();
    private InferenceConsumer inferenceConsumer = null;
    private ImageSaveConsumer imageSaveConsumer = null;


    private int maxConcurrentSaveTasks;
    private ExecuteTrainingTask trainingTask = null;
    private ExecutorService trainingSaveImageTasksExecutor;
    private MyApplication application = null;

    /**
     * This field will contains a callback for managing the basic Camera2 startup and shutdown
     * operations.
     */
    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            cameraDevice = camera;
            try {
                createCameraPreview();
            } catch (CameraAccessException e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            if (cameraDevice != null) {
                cameraDevice.close();
            }
            cameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            if (cameraDevice != null) {
                cameraDevice.close();
            }
            cameraDevice = null;
        }
    };
    private ClassSelectionFragment classSelectionFragment = null;
    private CountDownTimer countDown = null;

    /*////////////// >>  README  << /////////////////////
    In the next part of the code I tried to arrange the methods in a meaningful way, by following the
    natural user interaction order.

    The app is managed as a state machine with the following flow:
    - Idle Phase (entered from onCreate, or at the end of the training phase, or when closing and resuming the application)
        The Idle phase is the "inference-only" one.
        The Idle phase is started by calling startIdlePhase()
    - Countdown phase (entered from the idle phase after selecting an existing or new category slot)
        During this phase a simple countdown is shown.
        The Countdown phase is started by calling startCountdownPhase()
        Images obtained from the camera during this phase are discarded.
    - Gathering phase (entered at the end of the countdown phase)
        During this phase images are gathered and the latent features are extracted.
        The Gathering phase is stated by calling startTrainingImageGatherPhase()
    - Training phase (entered at the end of the gathering phase)
        During this phase the latent activations + instances from the replay buffer are used to
        incrementally train the model using the AR1 algorithm by running the replay at the
        last layer of the model.
        The Training phase is started by calling startTrainingPhase()
        Images obtained from the camera during this phase are discarded.
     - Back to idle phase.

     The next part of the file will contain the methods used to switch between those states in the
     order described above.

     Beware that these methods don't code the logic used to manage the image flow and the UI at
     the upper part of the screen. That part is managed in the createCameraPreview() method which
     will change its behaviour according to the current phase.
    ////////////////////////////////////////////*/

    /**
     * The onCreate method handles the setup of the UI and the internal state of the activity.
     *
     * Here the upper part of the UI is set up while the lower part is managed by a fragment.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        application = getMyApplication();

        maxConcurrentSaveTasks = Runtime.getRuntime().availableProcessors();
        trainingSaveImageTasksExecutor = Executors.newFixedThreadPool(maxConcurrentSaveTasks);
        Log.d(TAG, "Will use " + maxConcurrentSaveTasks + " threads");

        setContentView(R.layout.activity_classify_camera);

        Toolbar myToolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);

        textureView = findViewById(R.id.textureView);
        textureView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        textureView.setSurfaceTextureListener(textureListener);

        shadowView = findViewById(R.id.shadowView);

        // adapted from https://stackoverflow.com/a/51915218
        shadowView.setZOrderOnTop(true);
        SurfaceHolder mHolder = shadowView.getHolder();
        mHolder.setFormat(PixelFormat.TRANSPARENT);

        debug_tv = findViewById(R.id.debug_tv);
        if (SHOW_DEBUG_TEXT_VIEW) {
            debug_tv.setVisibility(View.VISIBLE);
        }

        status_tv = findViewById(R.id.status_tv);
        info_tv = findViewById(R.id.info_tv);

        String initialStatusText = "";
        for (int i = 0; i < RANKING_SIZE; i++) {
            initialStatusText += '\n';
        }
        status_tv.setText(initialStatusText);
        info_tv.setText("");

        classSelectionFragment = ((ClassSelectionFragment) getSupportFragmentManager()
                .findFragmentById(R.id.class_selection_component));

        findViewById(R.id.reset_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startResetProcedure();
            }
        });

        /*
         * The application starts in inference mode, in which the application continuously receives
         * images from the camera to be recognized. This initial mode is called "Idle mode"
         * throughout the code.
         */
        startIdlePhase();
    }

    /**
     * Method to manage the switch to the Idle (inference-only) phase.
     *
     * In practice, this will just set the this.trainingPhase field, re-enable buttons in the
     * lower part of the UI, remove all pending image consumers, and reset the running confidence
     * scores.
     */
    private void startIdlePhase() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if(trainingPhase == null ||
                    trainingPhase == TrainingPhase.EXECUTING_TRAINING ||
                    trainingPhase == TrainingPhase.IDLE) {
                // Set the trainingPhase field
                trainingPhase = TrainingPhase.IDLE;

                // Reset the start time and reset the FPS counter
                phaseStartTimeMs = System.currentTimeMillis();
                fps.clear(phaseStartTimeMs);

                // Reset the category confidence scores
                predictionsInTime = new DescriptiveStatistics[application.getCategoriesCount()];
                lastPredictions = new double[application.getCategoriesCount()];
                for (int i = 0; i < predictionsInTime.length; i++) {
                    predictionsInTime[i] = new DescriptiveStatistics(INFERENCE_SMOOTHING_FACTOR);
                }

                // (Re-)enable the lower part of the UI (the one with the thumbnails)
                // This is needed as, during the training phase, they are disabled.
                classSelectionFragment.enableSelection(true);
                classSelectionFragment.recreateButtons();

                // Remove any pending image consumer. Add the inference consumer.
                imageConsumers.clear();
                imageConsumers.add(inferenceConsumer = new InferenceConsumer(
                        this, PredictionTaskFromJpeg.getFactory()));

                info_tv.setText("");
            }
        }
    }

    /**
     * The callback method used to receive info from the fragment that manages the lower part of the UI.
     *
     * The lower part of the UI is the one with the thumbnails. To trigger a training phase, one has
     * to interact with those thumbnails. The fragment will take care of prompts and alerts needed to confirm
     * the name of the new category, ask if the thumbnail must be replaced, etcetera.
     *
     * At the end of the user interaction, the fragment will call this callback to trigger the
     * countdown phase (see {@link #startCountdownPhase()}).
     *
     * The fragment is implemented in the {@link ClassSelectionFragment}.
     *
     * @param categoryIndex The index/ID of the category.
     * @param categoryName The name of the category.
     * @param isNewCategory If true, an empty slot was selected.
     * @param replaceThumbnail If true, the thumbnail for that category will be replaced with a new one.
     */
    @Override
    public void onCategorySelected(int categoryIndex, String categoryName, boolean isNewCategory, boolean replaceThumbnail) {
        Log.d(TAG, "Selected class " + categoryIndex);
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if (trainingPhase == TrainingPhase.IDLE) {
                this.trainingCategoryId = categoryIndex;
                this.trainingCategoryName = categoryName;
                this.trainingCategoryIsNew = isNewCategory;
                this.trainingReplaceThumbnail = replaceThumbnail;
                startCountdownPhase(); // Transition from the idle to the countdown phase
            }
        }
    }

    /**
     * Method used to manage the switch to the Countdown phase.
     *
     * This will simply create a timer that will then start the Gathering phase.
     *
     * During the gathering phase, as short countdown will be shown on screen.
     */
    private void startCountdownPhase() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if (trainingPhase == TrainingPhase.IDLE) {
                trainingPhase = TrainingPhase.COUNTDOWN;

                countDown = new CountDownTimer(3000, 100) {
                    @Override
                    public void onTick(long millisUntilFinished) {
                        info_tv.setText(""+Math.max(1, (int) Math.ceil((double) millisUntilFinished / 1000.0)));
                        status_tv.setText("");
                    }

                    @Override
                    public void onFinish() {
                        startTrainingImageGatherPhase(); // Transition from the countdown to the gathering phase
                    }
                }.start();
            }
        }
    }

    /**
     * Method used to manage the switch to the Gathering phase.
     *
     * In this phase, images are gathered from the camera and then sent to the C++ library to
     * extract the latent features.
     *
     * Simply put, this method will create a {@link ImageSaveConsumer} consumer that will
     * receive the images.
     *
     * If needed, a {@link ThumbnailSaveConsumer} consumer will be added to store a thumbnail
     * for the category.
     */
    private void startTrainingImageGatherPhase() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if (trainingPhase == TrainingPhase.COUNTDOWN) {
                trainingPhase = TrainingPhase.GATHERING_IMAGES;

                phaseStartTimeMs = System.currentTimeMillis();
                currentTrainingBatchSize = 0;
                fps.clear(phaseStartTimeMs);
                trainingImageSaveQueue.startSession(TRAINING_BATCH_SIZE);

                imageSaveConsumer = new ImageSaveConsumer(this,
                        trainingImageSaveQueue,
                        onlyImageGather,
                        TrainingSaveImageTaskFromYUV.getFactory());
                this.imageConsumers.add(imageSaveConsumer);

                if (!INFERENCE_WHILE_TRAINING) {
                    if(inferenceConsumer != null) {
                        this.imageConsumers.remove(inferenceConsumer);
                        inferenceConsumer = null;
                    }
                }

                File prevThumb = new File(application.getTmpThumbnailPath());
                if (prevThumb.exists()) {
                    prevThumb.delete();
                }

                if(trainingReplaceThumbnail) {
                    imageConsumers.add(new ThumbnailSaveConsumer(this, THUMBNAIL_CAPTURE_FRAME_NUMBER));
                }

                classSelectionFragment.enableSelection(false);
                if (!INFERENCE_WHILE_TRAINING) {
                    classSelectionFragment.clearPredictions();
                }
            }
        }

        if (!INFERENCE_WHILE_TRAINING) {
            String gatheringImagesPhaseText = getString(R.string.gathering_images_status);
            for (int i = 0; i < RANKING_SIZE - 1; i++) {
                gatheringImagesPhaseText += '\n';
            }
            info_tv.setText(gatheringImagesPhaseText);
            status_tv.setText("");
        }
    }

    /**
     * Method called each time a training image has been processed.
     *
     * The processing procedure involves converting the image from YUV to RGB,
     * forwarding it through the model and then extracting the latent activations.
     *
     * Once {@link #TRAINING_BATCH_SIZE} images have been gathered and the new thumbnail has been
     * saved (if needed), the {@link #completeTrainingImageGatherPhase()} method is called to
     * switch to the training phase.
     *
     * @param endTimeMs Timestamp describing when the image processing procedure was completed. Used
     *                  to compute the amount of FPS.
     */
    public void onTrainingImageSaveCompleted(long endTimeMs) {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            fps.updateTime(endTimeMs);
            currentTrainingBatchSize++;
            //Log.d(TAG, "Saved images: " + currentTrainingBatchSize);
            if (trainingPhase == TrainingPhase.GATHERING_IMAGES &&
                    currentTrainingBatchSize >= TRAINING_BATCH_SIZE &&
                    (thumbnailSaved || !trainingReplaceThumbnail)) {
                //Log.d(TAG, "Saved images: completing");
                completeTrainingImageGatherPhase(); // Complete the gathering phase
                //Log.d(TAG, "Saved images: completed");
            }

            if (trainingPhase == TrainingPhase.GATHERING_IMAGES &&
                    currentTrainingBatchSize >= TRAINING_BATCH_SIZE) {
                Log.d(TAG, "Discarding image...");
            }
        }
    }

    /**
     * Method called once the thumbnail has been saved.
     *
     * Once {@link #TRAINING_BATCH_SIZE} images have been gathered and the new thumbnail has been
     * saved (if needed), the {@link #completeTrainingImageGatherPhase()} method is called to
     * switch to the training phase.
     */
    public void onThumbnailSaved() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            //Log.d(TAG, "Saved images: thumbnail");
            thumbnailSaved = true;
            if (trainingPhase == TrainingPhase.GATHERING_IMAGES &&
                    currentTrainingBatchSize >= TRAINING_BATCH_SIZE) {
                completeTrainingImageGatherPhase(); // Complete the gathering phase
            }
        }
    }

    /**
     * Method called when all the training images have been gathered and the new thumbnail stored.
     *
     * This method will show a toast message to the user and switch to the training phase.
     */
    private void completeTrainingImageGatherPhase() {
        int savedImages;
        long phaseStart;

        try (ResourceLock rl = trainingMutex.lockAsResource()){
            savedImages = currentTrainingBatchSize;
            phaseStart = phaseStartTimeMs;

            startTrainingPhase();  // Transition from the gathering to the training phase
        }

        long phaseTime = System.currentTimeMillis() - phaseStart;
        String message = String.format(Locale.US, "Gathered %d images in %d", savedImages, Math.round(phaseTime / 1000.0));
        Toast.makeText(ClassifyCamera.this, message, Toast.LENGTH_SHORT).show();
    }

    /**
     * Method called to start the training phase.
     *
     * This will reset the progress bar and add run the {@link ExecuteTrainingTask}.
     */
    private void startTrainingPhase() {

        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if (trainingPhase == TrainingPhase.GATHERING_IMAGES) {
                trainingPhase = TrainingPhase.EXECUTING_TRAINING;
                //trainingSaveImageTasks.clear();

                Log.d(TAG, "Starting training phase");

                if (onlyImageGather) {
                    completeTrainingPhase();
                } else {
                    application.setTrainingProgress(0.0f);
                    long trainingStartTime = System.currentTimeMillis();

                    trainingTask = new ExecuteTrainingTask(this,
                            application,
                            trainingStartTime);

                    trainingTask.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR, (Void) null);
                }
            }
        }

        if (!INFERENCE_WHILE_TRAINING) {
            String trainingPhaseText = getString(R.string.training_status, 0.0f);
            for (int i = 0; i < RANKING_SIZE - 1; i++) {
                trainingPhaseText += '\n';
            }
            info_tv.setText(trainingPhaseText);
            status_tv.setText("");
        }
    }

    /**
     * Method called upon completion of the training procedure.
     *
     * This will store the persistent application data (the updated model,
     * the new thumbnail, the new category name, etc.) and then switch back to idle mode.
     */
    private void completeTrainingPhase() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if (trainingPhase == TrainingPhase.EXECUTING_TRAINING) {
                trainingTask = null;

                if (trainingReplaceThumbnail) {

                    File toBeCopiedTo = new File(application.getDefaultThumbnailPath(trainingCategoryId));
                    if (toBeCopiedTo.exists()) {
                        toBeCopiedTo.delete();
                    }

                    File tmpThumbnailPath = new File(application.getTmpThumbnailPath());
                    if (!tmpThumbnailPath.renameTo(toBeCopiedTo)) {
                        throw new RuntimeException("Can't replace thumbnail at path: " + toBeCopiedTo.getAbsolutePath());
                    }

                    if (trainingCategoryIsNew) {
                        application.addCategory(trainingCategoryName, toBeCopiedTo.getAbsolutePath());
                    } else {
                        application.replaceCategoryThumbnail(trainingCategoryId, toBeCopiedTo.getAbsolutePath());
                    }

                }

                if (trainingCategoryName != null && !trainingCategoryIsNew) {
                    // Not implemented yet...
                    application.renameCategory(trainingCategoryId, trainingCategoryName);
                }

                Log.d(TAG, "IDLE phase... clearning");
                application.cwrClearTrainingBatch();
                Log.d(TAG, "IDLE phase... after clear");
                application.cwrSaveAppState(application.getStatePath());
                Log.d(TAG, "IDLE phase... saving");

                if (onlyImageGather) {
                    // Pack to -> zip
                    File trainingImagesPath = new File(application.getTrainingImagesPath());
                    ZipUtils zipUtil = new ZipUtils(trainingImagesPath.getAbsolutePath());
                    try {
                        File zipPath = application.generateTrainingImagesZipPath(trainingCategoryId);
                        zipUtil.zipIt(zipPath);

                        FileUtils.deleteDirectory(trainingImagesPath);
                        trainingImagesPath.mkdirs();

                        Toast.makeText(this, "Zip created (" + zipPath.getName() + ")", Toast.LENGTH_SHORT).show();
                    } catch (IOException e) {
                        Log.e(TAG, "Error creating zip (or deleting images folder)!", e);
                        Toast.makeText(this, "Error while creating zip!", Toast.LENGTH_LONG).show();
                    }
                } else {
                    Toast.makeText(this, "Training completed!", Toast.LENGTH_SHORT).show();
                }

                Log.d(TAG, "Switching back to IDLE phase");
                startIdlePhase(); // Transition from the training to the idle phase
            }
        }
    }

    /*///////////////// >>  README  << /////////////////////////
    This ends the methods used to manage the app state.
    The next methods manage the image acquisition process and the UI.

    The main loop is managed by a listener created in the following createCameraPreview() method.
    //////////////////////////////////////////*/

    /**
     * Implementation of the image acquisition procedure.
     *
     * The procedure is handled by a listener that receives images using the Camera2 API.
     *
     * That listener will dispatch images to any registered consumer. Consumers are registered when
     * switching to a new phase:
     * - Idle phase: InferenceConsumer
     * - Countdown phase: nothing
     * - Gathering phase: ImageSaveConsumer and (if needed) ThumbnailSaveConsumer
     * - Training phase: nothing
     *
     * You can find implementation for those consumers in the "consumers" package.
     *
     * After invoking the consumers, the method will then call
     * {@link #drawImageOverlay(Image, boolean, SurfaceHolder, Surface, int, int, int, int)} to draw
     * the UI (greyed area, central borders, the bottom-left confidence scores, progress bar,
     * etcetera).
     */
    protected void createCameraPreview() throws CameraAccessException {
        final SurfaceTexture texture = textureView.getSurfaceTexture();
        final SurfaceHolder shadowTexture = shadowView.getHolder();
        shadowTexture.setFormat(PixelFormat.TRANSPARENT);

        assert texture != null;
        texture.setDefaultBufferSize(
                imageDimension.getWidth(),
                imageDimension.getHeight());

        shadowTexture.setSizeFromLayout();

        final Surface surface = new Surface(texture);

        imageReader = ImageReader.newInstance(imageDimension.getWidth(),
                imageDimension.getHeight(), ImageFormat.YUV_420_888, 2);

        ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Image lastCameraImage = null;
                try {
                    lastCameraImage = reader.acquireLatestImage();

                    if (lastCameraImage == null) {
                        return;
                    }

                    final int w = lastCameraImage.getWidth();
                    final int h = lastCameraImage.getHeight();
                    final AndroidCameraImageModel predictionTaskData;

                    int cropX, cropY, cropW, cropH, thumbCropX, thumbCropY, thumbCropW, thumbCropH;
                    if (imageRotation == 90 || imageRotation == -90) {
                        /*
                        When keeping the phone in portrait, most Android smartphones will usually
                        return images rotated by 90 degrees (the more you know...). This means
                        we'll have to rotate the image and also adjust the central crop position.
                         */
                        cropX = (w - CAPTURE_CROP_H) / 2;
                        cropY = (h - CAPTURE_CROP_W) / 2;
                        cropW = CAPTURE_CROP_H;
                        cropH = CAPTURE_CROP_W;

                        thumbCropX = cropY;
                        thumbCropY = cropX;
                        thumbCropW = cropH;
                        thumbCropH = cropW;
                    } else {
                        cropX = (w - CAPTURE_CROP_W) / 2;
                        cropY = (h - CAPTURE_CROP_H) / 2;
                        cropW = CAPTURE_CROP_W;
                        cropH = CAPTURE_CROP_H;

                        thumbCropX = cropX;
                        thumbCropY = cropY;
                        thumbCropW = cropW;
                        thumbCropH = cropH;
                    }

                    /*Log.v(TAG, "Image size W = " + w + ", H = " + h);
                    Log.v(TAG, "Crop coords = (X " + cropX + ", Y " + cropY +
                            ", W " + cropW + ", H " + cropH + ")");*/

                    predictionTaskData = LazyImageModel.fromImage(
                            lastCameraImage, cropX, cropY, cropW, cropH, imageRotation);
                    /*
                    The image buffer is kept alive only if needed (if at least one consumer needs it)
                    Otherwise, it's discarded immediately.
                     */
                    boolean isImageNeeded = false;
                    for (ImageConsumer imageConsumer : imageConsumers) {
                        isImageNeeded = isImageNeeded || imageConsumer.needsNextImage();
                    }

                    if(isImageNeeded) {
                        if(predictionTaskData instanceof LazyImageModel) {
                            ((LazyImageModel)predictionTaskData).setInUse();
                        }
                        for (ImageConsumer imageConsumer : imageConsumers) {
                            imageConsumer.onImageAvailable(predictionTaskData);
                        }
                    }
                    /*
                    Call drawImageOverlay to draw the UI atop the camera preview.
                     */
                    final Surface shadowSurface = shadowTexture.getSurface();
                    drawImageOverlay(lastCameraImage,
                            imageRotation == 90 || imageRotation == -90,
                            shadowTexture, shadowSurface,
                            thumbCropX, thumbCropY, thumbCropW, thumbCropH);
                } catch (Exception e) {
                    Log.d(TAG, "Error processing lastCameraImage", e);
                } finally {
                    if (lastCameraImage != null) {
                        lastCameraImage.close();
                    }
                }
            }
        };

        imageReader.setOnImageAvailableListener(readerListener, mBackgroundHandler);

        captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        captureRequestBuilder.addTarget(surface);
        captureRequestBuilder.addTarget(imageReader.getSurface());

        cameraDevice.createCaptureSession(Arrays.asList(surface, imageReader.getSurface()), new CameraCaptureSession.StateCallback() {
            @Override
            public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                if (null == cameraDevice) {
                    return;
                }
                cameraCaptureSessions = cameraCaptureSession;
                updatePreview();
            }

            @Override
            public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    /*Toast.makeText(ClassifyCamera.this,
                            "Configuration change",
                            Toast.LENGTH_SHORT).show();*/
            }
        }, null);

    }

    private void drawImageOverlay(Image image,
                                  boolean rotated,
                                  SurfaceHolder surfaceHolder,
                                  Surface shadowSurface,
                                  int cropX,
                                  int cropY,
                                  int cropW,
                                  int cropH) {
        boolean isTraining;
        boolean isGatheringImages;
        boolean isCountdown;
        int currentBatchSize;

        try (ResourceLock rl = trainingMutex.lockAsResource()){
            isTraining = (trainingPhase == TrainingPhase.EXECUTING_TRAINING);
            isGatheringImages = (trainingPhase == TrainingPhase.GATHERING_IMAGES);
            isCountdown = (trainingPhase == TrainingPhase.COUNTDOWN);
            currentBatchSize = currentTrainingBatchSize;
        }

        Canvas c = surfaceHolder.lockCanvas();

        if (c == null) {
            Log.v(TAG, "Image surface is not valid");
            return;
        }

        try {
            /*Log.v(TAG, "Canvas size: w = " + c.getWidth() + " h = " + c.getHeight());
            Log.v(TAG, "Control size: w = " + shadowView.getWidth() + " h = " + shadowView.getHeight());
            Log.v(TAG, "ControlBack size: w = " + textureView.getWidth() + " h = " + textureView.getHeight());
            Log.v(TAG, String.format("Crop: (%d, %d, %d, %d)", cropX, cropY, cropW, cropH));
            Log.v(TAG, String.format("Img size: (%d, %d)", image.getWidth(), image.getHeight()));
            Log.v(TAG, String.format("Img ratio: %f", (double) image.getWidth() / image.getHeight()));
            Log.v(TAG, String.format("Shadow ratio: %f", (double) c.getWidth() / c.getHeight()));*/

            int realImageW = rotated ? image.getHeight() : image.getWidth();
            int realImageH = rotated ? image.getWidth() : image.getHeight();

            double densityRatio = (double) c.getWidth() / realImageW;

            /*Log.v(TAG, String.format("Density ratio w: %f, h %f", densityRatio,
                    (double) c.getHeight() / realImageH));*/

            int shadowMaxY = drawShadow(c, densityRatio, cropX, cropY, cropW, cropH);

            drawStatusText(c, densityRatio);
            if(getTrainingPhase() == TrainingPhase.EXECUTING_TRAINING) {
                final float progress = application.getTrainingProgress() * 100.0f;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        info_tv.setText(getString(R.string.training_status, progress));
                    }
                });
            }
            drawInfo(c, densityRatio, shadowMaxY);

            if (isGatheringImages) {
                int fpsTextMaxY = drawFps(c, densityRatio);
                drawProgressBar(c, densityRatio,
                        (double) currentBatchSize / TRAINING_BATCH_SIZE,
                        fpsTextMaxY);
            } else if(!isCountdown) {
                drawFps(c, densityRatio);
            }
        } finally {
            surfaceHolder.unlockCanvasAndPost(c);
        }
    }

    private int drawShadow(Canvas c,
                           double densityRatio,
                           int cropX,
                           int cropY,
                           int cropW,
                           int cropH) {
        int frameAdditionalLength = 30;
        int frameStroke = 7;

        Paint shadowPaint = new Paint();
        c.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);

        shadowPaint.setColor(PREVIEW_CROP_SHADOW_COLOR);
        shadowPaint.setStrokeWidth(1);
        shadowPaint.setStyle(Paint.Style.FILL);

        cropX *= densityRatio;
        cropY *= densityRatio;
        cropW *= densityRatio;
        cropH *= densityRatio;

        Rect top = new Rect(0, 0, c.getWidth(), cropY);
        Rect bottom = new Rect(0, cropY + cropH, c.getWidth(), c.getHeight());
        Rect left = new Rect(0, cropY, cropX, cropY + cropH);
        //Log.v(TAG, "Bottom: " + (cropY + cropH) + " , " + (c.getHeight() - cropY));

        Rect right = new Rect(cropX + cropW, cropY, c.getWidth(), cropY + cropH);

        c.drawRect(top, shadowPaint);
        c.drawRect(bottom, shadowPaint);
        c.drawRect(left, shadowPaint);
        c.drawRect(right, shadowPaint);

        if (DRAW_PREVIEW_FRAME) {
            Paint framePaint = new Paint();

            framePaint.setColor(PREVIEW_CROP_FRAME_COLOR);
            framePaint.setStrokeWidth(1);
            framePaint.setStyle(Paint.Style.FILL);

            c.drawRect(cropX - frameAdditionalLength, cropY - frameStroke, c.getWidth() - cropX + frameAdditionalLength, cropY, framePaint);//Top
            c.drawRect(cropX - frameAdditionalLength,
                    cropY + cropH,
                    c.getWidth() - cropX + frameAdditionalLength,
                    cropY + cropH + frameStroke, framePaint);//Bottom
            c.drawRect(cropX - frameStroke, cropY - frameAdditionalLength, cropX, c.getHeight() - cropY + frameAdditionalLength, framePaint);//left
            c.drawRect(c.getWidth() - cropX, cropY - frameAdditionalLength, c.getWidth() - cropX + frameStroke, c.getHeight() - cropY + frameAdditionalLength, framePaint);//Right
        }

        return cropY + cropH + frameStroke;
    }

    private void drawStatusText(Canvas c, double densityRatio) {
        TextPaint textPaint = new TextPaint();
        textPaint.setAntiAlias(true);
        textPaint.setTextSize(status_tv.getTextSize());
        textPaint.setColor(0xFFFFFFFF);
        textPaint.setTypeface(status_tv.getTypeface());

        int totalH = 0;

        String text = status_tv.getText().toString();
        for (String line : text.split("\n")) {
            //Log.d(TAG, "Drawing line: " + line);
            Rect bounds = new Rect();
            textPaint.getTextBounds(line, 0, line.length(), bounds);
            int width = (int) Math.ceil(textPaint.measureText(line));
            int height = bounds.height();
            totalH += (height * 1.5);
        }

        float actualH = c.getHeight() - totalH - 10f;

        for (String line : text.split("\n")) {
            Rect bounds = new Rect();
            textPaint.getTextBounds(line, 0, line.length(), bounds);
            int width = (int) Math.ceil(textPaint.measureText(line));
            int height = bounds.height();

            StaticLayout.Builder builder = StaticLayout.Builder.obtain(line, 0, line.length(), textPaint, width)
                    .setAlignment(Layout.Alignment.ALIGN_NORMAL)
                    .setLineSpacing(0, 1.0f)
                    .setIncludePad(true);

            StaticLayout staticLayout = builder.build();
            c.save();
            c.translate(10f, actualH);
            staticLayout.draw(c);
            c.restore();

            actualH += (height * 1.5);
        }
    }

    private void drawInfo(Canvas c, double densityRatio, int baseYOffset) {
        TextPaint textPaint = new TextPaint();
        textPaint.setAntiAlias(true);
        textPaint.setTextSize(info_tv.getTextSize());
        textPaint.setColor(0xFFFFFFFF);
        textPaint.setTypeface(info_tv.getTypeface());

        int totalH = 0;
        int singleLineH = 0;
        int maxWidth = 0;

        String text = info_tv.getText().toString();
        for (String line : text.split("\n")) {
            //Log.d(TAG, "Drawing line: " + line);
            Rect bounds = new Rect();
            textPaint.getTextBounds(line, 0, line.length(), bounds);
            int width = (int) Math.ceil(textPaint.measureText(line));
            int height = bounds.height();
            totalH += (height * 1.5);

            if(singleLineH == 0) {
                singleLineH = height;
            }

            maxWidth = Math.max(maxWidth, width);
        }

        baseYOffset += singleLineH;

        for (String line : text.split("\n")) {
            Rect bounds = new Rect();
            textPaint.getTextBounds(line, 0, line.length(), bounds);
            int width = (int) Math.ceil(textPaint.measureText(line));
            int height = bounds.height();

            StaticLayout.Builder builder = StaticLayout.Builder.obtain(line, 0, line.length(), textPaint, width)
                    .setAlignment(Layout.Alignment.ALIGN_NORMAL)
                    .setLineSpacing(0, 1.0f)
                    .setIncludePad(true);

            float offsetX, offsetY;
            offsetX = (c.getWidth() - maxWidth) / 2.0f;
            offsetY = baseYOffset;

            StaticLayout staticLayout = builder.build();
            c.save();
            c.translate(offsetX, offsetY);
            staticLayout.draw(c);
            c.restore();

            baseYOffset += (height * 1.5);
        }
    }

    private int drawFps(Canvas c, double densityRatio) {
        NumberFormat formatter = new DecimalFormat("#0.00");
        String fpsString = formatter.format(fps.getFps());
        String txt = fpsString + " FPS";

        // Text drawing
        TextPaint textPaint = new TextPaint();
        textPaint.setAntiAlias(true);
        textPaint.setTextSize(status_tv.getTextSize());
        textPaint.setColor(0xFFFFFFFF);

        Rect bounds = new Rect();
        textPaint.getTextBounds(txt, 0, txt.length(), bounds);
        int textWidth = (int) Math.ceil(textPaint.measureText(txt));
        int textHeight = bounds.height();

        int fpsXOffset = c.getWidth() - 10 - textWidth;
        int fpsYOffset = 10;

        StaticLayout.Builder builder = StaticLayout.Builder.obtain(txt, 0, txt.length(), textPaint, textWidth)
                .setAlignment(Layout.Alignment.ALIGN_NORMAL)
                .setLineSpacing(0, 1.0f)
                .setIncludePad(false);

        StaticLayout staticLayout = builder.build();
        c.save();
        c.translate(fpsXOffset, fpsYOffset);
        staticLayout.draw(c);
        c.restore();

        return fpsYOffset + textHeight;
    }

    private void drawProgressBar(Canvas c, double densityRatio, double progress, int baseYOffset) {
        int textH;

        {
            Rect bounds = new Rect();
            TextPaint textPaint = new TextPaint();
            textPaint.setAntiAlias(true);
            textPaint.setTextSize(status_tv.getTextSize());
            textPaint.getTextBounds("AAAAA", 0, "AAAAA".length(), bounds);
            textH = bounds.height();
        }

        int barH = (int) (textH * 2.0);
        int barXOffset = (int) (textH * 1.5);
        int barYOffset = (int) (textH * 0.8) + baseYOffset;

        int barWidth = c.getWidth() - (barXOffset * 2);

        int progressSize = (int) Math.floor(barWidth * progress);

        Paint borderPaint = new Paint();
        borderPaint.setColor(0xFFFFFFFF);
        borderPaint.setStrokeWidth(1);
        borderPaint.setStyle(Paint.Style.STROKE);

        Paint fillPaint = new Paint();
        fillPaint.setColor(0xFFFFFFFF);
        fillPaint.setStrokeWidth(1);
        fillPaint.setStyle(Paint.Style.FILL);

        Rect barRect = new Rect(barXOffset, barYOffset, barXOffset + barWidth, barYOffset + barH);
        Rect fillRect = new Rect(barXOffset, barYOffset, barXOffset + progressSize, barYOffset + barH);

        c.drawRect(barRect, borderPaint);
        c.drawRect(fillRect, fillPaint);
    }

    /*///////////// >>  README  << //////////////////
    That's it for the UI part!

    The next methods/classes implement part of the logic handling the image gathering and training
    procedures.
    ///////////////////////////////*/

    /**
     * Method called when a prediction is completed for an image.
     *
     * This will update the running scores, the FPS counter, the visual confidence shown in the
     * lower fragment, etcetera.
     *
     * This method is invoked from a prediction consumer like {@link PredictionTaskFromJpeg} or
     * {@link PredictionTaskFromYUV}.
     *
     * @param predictions The per-category confidence scores.
     * @param endTimeMs A timestamp describing when the inference process completed (used to update
     *                  the FPS counter).
     */
    public void onPredictionTaskCompleted(float[] predictions, long endTimeMs) {
        if(predictions == null) {
            // Error or aborted
            return;
        }

        for (int i = 0; i < predictionsInTime.length; i++) {
            predictionsInTime[i].addValue(predictions[i]);
            lastPredictions[i] = predictionsInTime[i].getMean();
        }

        int predictedCategory = -1;
        double predictionScore = -1;
        for (int i = 0; i < lastPredictions.length; i++) {
            if (lastPredictions[i] > predictionScore) {
                predictionScore = lastPredictions[i];
                predictedCategory = i;
            }
        }

        @SuppressWarnings("unchecked")
        Pair<Integer, Double>[] rank = new Pair[lastPredictions.length];

        for (int i = 0; i < lastPredictions.length; i++) {
            rank[i] = new Pair<>(i, lastPredictions[i]);
        }

        Arrays.sort(rank, new Comparator<Pair<Integer, Double>>() {
            @Override
            public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
                return Double.compare(o1.second, o2.second);
            }
        });

        previousPredictedCategory = rank[rank.length - 1].first;

        boolean updateView = true;
        if (!INFERENCE_WHILE_TRAINING) {
            updateView = getTrainingPhase() == TrainingPhase.IDLE;
        }

        if (updateView) {
            classSelectionFragment.updatePredictions(rank);

            NumberFormat formatter = new DecimalFormat("#0.00");
            StringBuilder statusBuilder = new StringBuilder();
            for (int i = 0; i < RANKING_SIZE && i < rank.length; i++) {

                String category = application.getCategoryLabel(rank[rank.length - i - 1].first);
                category = category.replace('_', ' ');
                category = category.substring(0, 1).toUpperCase() + category.substring(1);

                double score = Math.round(rank[rank.length - i - 1].second * 100.0 * 100.0) / 100.0;
                statusBuilder
                        .append(category)
                        .append(": ");
                if (score < 0) {
                    statusBuilder.append("< 0.01%");
                } else {
                    statusBuilder.append(formatter.format(score)).append('%');
                }

                if (i != 2) {
                    statusBuilder.append("\n");
                }
            }

            status_tv.setText(statusBuilder.toString());
        }

        fps.updateTime(endTimeMs);
    }

    /**
     * Async task used to run the training step.
     *
     * Once completed, this will call {@link #completeTrainingPhase()}.
     */
    private static class ExecuteTrainingTask extends AsyncTask<Void, Void, Void> {

        private final WeakReference<ClassifyCamera> activityReference;
        private final MyApplication application;
        private final long trainingStartTime;

        ExecuteTrainingTask(ClassifyCamera activity,
                            MyApplication application,
                            long trainingStartTime) {
            this.activityReference = new WeakReference<>(activity);
            this.application = application;
            this.trainingStartTime = trainingStartTime;
        }

        @Override
        protected void onPreExecute() {
            Log.d(TAG, "Starting Training Task");
        }

        @Override
        protected void onPostExecute(Void ignored) {
            long endTime = System.currentTimeMillis();
            long difference = endTime - trainingStartTime;
            Log.d(TAG, "Ended training in " + difference + "ms");

            ClassifyCamera activity = activityReference.get();
            if (activity == null || activity.isFinishing()) {
                return;
            }

            if (!activity.isFinishing()) {
                activity.completeTrainingPhase();
            }
        }

        @Override
        protected Void doInBackground(Void... ignored) {
            ClassifyCamera activity = activityReference.get();
            if (activity == null || activity.isFinishing()) return null;

            boolean proceed = false;
            int trainingLabel = -1;

            try (ResourceLock rl = activity.trainingMutex.lockAsResource()){
                if (activity.trainingPhase == TrainingPhase.EXECUTING_TRAINING) {
                    proceed = true;
                    trainingLabel = activity.trainingCategoryId;
                }
            }

            if (proceed) {
                long startTime = System.currentTimeMillis();

                try {
                    application.cwrTrainingStep(trainingLabel);
                } catch (Throwable e) {
                    Log.e(TAG, "Error during training", e);
                }

                Log.d(TAG, "Ended training. It took " +
                        (System.currentTimeMillis() - startTime) + " ms");
            }

            return null;
        }
    }

    /*///////////// >>  README  << //////////////////
    Finally, the next methods are the boring ones. These methods handle Android-specific stuff
    including the setup procedure for the Camera2 API, permissions, ...
    ///////////////////////////////*/

    /**
     * Callback used to check if the user granted the access to the camera.
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                        ClassifyCamera.this,
                        "You can't use this app without granting permissions",
                        Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    /**
     * Create the menu (the buttons on the upper-right bar of the application).
     */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.my_menu, menu);
        return true;
    }

    /**
     * Method used to handle the menu actions.
     *
     *  As of now, 2 options are available:
     *  - Reset the app to the initial state (revert to the initial model).
     *  - Switch to the image gathering phase.
     */
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {
            case R.id.action_clear_data:
                startResetProcedure();
                return true;

            /*
            // TODO: already implemented. Needs just a little bit of polishing.
            case R.id.action_gather_images:
                Toast toast;
                boolean isIdle = getTrainingPhase() == TrainingPhase.IDLE;

                if (!isIdle) {
                    toast = Toast.makeText(this, "Can't change while training", Toast.LENGTH_SHORT);
                } else {
                    onlyImageGather = !onlyImageGather;
                    trainingSaveImageTasksExecutor.shutdown();

                    if (onlyImageGather) {
                        maxConcurrentSaveTasks = 1;
                        toast = Toast.makeText(this, "Image gathering activated", Toast.LENGTH_SHORT);
                    } else {
                        maxConcurrentSaveTasks = Runtime.getRuntime().availableProcessors();
                        toast = Toast.makeText(this, "Image gathering deactivated", Toast.LENGTH_SHORT);
                    }

                    trainingSaveImageTasksExecutor = Executors.newFixedThreadPool(maxConcurrentSaveTasks);
                }
                toast.show();

                return true;*/

            /*
            case R.id.action_about:  // TODO: implement
                return true;
            */

            default:
                return super.onOptionsItemSelected(item);

        }
    }

    /**
     * Starts the thread that will receive the images using the Camera2 API.
     */
    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /**
     * Stops the thread that receiving the images.
     */
    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Manages the app resume process. This will start the background thread used to gather images.
     *
     * For non-Android experts: this is always called after {@link #onCreate(Bundle)}.
     */
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    /**
     * Manages the app pause process. This will stop the background thread used to gather images.
     * This will also abort the countdown phase if in progress.
     */
    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();

        try (ResourceLock rl = trainingMutex.lockAsResource()){
            if(trainingPhase == TrainingPhase.COUNTDOWN) {
                if (countDown != null) {
                    countDown.cancel();
                    countDown = null;
                }
                trainingPhase = TrainingPhase.IDLE;
            }
        }
        super.onPause();
    }

    /**
     * Used to setup the Camera2 image acquisition process.
     *
     * This method will pick the best camera resolution, obtain the orientation, check permissions,
     * and finally call the camera manager to access the camera.
     *
     * The image listener is set up later in {@link #createCameraPreview()}.
     */
    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {

            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;

            imageRotation = getCameraOrientation();

            Log.d(TAG, "Image rotation = " + imageRotation);

            imageDimension = getBestSizeForTask(map.getOutputSizes(SurfaceTexture.class),
                    CAPTURE_W, CAPTURE_H);
            Log.d(getPackageName(), "Image dimension = " + imageDimension);
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(ClassifyCamera.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }

            int fitViewW = (imageRotation == 90 || imageRotation == -90) ? imageDimension.getHeight() : imageDimension.getWidth();
            int fitViewH = (imageRotation == 90 || imageRotation == -90) ? imageDimension.getWidth() : imageDimension.getHeight();
            if (textureView instanceof AutoFitTextureView) {
                Log.d(TAG, "Using AutoFitTextureView");
                ((AutoFitTextureView) textureView).setAspectRatio(fitViewW, fitViewH);
            }

            if (shadowView instanceof AutoFitSurfaceView) {
                Log.d(TAG, "Using AutoFitSurfaceView");
                ((AutoFitSurfaceView) shadowView).setAspectRatio(fitViewW, fitViewH);
            }

            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * Used to select the best camera resolution among the available ones.
     */
    private Size getBestSizeForTask(Size[] availableImageSizes, int preferredW, int preferredH) {
        boolean rotated = imageRotation == 90 || imageRotation == -90;

        for (int i = availableImageSizes.length - 1; i >= 0; i--) {
            int realW = rotated ? availableImageSizes[i].getHeight() : availableImageSizes[i].getWidth();
            int realH = rotated ? availableImageSizes[i].getWidth() : availableImageSizes[i].getHeight();

            double ratio = (double) realW / realH;
            //Log.d(TAG, "Real: " + realW + "; " + realH + " -> " + ratio);

            if (realW >= preferredW && realH >= preferredH &&
                    ratio >= MIN_ASPECT_RATIO && ratio <= MAX_ASPECT_RATIO) {
                return availableImageSizes[i];
            }
        }

        return availableImageSizes[0];
    }

    protected void updatePreview() {
        if (null == cameraDevice) {
            return;
        }

        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    //https://stackoverflow.com/questions/38282076/how-to-save-images-with-correct-orientation-using-camera2
    //https://medium.com/@kenodoggy/solving-image-rotation-on-android-using-camera2-api-7b3ed3518ab6
    private int getJpegOrientation(CameraCharacteristics c,
                                   int deviceOrientation) {
        if (deviceOrientation == android.view.OrientationEventListener.ORIENTATION_UNKNOWN)
            return 0;
        Integer sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION);
        assert sensorOrientation != null;

        // Round device orientation to a multiple of 90
        deviceOrientation = (deviceOrientation + 45) / 90 * 90;

        // Reverse device orientation for front-facing cameras
        Integer lensFacing = c.get(CameraCharacteristics.LENS_FACING);
        assert lensFacing != null;
        boolean facingFront = lensFacing == CameraCharacteristics.LENS_FACING_FRONT;
        if (facingFront) deviceOrientation = -deviceOrientation;

        // Calculate desired JPEG orientation relative to camera orientation to make
        // the lastCameraImage upright relative to the device orientation
        return -(sensorOrientation + deviceOrientation + 360) % 360;
    }

    public void updateProcessingTimeData(long predictTime) {
        processingTime.addValue(predictTime);
    }

    private int getCameraOrientation() throws CameraAccessException {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
        int deviceRotation = getWindowManager().getDefaultDisplay().getRotation();
        return getJpegOrientation(characteristics, deviceRotation);
    }

    /**
     * Reset procedure used to revert the app to the factory state.
     *
     * Simply put, this will remove the folder containing all the current data (models, thumbnails,
     * replay buffer) and then close the app.
     */
    private void startResetProcedure() {
        final MyApplication.ResetProcedureListener resetListener =
                new MyApplication.ResetProcedureListener() {
                    @Override
                    public void onResetCompleted() {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                new AlertDialog.Builder(ClassifyCamera.this)
                                        .setMessage("Reset completed. The app will now close...")
                                        .setCancelable(false)
                                        .setPositiveButton(android.R.string.ok, null)
                                        .setOnDismissListener(new DialogInterface.OnDismissListener() {
                                            @Override
                                            public void onDismiss(DialogInterface dialog) {
                                                finish();
                                                android.os.Process.sendSignal(android.os.Process.myPid(),
                                                        android.os.Process.SIGNAL_KILL);
                                            }
                                        })
                                        .show();
                            }
                        });
                    }
                };

        DialogInterface.OnClickListener resetConfirmListener = new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which) {
                    case DialogInterface.BUTTON_POSITIVE:
                        // Reset the app
                        application.resetAppState(resetListener);
                        break;

                    case DialogInterface.BUTTON_NEGATIVE:
                        break;
                }
            }
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("Do you really want to clear application data? This will destroy the current model!")
                .setPositiveButton("Yes", resetConfirmListener)
                .setNegativeButton("No", resetConfirmListener).show();
    }

    public enum TrainingPhase {
        IDLE, COUNTDOWN, GATHERING_IMAGES, EXECUTING_TRAINING
    }

    public enum TaskExecutors {
        INFERENCE, TRAINING, SAVE_TRAINING_IMAGES, SAVE_THUMBNAIL, OTHER
    }

    public TrainingPhase getTrainingPhase() {
        try (ResourceLock rl = trainingMutex.lockAsResource()){
            return trainingPhase;
        }
    }

    public void removeImageConsumer(ImageConsumer consumer) {
        this.imageConsumers.remove(consumer);
    }

    public MyApplication getMyApplication() {
        return ((MyApplication) getApplication());
    }

    // TODO: unify executors
    @SafeVarargs
    public final <X> void executeConsumerTask(TaskExecutors executor, AsyncTask<X, ?, ?> task, X... parameters) {
        if(executor == null) {
            throw new IllegalArgumentException("Executor can't be null. You can use OTHER instead");
        }
        Executor chosenOne = null;

        switch (executor) {
            case INFERENCE:
            case SAVE_TRAINING_IMAGES:
                chosenOne = trainingSaveImageTasksExecutor;
                break;
            case TRAINING:
            case SAVE_THUMBNAIL:
            case OTHER:
                chosenOne = AsyncTask.THREAD_POOL_EXECUTOR;
                break;
        }

        task.executeOnExecutor(chosenOne, parameters);
    }
}