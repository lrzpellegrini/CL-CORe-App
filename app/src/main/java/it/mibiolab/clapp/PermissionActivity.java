package it.mibiolab.clapp;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.commons.io.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static android.text.Layout.JUSTIFICATION_MODE_INTER_WORD;
import static it.mibiolab.clapp.Core50Constants.CURRENT_LICENSE_VERSION;

/**
 * This activity is used to manage the application startup by copying the needed resources
 * in the internal memory. This includes the initial pre-trained model, the initial replay
 * memory, the thumbnails, etcetera.
 *
 * This will also manage the initial license acceptance.
 */
public class PermissionActivity extends AppCompatActivity {
    private static final String TAG = "PermissionActivity";

    private static final int TRAIN_THREADS = 4;
    private static final int FEATURE_EXTRACTION_THREADS = TRAIN_THREADS;
    private static final boolean RESOURCE_CHECK_BY_FILENAME = true;
    private static final int REQUEST_NEEDED_PERMISSIONS_CODE = 10;

    private boolean assetsCopied = false;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ActivityManager am = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        am.getMemoryInfo(mi);
        long avmem = mi.availMem / 1048576L;
        Log.d(TAG, "onCreate GetMemoryClass => " + am.getMemoryClass() );
        Log.d(TAG, "onCreate getLargeMemoryClass => " + am.getLargeMemoryClass() );
        Log.d(TAG, "onCreate avmem => " + avmem + " MB");

        setContentView(R.layout.permission_activity);

        File pathLicenseSDCard = new File(Environment.getExternalStorageDirectory() +
                "/CaffeAndroid/" + "accepted_licenses.txt");

        if(shouldPromptLicense()) {
            // First, before copying anything to the internal memory, prompt for license
            TextView licenseContainer = findViewById(R.id.license_container);
            String licenseText = readRawTextFile(this, R.raw.licenses);
            licenseContainer.setText(licenseText);

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                licenseContainer.setJustificationMode(JUSTIFICATION_MODE_INTER_WORD);
            }

            findViewById(R.id.license_ok_btn).setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (requestAppPermissions()) {
                        onPermissionsGranted();
                    }
                }
            });
        } else {
            // License already accepted
            if (requestAppPermissions()) {
                onPermissionsGranted();
            }
        }
    }

    //Credits: https://stackoverflow.com/questions/48172519/oreo-write-external-storage-permission
    public boolean requestAppPermissions() {
        if (android.os.Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            return true;
        }

        if (hasReadPermissions() && hasWritePermissions() && hasCameraPermissions()) {
            return true;
        }

        ActivityCompat.requestPermissions(this,
                new String[]{
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE,
                        Manifest.permission.CAMERA
                }, REQUEST_NEEDED_PERMISSIONS_CODE);
        return false;
    }

    public boolean hasReadPermissions() {
        return (ContextCompat.checkSelfPermission(getBaseContext(),
                Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED);
    }

    public boolean hasWritePermissions() {
        return (ContextCompat.checkSelfPermission(getBaseContext(),
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED);
    }

    public boolean hasCameraPermissions() {
        return (ContextCompat.checkSelfPermission(getBaseContext(),
                Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String permissions[],
                                           @NonNull int[] grantResults) {
        switch (requestCode) {
            case REQUEST_NEEDED_PERMISSIONS_CODE: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    this.onPermissionsGranted();
                } else {
                    this.onPermissionsDenied();
                }
            }
        }
    }

    /**
     This method is in charge of the actual copy-to-storage procedure.
     */
    private void copyAssets() throws IOException {

        if(assetsCopied) {
            return;
        }

        assetsCopied = true;
        MyApplication application = ((MyApplication) getApplication());

        String solverProto, netProto, netWeights, labels, pattern, initial_rehe;
        String meanImage = null;
        float[] meanValues = null;

        copyFiletoExternalStorage(R.raw.licenses,
                "accepted_licenses.txt");

        updateLicenseVersion();

        solverProto = copyFiletoExternalStorage(Core50Constants.SOLVER_PROTOTXT_ID,
                Core50Constants.SOLVER_TARGET_NAME);
        netProto = copyFiletoExternalStorage(Core50Constants.NET_PROTOTXT_ID,
                Core50Constants.PROTOTXT_TARGET_NAME);
        netWeights = copyFiletoExternalStorage(Core50Constants.CAFFEMODEL_ID,
                Core50Constants.CAFFEMODEL_TARGET_NAME);
        meanValues = Core50Constants.CORE50_MEAN;
        labels = copyFiletoExternalStorage(Core50Constants.LABELS_ID,
                Core50Constants.LABELS_TARGET_NAME);
        initial_rehe = copyFiletoExternalStorage(Core50Constants.INITIAL_REHE_ID,
                Core50Constants.INITIAL_REHE_TARGET_NAME);
        pattern = copyFiletoExternalStorage(R.raw.core50_example_pattern,
                "pattern_core_50.png");

        List<String> labelsList = new LinkedList<>();
        if(application.cwrPreviousStateExists(application.getStatePath())) {
            Log.i(TAG, "Previous state exists");
            application.cwrReloadApp(
                    application.getStatePath(),
                    TRAIN_THREADS,
                    FEATURE_EXTRACTION_THREADS);
            Log.d(TAG, "Previous state loaded");
            String[] labelsArray = application.cwrGetCurrentLabels();
            labelsList.addAll(Arrays.asList(labelsArray));

            Log.d(TAG, "Loaded " + labelsList.size() + " labels");
            // TODO: others...
        } else {
            Log.d(TAG, "Previous state doesn't exist... loading defaults");
            try (FileInputStream is = new FileInputStream(new File(labels));
                 BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
                String line = reader.readLine();
                while (line != null) {
                    labelsList.add(line);
                    line = reader.readLine();
                }
            }
            String[] labelsArray = new String[labelsList.size()];
            labelsList.toArray(labelsArray);

            // TODO: generalize (initialization parameters from static fields)
            /*
            Here we initialize the CWR application for the first time. The app is initialized
            with 10 initial classes and replay at the last layer ("pool" variant).
             */
            float[] class_updates = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
            application.cwrInitApp(solverProto, netWeights, class_updates,
                    8, 10, new String[]{"mid_fc7"},
                    "softmax", "pool6", labelsArray,
                    Core50Constants.CORE50_MEAN[0],
                    Core50Constants.CORE50_MEAN[1],
                    Core50Constants.CORE50_MEAN[2],
                    TRAIN_THREADS,
                    FEATURE_EXTRACTION_THREADS,
                    "pool6",
                    initial_rehe);

            application.cwrSaveAppState(application.getStatePath());
        }

        copyThumbnails();

        List<String> thumbnails = loadThumbnailsList(labelsList.size());

        application.loadInitialCategories(labelsList, thumbnails);

        File trainingImagesPathFile = new File(application.getTrainingImagesPath());
        if(trainingImagesPathFile.exists()) {
            FileUtils.deleteDirectory(trainingImagesPathFile);
        }
        trainingImagesPathFile.mkdirs();
        application.getTrainingImagesZipDir().mkdirs();
    }

    private String copyFiletoExternalStorage(int resourceId, String resourceName) throws IOException {
        File pathSDCard = new File(Environment.getExternalStorageDirectory() +
                "/CaffeAndroid/" + resourceName);
        Log.v("CopyFilesToSDCard", resourceName + " -> " + pathSDCard.getAbsolutePath());

        if (!RESOURCE_CHECK_BY_FILENAME || !pathSDCard.exists()) {
            pathSDCard.getParentFile().mkdirs();
            try (InputStream fileIn = getResources().openRawResource(resourceId);
                 FileOutputStream fileOut = new FileOutputStream(pathSDCard)) {

                byte[] buff = new byte[4 * 1024 * 1024];
                int read;
                while ((read = fileIn.read(buff)) > 0) {
                    fileOut.write(buff, 0, read);
                }
            }
        }

        return pathSDCard.getPath();
    }

    private void onPermissionsGranted() {
        try {
            copyAssets();

            Intent intent = new Intent(this, ClassifyCamera.class);
            //intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
            finishAffinity();
            startActivity(intent);
        } catch (IOException e) {
            Log.e(getPackageName(), "Error copying assets", e);

            Toast.makeText(this, "Error copying assets", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void onPermissionsDenied() {
        Log.e(getPackageName(), "Access denied (external storage and camera)");
        finish();
    }

    private List<String> copyThumbnails() throws IOException {
        List<String> result = new LinkedList<>();
        int categoryIndex = 0;
        for (int core50CategoriesThumbnail : Core50Constants.INITIAL_CORE50_CATEGORIES_THUMBNAILS_RAW) {
            result.add(copyFiletoExternalStorage(core50CategoriesThumbnail, "cat" + categoryIndex + "_thumbnail.png")); // TODO: generalize
            categoryIndex++;
        }

        return result;
    }

    private List<String> loadThumbnailsList(int expected) throws IllegalStateException {
        List<String> result = new LinkedList<>();
        for (int i = 0; i < expected; i++) {
            File pathSDCard = new File(((MyApplication) getApplication()).getDefaultThumbnailPath(i));
            if(!pathSDCard.exists()) {
                throw new IllegalStateException("Not enough thumbnails. File not found: " + pathSDCard.getAbsolutePath());
            }

            result.add(pathSDCard.getAbsolutePath());
        }

        return result;
    }

    private void updateLicenseVersion() {
        SharedPreferences sharedPref = getSharedPreferences(
                getString(R.string.preference_license), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPref.edit();
        editor.putInt(getString(R.string.preference_license_version), CURRENT_LICENSE_VERSION);
        editor.apply();
    }

    private boolean shouldPromptLicense() {
        SharedPreferences sharedPref = getSharedPreferences(
                getString(R.string.preference_license), Context.MODE_PRIVATE);
        int licenseVersion = sharedPref.getInt(getString(R.string.preference_license_version), -1);
        if(licenseVersion < CURRENT_LICENSE_VERSION) {
            return true;
        }

        return false;
    }

    public static String readRawTextFile(Context ctx, int resId) {
        InputStream inputStream = ctx.getResources().openRawResource(resId);

        InputStreamReader inputreader = new InputStreamReader(inputStream);
        BufferedReader buffreader = new BufferedReader(inputreader);
        String line;
        StringBuilder text = new StringBuilder();

        try {
            while (( line = buffreader.readLine()) != null) {
                text.append(line);
                text.append('\n');
            }
        } catch (IOException e) {
            return null;
        }
        return text.toString();
    }
}
