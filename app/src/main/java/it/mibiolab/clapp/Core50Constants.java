package it.mibiolab.clapp;

/**
 * A class containing resource IDs, category names, and other constants.
 */
public final class Core50Constants {

    public static final String[] CORE50_CATEGORIES = {
            "plug_adapter",
            "mobile_phone",
            "scissor",
            "light_bulb",
            "can",
            "glass",
            "ball",
            "marker",
            "cup",
            "remote_control"
    };

    public static final float[] CORE50_MEAN = {104, 117, 123};
    public static final int[] CORE50_IMAGES_SIZE = {128, 128};
    public static final int CAFFEMODEL_ID = R.raw.weights;
    public static final int NET_PROTOTXT_ID = R.raw.mobilenetv1dw10_deploy_phases;
    public static final int SOLVER_PROTOTXT_ID = R.raw.solver;
    public static final int LABELS_ID = R.raw.core50_labels_cat;
    public static final int INITIAL_REHE_ID = R.raw.initial_rehe_data;

    public static final String CAFFEMODEL_TARGET_NAME = "weights.caffemodel";
    public static final String PROTOTXT_TARGET_NAME = "mobilenetv1dw_net.prototxt";
    public static final String LABELS_TARGET_NAME = "core50_labels_cat.txt";
    public static final String SOLVER_TARGET_NAME = "solver.prototxt";
    public static final String INITIAL_REHE_TARGET_NAME = "initial_rehe_data.dat";

    public static final int[] INITIAL_CORE50_CATEGORIES_THUMBNAILS_RAW = {
            R.raw.core50_cat0_thumb_fore,
            R.raw.core50_cat1_thumb_fore,
            R.raw.core50_cat2_thumb_fore,
            R.raw.core50_cat3_thumb_fore,
            R.raw.core50_cat4_thumb_fore,
            R.raw.core50_cat5_thumb_fore,
            R.raw.core50_cat6_thumb_fore,
            R.raw.core50_cat7_thumb_fore,
            R.raw.core50_cat8_thumb_fore,
            R.raw.core50_cat9_thumb_fore
    };

    public static final int CURRENT_LICENSE_VERSION = 2;

    private Core50Constants() {}
}
