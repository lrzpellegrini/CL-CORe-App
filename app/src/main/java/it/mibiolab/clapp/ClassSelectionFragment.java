package it.mibiolab.clapp;

import android.content.Context;
import android.content.DialogInterface;
import android.content.res.TypedArray;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.text.InputType;
import android.util.Log;
import android.util.Pair;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TableLayout;
import android.widget.TableRow;

import java.util.LinkedList;
import java.util.List;

/**
 * The class selection fragment.
 *
 * This is the fragment in charge of managing the lower part of the UI.
 * This includes showing the thumbnails, the visual feedback for the class confidence, and
 * the prompts for the new category name.
 *
 * This fragment is used in the main activity {@link ClassifyCamera}.
 */
public class ClassSelectionFragment extends Fragment {

    private static final String TAG = "ClassSelectionFragment";

    private static final int CATEGORIES_PER_ROW = 5;
    private static final int RANKING_SIZE = 3;

    private static final int CLASSIFICATION_RANK_GRADIENTS_MIN = 3; // -1 = draw all colors
    private static final int CLASSIFICATION_RANK_GRADIENTS_ALPHA = 0xCF;
    private static final int[] CLASSIFICATION_RANK_GRADIENTS = {
            0xE92C00,
            0xE94F00,
            0xE97100,
            0xE99300,
            0xE9B600,
            0xEAD800,
            0xD9EA00,
            0xB6EA00,
            0x94EA00,
            0x71EB00
    };

    static {
        for (int i = 0; i < CLASSIFICATION_RANK_GRADIENTS.length; i++) {
            CLASSIFICATION_RANK_GRADIENTS[i] = (CLASSIFICATION_RANK_GRADIENTS_ALPHA << 24) | CLASSIFICATION_RANK_GRADIENTS[i];
        }
    }

    private OnFragmentInteractionListener mListener;
    private MyApplication application;

    private View[] categoriesButtons = null;
    private View[] emptyCategoriesButtons = null;
    private List<TableRow> tableRowsList = null;
    private TableLayout classesTable = null;

    private List<View> clickViewReceivers = null;
    private View[] thumbnailButtonsDecorations = null;

    private boolean isSelectionEnabled = true;

    public ClassSelectionFragment() {
        // Required empty public constructor
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @return A new instance of fragment ClassSelectionFragment.
     */
    public static ClassSelectionFragment newInstance() {
        ClassSelectionFragment fragment = new ClassSelectionFragment();
        Bundle args = new Bundle();
        // No parameters!
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // No parameters!

        this.application = ((MyApplication) getActivity().getApplication());
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_class_selection, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        createTableButtons();
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        if (context instanceof OnFragmentInteractionListener) {
            mListener = (OnFragmentInteractionListener) context;
        } else {
            throw new RuntimeException(context.toString()
                    + " must implement OnFragmentInteractionListener");
        }
    }

    @Override
    public void onDetach() {
        super.onDetach();
        mListener = null;
    }

    public void clearPredictions() {
        for (int i = 0; i < categoriesButtons.length; i++) {
            categoriesButtons[i].setBackgroundColor(Color.TRANSPARENT);
        }
    }

    public void recreateButtons() {
        createTableButtons();
    }

    public void enableSelection(boolean enabled) {
        this.isSelectionEnabled = enabled;
        applyEnableEffect();
    }

    public void updatePredictions(Pair<Integer, Double>[] rank) {
        for (int i = 0; i < rank.length; i++) {
            Log.v(TAG, "Prediction " + i + ": " + rank[i].first + " -> " + rank[i].second);
            int colorIndex = (int) Math.max(Math.min(Math.floor(rank[i].second * CLASSIFICATION_RANK_GRADIENTS.length), CLASSIFICATION_RANK_GRADIENTS.length-1), 0);
            Log.v(TAG, "Color index: " + colorIndex);

            if(CLASSIFICATION_RANK_GRADIENTS_MIN < 0 || colorIndex >= CLASSIFICATION_RANK_GRADIENTS_MIN) {
                int color = CLASSIFICATION_RANK_GRADIENTS[colorIndex];
                categoriesButtons[rank[i].first].setBackgroundColor(color);
            } else {
                categoriesButtons[rank[i].first].setBackgroundColor(Color.TRANSPARENT);
            }
        }
    }

    private void createTableButtons() {
        if(classesTable == null) {
            classesTable = getView().findViewById(R.id.classes_table);

            //https://stackoverflow.com/questions/3327599/get-all-tablerows-in-a-tablelayout
            tableRowsList = new LinkedList<>();
            for (int i = 0, j = classesTable.getChildCount(); i < j; i++) {
                View view = classesTable.getChildAt(i);
                if (view instanceof TableRow) {
                    tableRowsList.add((TableRow) view);
                }
            }
        }

        for (TableRow tableRow : tableRowsList) {
            tableRow.removeAllViews();
        }

        int categoriesCount = application.getCategoriesCount();
        categoriesButtons = new View[categoriesCount];
        thumbnailButtonsDecorations = new View[categoriesCount];
        emptyCategoriesButtons = new View[application.cwrGetMaxCategories() - categoriesCount];
        clickViewReceivers = new LinkedList<>();
        int categoryIndex = 0;
        int emptyCategoryIndex = 0;

        //https://stackoverflow.com/questions/19894581/tablelayout-rows-textview-margin
        TableRow.LayoutParams tableRowParams = new TableRow.LayoutParams();
        //tableRowParams.setMargins(1, 1, 1, 1);
        tableRowParams.weight = 1;

        for (TableRow tableRow : tableRowsList) {
            int categoriesInThisRow = 0;
            for (; categoriesInThisRow < CATEGORIES_PER_ROW && categoryIndex < categoriesButtons.length;
                 categoriesInThisRow++, categoryIndex++) {
                final int categoryId = categoryIndex;

                View buttonWrapper = getLayoutInflater().inflate(R.layout.rounded_button_layout, null);
                ImageButton thumbnailButton = buttonWrapper.findViewById(R.id.thumbnail);
                thumbnailButton.setImageBitmap(application.getCategoryThumbnail(categoryIndex));
                categoriesButtons[categoryIndex] = buttonWrapper;
                thumbnailButtonsDecorations[categoryIndex] = buttonWrapper.findViewById(R.id.thumbnail_decoration);
                tableRow.addView(buttonWrapper, tableRowParams);

                View clickReceiver = buttonWrapper.findViewById(R.id.thumbnail_wrapper);

                View.OnClickListener listener = new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if(isSelectionEnabled) {
                            onExistingClassSelection(categoryId);
                        }
                    }
                };

                clickReceiver.setOnClickListener(listener);
                clickViewReceivers.add(clickReceiver);
            }

            for (; categoriesInThisRow < CATEGORIES_PER_ROW && emptyCategoryIndex < emptyCategoriesButtons.length;
                 categoriesInThisRow++, emptyCategoryIndex++) {

                final int categoryId = categoryIndex + emptyCategoryIndex;

                View buttonWrapper = getLayoutInflater().inflate(R.layout.empty_rounded_button_layout, null);
                emptyCategoriesButtons[emptyCategoryIndex] = buttonWrapper;
                tableRow.addView(buttonWrapper, tableRowParams);

                View clickReceiver = buttonWrapper.findViewById(R.id.thumbnail_wrapper);

                View.OnClickListener listener = new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if(isSelectionEnabled) {
                            onNewClassSelection(categoryId);
                        }
                    }
                };

                clickReceiver.setOnClickListener(listener);
                clickViewReceivers.add(clickReceiver);
            }
        }

        if (categoryIndex < categoriesButtons.length) {
            Log.e(TAG, "Insufficient rows for " + categoriesButtons.length + " categories");
        }

        applyEnableEffect();
    }

    private void applyEnableEffect() {
        if (isSelectionEnabled) {
            if(clickViewReceivers != null) {
                for (View clickViewReceiver : clickViewReceivers) {
                    // https://stackoverflow.com/a/27474621
                    clickViewReceiver.setForeground(getSelectedItemDrawable());
                    clickViewReceiver.setClickable(true);
                }

                for (View thumbnailButtonsDecoration : thumbnailButtonsDecorations) {
                    thumbnailButtonsDecoration.setBackgroundResource(R.drawable.button_bg_small_plus);
                }
            }
        } else {
            if(clickViewReceivers != null) {
                for (View clickViewReceiver : clickViewReceivers) {
                    clickViewReceiver.setForeground(
                            new ColorDrawable(ContextCompat.getColor(getActivity(), android.R.color.transparent)));
                    clickViewReceiver.setClickable(false);
                }

                for (View thumbnailButtonsDecoration : thumbnailButtonsDecorations) {
                    thumbnailButtonsDecoration.setBackgroundResource(R.drawable.button_bg_small_noplus);
                }
            }
        }
    }

    private Drawable getSelectedItemDrawable() {
        int[] attrs = new int[]{R.attr.selectableItemBackground};
        TypedArray ta = getActivity().obtainStyledAttributes(attrs);
        Drawable selectedItemDrawable = ta.getDrawable(0);
        ta.recycle();
        return selectedItemDrawable;
    }

    private void onExistingClassSelection(final int labelId) {
        DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which){
                    case DialogInterface.BUTTON_POSITIVE:
                        DialogInterface.OnClickListener thumbnailReplaceListener = new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                switch (which){
                                    case DialogInterface.BUTTON_POSITIVE:
                                        confirmSelection(labelId, null, false, true);
                                        break;

                                    case DialogInterface.BUTTON_NEGATIVE:
                                        confirmSelection(labelId, null, false, false);
                                        break;
                                }
                            }
                        };

                        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
                        builder.setMessage(R.string.replace_thumbnail_message)
                                .setPositiveButton(R.string.yes, thumbnailReplaceListener)
                                .setNegativeButton(R.string.no, thumbnailReplaceListener).show();
                        break;

                    case DialogInterface.BUTTON_NEGATIVE:
                        break;
                }
            }
        };

        String catName = application.getCategoryLabel(labelId);
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());

        builder.setMessage(getString(R.string.improve_cat_message, catName))
                .setPositiveButton(R.string.yes, dialogClickListener)
                .setNegativeButton(R.string.no, dialogClickListener).show();
    }

    private void onNewClassSelection(final int labelId) {
        // Label ID can be ignored
        DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which){
                    case DialogInterface.BUTTON_POSITIVE:
                        startCreateNewCategoryPhase();
                        break;

                    case DialogInterface.BUTTON_NEGATIVE:
                        break;
                }
            }
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setMessage(R.string.new_cat_confirm_message)
                .setPositiveButton(R.string.yes, dialogClickListener)
                .setNegativeButton(R.string.no, dialogClickListener).show();
    }

    private void startCreateNewCategoryPhase() {
        // https://stackoverflow.com/a/10904665

        final EditText input = new EditText(getActivity());
        input.setInputType(InputType.TYPE_CLASS_TEXT);

        DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                switch (which){
                    case DialogInterface.BUTTON_POSITIVE:
                        String newCategoryName = input.getText().toString();
                        newCategoryName = newCategoryName.trim();
                        if(newCategoryName.isEmpty()) {
                            new Handler(Looper.getMainLooper()).post(new Runnable() {
                                public void run() {
                                    showInfoAlert(getString(R.string.invalid_cat_name_title),
                                            getString(R.string.invalid_cat_name_message),
                                            new DialogInterface.OnClickListener() {
                                                public void onClick(DialogInterface dialog, int which) {
                                                    dialog.dismiss();
                                                    startCreateNewCategoryPhase();
                                                }
                                            });
                                }
                            });
                        } else {
                            int newCategoryId = application.cwrAddNewCategory(newCategoryName);
                            confirmSelection(newCategoryId, newCategoryName, true, true);
                        }
                        break;

                    case DialogInterface.BUTTON_NEGATIVE:
                        break;
                }
            }
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setTitle(R.string.insert_new_cat_name_title);
        builder.setView(input);

        builder.setPositiveButton(android.R.string.yes, dialogClickListener)
                .setNegativeButton(android.R.string.no, dialogClickListener).show();
        input.requestFocus();
        InputMethodManager imm = (InputMethodManager) getActivity().getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.toggleSoftInput(InputMethodManager.SHOW_FORCED, 0);
    }

    private void confirmSelection(int categoryId, String categoryName, boolean isNew, boolean replaceThumbnail) {
        if (mListener != null) {
            mListener.onCategorySelected(categoryId, categoryName, isNew, replaceThumbnail);
        }
    }

    private void showInfoAlert(String title, String message, DialogInterface.OnClickListener listener) {
        AlertDialog alertDialog = new AlertDialog.Builder(getActivity()).create();
        alertDialog.setTitle(title);
        alertDialog.setMessage(message);
        alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, getString(android.R.string.ok), listener);

        alertDialog.show();
    }

    public interface OnFragmentInteractionListener {
        void onCategorySelected(int categoryIndex, String categoryName, boolean isNewCategory, boolean replaceThumbnail);
    }
}
