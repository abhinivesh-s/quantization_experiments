import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer # For dummy data generation
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm # tqdm for progress bars
import pandas as pd # For better display of consolidated results

# --- 0. (Optional) Generate Dummy Data if you don't have yours ready ---
def generate_dummy_data(n_samples=1000, n_features=2000, n_classes=4): # Increased for more realistic scenario
    texts = ["sample text " + "word" + str(i % 500) + " entity" + str(i%100) + " class" + str(j % n_classes) for i, j in enumerate(np.random.randint(0, 1000, size=(n_samples, 2)))]
    labels_text = ["class_" + str(i % n_classes) for i in np.random.randint(0, n_classes, n_samples)]
    le = LabelEncoder()
    labels = le.fit_transform(labels_text)
    train_size = int(n_samples * 0.6)
    val_size = int(n_samples * 0.2)
    X_text_train, X_text_val, X_text_holdout = np.split(texts, [train_size, train_size + val_size])
    y_train, y_val, y_holdout = np.split(labels, [train_size, train_size + val_size])
    tfidf_vectorizer = TfidfVectorizer(max_features=n_features, min_df=2, ngram_range=(1,1))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_text_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_text_val)
    X_holdout_tfidf = tfidf_vectorizer.transform(X_text_holdout)
    print(f"Dummy data shapes: X_train_tfidf: {X_train_tfidf.shape}, y_train: {y_train.shape}")
    print(f"Dummy data shapes: X_val_tfidf: {X_val_tfidf.shape}, y_val: {y_val.shape}")
    print(f"Dummy data shapes: X_holdout_tfidf: {X_holdout_tfidf.shape}, y_holdout: {y_holdout.shape}")
    print(f"Classes: {le.classes_}")
    return X_train_tfidf, y_train, X_val_tfidf, y_val, X_holdout_tfidf, y_holdout, le.classes_

# Replace with your actual data loading and TF-IDF transformation
X_train_tfidf, y_train, X_val_tfidf, y_val, X_holdout_tfidf, y_holdout, class_names = generate_dummy_data()
num_classes = len(class_names)

# --- 1. Define Comprehensive Hyperparameter Grid for LinearSVC ---
param_grid_linearsvc = []
C_values = [0.001, 0.01, 0.1, 1, 10, 100]  # Logarithmic scale for C
max_iter_values = [1000, 2000, 5000, 10000] # Range for max_iter
tol_value = 1e-4 # Standard tolerance, could also be tuned
class_weight_options = [None, 'balanced'] # Option for class weighting

# L2 Regularization
for c_val in C_values:
    for mi_val in max_iter_values:
        for cw in class_weight_options:
            # L2 with hinge loss (requires dual=True)
            param_grid_linearsvc.append({
                'C': c_val,
                'penalty': 'l2',
                'loss': 'hinge',
                'dual': True,
                'max_iter': mi_val,
                'tol': tol_value,
                'class_weight': cw,
                'random_state': 42
            })
            # L2 with squared_hinge loss (prefers dual=False for n_samples > n_features)
            # For text data, n_features can be > n_samples, so dual=True might be faster.
            # We will test both dual options where applicable.
            param_grid_linearsvc.append({
                'C': c_val,
                'penalty': 'l2',
                'loss': 'squared_hinge',
                'dual': False,
                'max_iter': mi_val,
                'tol': tol_value,
                'class_weight': cw,
                'random_state': 42
            })
            # Test L2 squared_hinge with dual=True as well (might be better if n_features > n_samples)
            # This combination is valid.
            param_grid_linearsvc.append({
                'C': c_val,
                'penalty': 'l2',
                'loss': 'squared_hinge',
                'dual': True,
                'max_iter': mi_val,
                'tol': tol_value,
                'class_weight': cw,
                'random_state': 42
            })


# L1 Regularization (only works with squared_hinge loss and dual=False for LinearSVC's liblinear solver)
for c_val in C_values:
    for mi_val in max_iter_values:
        for cw in class_weight_options:
            param_grid_linearsvc.append({
                'C': c_val,
                'penalty': 'l1',
                'loss': 'squared_hinge',
                'dual': False, # L1 with dual=True is not supported by liblinear
                'max_iter': mi_val,
                'tol': tol_value,
                'class_weight': cw,
                'random_state': 42
            })

print(f"Number of LinearSVC parameter combinations to test: {len(param_grid_linearsvc)}\n")

# --- 2. Iterate, Train, and Evaluate LinearSVC ---
results_svc_all = [] # To store all LinearSVC results

print("--- Stage 1: Tuning LinearSVC ---")
# tqdm for the outer loop of LinearSVC hyperparameter tuning
for params in tqdm(param_grid_linearsvc, desc="LinearSVC Hyperparameters"):
    current_run_warnings = []
    # Sanitize params for printing (e.g., shorten long lists if any)
    # For LinearSVC, params are usually simple enough.
    print(f"\nTesting LinearSVC with Parameters: {params}")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            model = LinearSVC(**params)
            model.fit(X_train_tfidf, y_train)

            # Check for convergence warnings specific to this fit
            for cw_obj in caught_warnings:
                # Check if the warning message is relevant to LinearSVC convergence
                if "Liblinear failed to converge" in str(cw_obj.message) or \
                   "loss == 'hinge' and penalty == 'l2'" in str(cw_obj.message): # Common pattern for hinge
                    current_run_warnings.append(str(cw_obj.message))
            # caught_warnings.clear() # Not strictly needed here as context manager handles it

            if current_run_warnings:
                print(f"  ConvergenceWarning(s): {current_run_warnings}")

            y_val_pred = model.predict(X_val_tfidf)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision_macro = precision_score(y_val, y_val_pred, average='macro', zero_division=0)

            y_holdout_pred = model.predict(X_holdout_tfidf)
            holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
            holdout_precision_macro = precision_score(y_holdout, y_holdout_pred, average='macro', zero_division=0)

            print(f"  LinearSVC - Validation: Accuracy = {val_accuracy:.4f}, Precision (Macro) = {val_precision_macro:.4f}")
            print(f"  LinearSVC - Holdout:    Accuracy = {holdout_accuracy:.4f}, Precision (Macro) = {holdout_precision_macro:.4f}")

            results_svc_all.append({
                'params': params,
                'val_accuracy': val_accuracy,
                'val_precision_macro': val_precision_macro,
                'holdout_accuracy': holdout_accuracy,
                'holdout_precision_macro': holdout_precision_macro,
                'convergence_warnings': bool(current_run_warnings)
            })

        except ValueError as e: # Catches incompatible parameter combinations primarily
            print(f"  SKIPPING LinearSVC combination due to ValueError: {e}")
            results_svc_all.append({
                'params': params, 'error': f"ValueError: {str(e)}"
            })
        except Exception as e: # Catches any other unexpected errors during fit/predict
            print(f"  SKIPPING LinearSVC combination due to unexpected error: {type(e).__name__} - {str(e)}")
            results_svc_all.append({
                'params': params, 'error': f"{type(e).__name__}: {str(e)}"
            })
    print("-" * 60) # Increased separator length for clarity


# --- 3. Final Consolidated Results ---
print("\n\n--- FINAL CONSOLIDATED LinearSVC Tuning RESULTS ---")

if results_svc_all:
    df_svc = pd.DataFrame(results_svc_all)
    # Ensure 'params' column is string for consistent display if it contains dicts
    df_svc['params_str'] = df_svc['params'].astype(str)

    # Select relevant columns for display, ensuring they exist
    cols_to_show_svc = [
        'params_str',
        'holdout_accuracy', 'val_accuracy',
        'holdout_precision_macro', 'val_precision_macro',
        'convergence_warnings', 'error'
    ]
    # Filter out columns that might not exist in all rows (e.g., 'error' only for failed runs)
    existing_cols_svc = [col for col in cols_to_show_svc if col in df_svc.columns]

    # Sort by holdout_accuracy, handling cases where it might be NaN (for errored runs)
    # Place NaNs (errors) at the bottom when sorting
    if 'holdout_accuracy' in df_svc.columns:
        df_svc_sorted = df_svc.sort_values(by='holdout_accuracy', ascending=False, na_position='last')
    else:
        df_svc_sorted = df_svc # No holdout_accuracy to sort by if all failed before that point

    print(df_svc_sorted[existing_cols_svc].to_string(index=False, max_colwidth=80)) # Increased max_colwidth

    # Print best model based on holdout accuracy
    if 'holdout_accuracy' in df_svc_sorted.columns and not df_svc_sorted['holdout_accuracy'].isnull().all():
        best_model_row = df_svc_sorted.dropna(subset=['holdout_accuracy']).iloc[0]
        print("\n\n--- BEST LinearSVC MODEL (by Holdout Accuracy) ---")
        print(f"Parameters: {best_model_row['params']}")
        print(f"  Holdout Accuracy:          {best_model_row.get('holdout_accuracy', 'N/A'):.4f}")
        print(f"  Validation Accuracy:       {best_model_row.get('val_accuracy', 'N/A'):.4f}")
        print(f"  Holdout Precision (Macro): {best_model_row.get('holdout_precision_macro', 'N/A'):.4f}")
        print(f"  Validation Precision (Macro):{best_model_row.get('val_precision_macro', 'N/A'):.4f}")
        if best_model_row.get('convergence_warnings', False):
            print("  NOTE: This model reported convergence warnings.")
    else:
        print("\nNo models were successfully evaluated to determine a best model.")

else:
    print("No results were recorded from LinearSVC tuning.")
