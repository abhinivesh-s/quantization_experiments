import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# --- Configuration ---
N_CLASSES = 37 # Your number of classes
N_SAMPLES = N_CLASSES * 30 # Ensure enough samples per class for reasonable training
N_FEATURES = 50

# --- Generate Sample Data (replace with your actual data) ---
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=int(N_FEATURES * 0.7), # e.g., 70% informative
    n_redundant=int(N_FEATURES * 0.1), # e.g., 10% redundant
    n_classes=N_CLASSES,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Model Training ---

# 1. Define the base LinearSVC
# dual="auto" is generally a good choice. For n_samples > n_features, it often defaults to False (primal).
# For n_features > n_samples, it defaults to True (dual).
# Explicitly set max_iter if you encounter convergence warnings.
base_svc = LinearSVC(random_state=42, dual="auto", max_iter=2000)

# 2. Wrap it in CalibratedClassifierCV
# Common scenario: using cross-validation for calibration
# Let's use cv=3 for demonstration; 5 is the default.
# 'sigmoid' is Platt scaling, 'isotonic' is isotonic regression.
calibrated_clf = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3) # Use cv=5 for production
calibrated_clf.fit(X_train, y_train)

# --- Getting the Scores ---


print(f"\n--- Interpretation 2: Individual Calibrated OvR Probabilities (before final sum-to-1 normalization) ---")
# These are P(class_k vs Rest | X) for each class k, after calibration but before normalization across classes.
# Shape: (n_samples, n_classes)

all_individual_calibrated_probs_folds = []
for calibrator_item in calibrated_clf.calibrated_classifiers_:
    # _CalibratedClassifier object
    fold_base_estimator = calibrator_item.base_estimator_
    fold_calibrators = calibrator_item.calibrators_ # List of actual calibrator objects (e.g., _SigmoidCalibration)

    # Get decision scores from this fold's base estimator
    decision_scores_fold = fold_base_estimator.decision_function(X_test) # (n_samples, n_classes)

    # Apply each class's calibrator to its respective decision scores
    individual_probs_this_fold = np.zeros_like(decision_scores_fold)
    for k_class in range(N_CLASSES):
        # Scores for class k (OvR)
        class_k_decision_scores = decision_scores_fold[:, k_class]
        
        # Calibrator for class k
        calibrator_for_class_k = fold_calibrators[k_class]
        
        # Predict probability using this specific calibrator
        # For _SigmoidCalibration or _IsotonicRegression, .predict() gives P(positive class | score)
        # The input to .predict() needs to be 2D for some versions or calibrators, usually (n_samples, 1)
        # However, for the internal _SigmoidCalibration and _IsotonicRegression, predict takes 1D.
        prob_class_k = calibrator_for_class_k.predict(class_k_decision_scores)
        individual_probs_this_fold[:, k_class] = prob_class_k
        
    all_individual_calibrated_probs_folds.append(individual_probs_this_fold)

# Average these individual calibrated OvR probabilities across the folds
avg_individual_calibrated_probs = np.mean(all_individual_calibrated_probs_folds, axis=0)

print(f"Shape of averaged individual OvR calibrated probabilities: {avg_individual_calibrated_probs.shape}")
if X_test.shape[0] > 0:
    print(f"Example individual calibrated OvR probs for first test sample (first 5 classes):\n{avg_individual_calibrated_probs[0, :5]}")
    print(f"Sum of these for first sample (should NOT necessarily be 1): {np.sum(avg_individual_calibrated_probs[0, :])}")
    print(f"Full individual calibrated OvR probs for first test sample (all {N_CLASSES} classes):\n{avg_individual_calibrated_probs[0, :]}")


# For comparison, here's what CalibratedClassifierCV.predict_proba() gives:
if X_test.shape[0] > 0:
    final_normalized_probs = calibrated_clf.predict_proba(X_test)
    print(f"\n--- For Comparison: Final predict_proba() output (normalized) ---")
    print(f"Shape of final normalized probabilities: {final_normalized_probs.shape}")
    print(f"Example final normalized probs for first test sample (first 5 classes):\n{final_normalized_probs[0, :5]}")
    print(f"Sum of these for first sample (SHOULD be 1): {np.sum(final_normalized_probs[0, :])}")

    # You can verify that normalizing avg_individual_calibrated_probs gives final_normalized_probs
    manually_normalized = avg_individual_calibrated_probs / np.sum(avg_individual_calibrated_probs, axis=1, keepdims=True)
    assert np.allclose(manually_normalized, final_normalized_probs), "Manual normalization mismatch!"
    print("Manual normalization of Interpretation 2 matches predict_proba output: True")
