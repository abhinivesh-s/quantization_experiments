import dspy
import os
from typing import List, Dict, Any
from sklearn.metrics import precision_score, accuracy_score, classification_report
import random
import pandas as pd
import traceback

# --- Configuration & Data Setup (Assume this part is now perfect) ---
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=50)
dspy.settings.configure(lm=turbo)

NUM_CLASSES = 5
ALL_CLASSES = [f"RC_{i:02d}" for i in range(1, NUM_CLASSES + 1)]
CLASS_DEFINITIONS_STR = "Available record class codes:\n" + "\n".join([f"- {cls}" for cls in ALL_CLASSES])

# Data creation function (assuming it's been perfected and validated)
def create_perfect_dspy_examples(df: pd.DataFrame, name: str) -> List[dspy.Example]:
    # This function should now contain all your rigorous cleaning and validation
    # and return a list of pristine dspy.Example objects or raise an error.
    print(f"Creating (assumed perfect) examples for {name}...")
    examples = []
    for _, row in df.iterrows():
        # Simplified for this example, assuming 'text' and 'RCC' are clean strings
        if isinstance(row['text'], str) and isinstance(row['RCC'], str) and row['text'] and row['RCC']:
            examples.append(dspy.Example(text=row['text'], true_class=row['RCC']).with_inputs('text'))
        else:
            print(f"Skipping potentially problematic row in {name}: {row.to_dict()}") # Should not happen if data is perfect
    print(f"Created {len(examples)} for {name}.")
    if not examples: raise ValueError(f"{name} data is empty after creation.")
    return examples

# Dummy DataFrames
train_df_raw = pd.DataFrame({
    'text': [f"Perfect train text {i}" for i in range(10)],
    'RCC': [random.choice(ALL_CLASSES) for _ in range(10)]
})
dev_df_raw = pd.DataFrame({
    'text': [f"Perfect dev text {i}" for i in range(5)],
    'RCC': [random.choice(ALL_CLASSES) for _ in range(5)]
})

train_data = create_perfect_dspy_examples(train_df_raw, "train_data")
dev_data = create_perfect_dspy_examples(dev_df_raw, "dev_data") # This is the 'gold' data

# --- Sanity checks right before compile (essential) ---
print("\n--- FINAL PRE-COMPILE SANITY CHECK ---")
for i, ex in enumerate(train_data):
    assert isinstance(ex, dspy.Example) and hasattr(ex, 'true_class') and hasattr(ex, 'text'), f"Train data malformed at index {i}"
for i, ex in enumerate(dev_data):
    assert isinstance(ex, dspy.Example) and hasattr(ex, 'true_class') and hasattr(ex, 'text'), f"Dev data malformed at index {i}"
print("--- FINAL PRE-COMPILE SANITY CHECK PASSED ---")
# --- End Data Setup ---


# --- Signature and Module (Same) ---
class MultiClassClassificationSignature(dspy.Signature): # ... as before
    text: str = dspy.InputField(desc="The text to classify.")
    predicted_class: str = dspy.OutputField(desc=f"One of: {', '.join(ALL_CLASSES)}")

class SimpleClassifier(dspy.Module): # ... as before
    def __init__(self, class_definitions_str: str):
        super().__init__()
        self.class_definitions_str = class_definitions_str
        instruction = (
            f"Classify into one of: {', '.join(ALL_CLASSES)}. "
            f"{self.class_definitions_str} Respond with ONLY the class code."
        )
        self.classifier = dspy.Predict(MultiClassClassificationSignature.with_instructions(instruction))
    def forward(self, text: str) -> dspy.Prediction: return self.classifier(text=text)
# --- End Signature and Module ---


# --- Hyper-Defensive Metric Function ---
metric_call_count = 0
def precision_accuracy_metric(gold: List[Any], preds: List[Any], trace=None) -> float:
    global metric_call_count
    metric_call_count += 1
    print(f"\n--- METRIC CALL #{metric_call_count} --- ENTER --- Trace: {trace is not None} ---")
    print(f"Type of gold: {type(gold)}, Length: {len(gold) if gold else 0}")
    print(f"Type of preds: {type(preds)}, Length: {len(preds) if preds else 0}")

    # === DSPY BUG WORKAROUND/DETECTION BLOCK ===
    if gold and isinstance(gold[0], str):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"METRIC DETECTED UNEXPECTED `gold[0]` TYPE: {type(gold[0])}, VALUE: '{gold[0]}'")
        if gold[0] == "text" and len(gold) == 1: # Check for the specific "text" string scenario
            print("  This matches the suspected DSPy issue where a field name is passed as data.")
            print("  Attempting to gracefully handle by returning a penalty score (0.0).")
            print("  This specific metric call will be skipped for calculation.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return 0.0 # Return a penalty score and skip this evaluation
        else:
            print("  `gold[0]` is a string, but not the specific 'text' case. This is also unexpected.")
            print("  Proceeding with caution, but this is likely to fail.")
            # Fall through to normal error handling if it's a different string.
    # === END DSPY BUG WORKAROUND/DETECTION BLOCK ===

    if not gold or not isinstance(gold[0], dspy.Example): # If not the string "text" bug, but still bad
        print(f"METRIC ERROR: `gold` data is malformed or empty despite workaround.")
        print(f"  Type of gold[0] now: {type(gold[0]) if gold else 'N/A'}, Value: {gold[0] if gold else 'N/A'}")
        # This means the workaround didn't catch it, or it's a different issue.
        return 0.0 # Penalty score

    # --- Standard Gold Label Extraction (assuming gold is now List[dspy.Example]) ---
    gold_labels = []
    try:
        for i, g in enumerate(gold):
            if not isinstance(g, dspy.Example) or not hasattr(g, 'true_class') or \
               not isinstance(g.true_class, str) or not g.true_class.strip():
                print(f"METRIC ERROR: Malformed dspy.Example in gold at index {i}. Val: {g}. Skipping this metric call.")
                return 0.0 # Problem with an individual example
            gold_labels.append(g.true_class)
    except Exception as e_gold_proc:
        print(f"METRIC CRITICAL ERROR processing gold labels: {e_gold_proc}")
        traceback.print_exc()
        return 0.0

    # --- Standard Prediction Label Extraction & Normalization ---
    pred_labels = []
    if len(preds) != len(gold): # Basic check, DSPy should handle this
        print(f"METRIC WARNING: preds length ({len(preds)}) != gold length ({len(gold)}). This is problematic.")
        # Pad pred_labels or return 0.0. For now, returning 0.0 is safer.
        return 0.0

    for i, p in enumerate(preds):
        if not isinstance(p, dspy.Prediction) or not hasattr(p, 'predicted_class') or p.predicted_class is None:
            pred_labels.append("INVALID_PRED_" + random.choice(ALL_CLASSES)) # Placeholder
            continue
        raw_pred = str(p.predicted_class).strip()
        if not raw_pred:
            pred_labels.append("EMPTY_PRED_" + random.choice(ALL_CLASSES)) # Placeholder
            continue
        
        best_match = raw_pred
        if raw_pred not in ALL_CLASSES: # Simple normalization
            best_match = next((kc for kc in ALL_CLASSES if kc in raw_pred), raw_pred)
        pred_labels.append(best_match)

    # --- Scikit-learn Calculation ---
    if not gold_labels: # Should have been caught if gold was empty
        print("METRIC WARNING: gold_labels list became empty. Returning 0.0")
        return 0.0
        
    final_score = 0.0
    try:
        macro_precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0, labels=ALL_CLASSES)
        final_score = macro_precision
        if trace is None or metric_call_count % 1 == 0: # Log periodically or on final eval
            accuracy = accuracy_score(gold_labels, pred_labels)
            print(f"METRIC Scores: Macro Precision: {macro_precision:.4f}, Accuracy: {accuracy:.4f}")
            # print(classification_report(gold_labels, pred_labels, zero_division=0, labels=ALL_CLASSES, target_names=ALL_CLASSES))
    except ValueError as ve: # Specific sklearn error
        print(f"METRIC ERROR during scikit-learn (ValueError): {ve}")
        print(f"  Gold labels ({len(gold_labels)}): {gold_labels[:5]}...")
        print(f"  Pred labels ({len(pred_labels)}): {pred_labels[:5]}...")
        traceback.print_exc()
        return 0.0 # Penalty score
    except Exception as e_sklearn: # Other sklearn errors
        print(f"METRIC UNEXPECTED ERROR during scikit-learn: {e_sklearn}")
        traceback.print_exc()
        return 0.0 # Penalty score

    print(f"--- METRIC RETURNING SCORE: {final_score:.4f} ---")
    return final_score
# --- End Metric Function ---


# --- Main Execution Block ---
if __name__ == "__main__":
    # Assertions for train_data and dev_data should have passed above.
    if not train_data or not dev_data:
        print("CRITICAL: train_data or dev_data is empty. Exiting.")
        exit()

    uncompiled_classifier = SimpleClassifier(class_definitions_str=CLASS_DEFINITIONS_STR)
    optimizer = dspy.BootstrapFewShot(
        metric=precision_accuracy_metric,
        max_bootstrapped_demos=2, # Keep small for debugging
        max_labeled_demos=5,
    )

    print("\n--- Starting Compilation (BootstrapFewShot) ---")
    try:
        compiled_classifier = optimizer.compile(
            student=uncompiled_classifier,
            trainset=train_data # dev_data is accessed globally by the metric
        )
        print("--- Compilation Finished ---")
    except Exception as e:
        print(f"\n!!! ERROR DURING optimizer.compile() !!!")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        print("If METRIC logs show detection of 'gold[0] == \"text\"', the workaround was triggered.")
        print("This points to an issue with how DSPy calls the metric.")
        exit("Halting.")

    print("\n--- Evaluating Compiled Classifier ---")
    if dev_data:
        # Explicitly pass the known-good dev_data for evaluation
        compiled_classifier.evaluate(dev_data, metric=precision_accuracy_metric, display_progress=True, display_table=0)
    else:
        print("No dev_data to evaluate.")
