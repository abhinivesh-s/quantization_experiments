import dspy
import os
from typing import List, Dict, Any
from sklearn.metrics import precision_score, accuracy_score, classification_report
import random # For generating sample data

# --- 1. Configuration ---
# Replace with your actual API key and model
# For OpenAI:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=50)
# Or, if you prefer chat models:
turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=50) # Output is just a class label

# For Anthropic:
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
# turbo = dspy.Anthropic(model='claude-3-haiku-20240307', max_tokens=50)

dspy.settings.configure(lm=turbo)

# --- 2. Class Definitions for 37 Record Class Codes ---
# Generate placeholder record class codes
NUM_CLASSES = 37
ALL_CLASSES = [f"RC_{i:02d}" for i in range(1, NUM_CLASSES + 1)] # e.g., RC_01, RC_02, ..., RC_37

# Your predefined class definitions as a string
# For 37 classes, this could be just a list of the codes,
# or short descriptions if available and they fit the prompt.
# Example: if you have descriptions, format them clearly.
# For simplicity here, I'll just list the codes.
CLASS_DEFINITIONS_STR = "Available record class codes:\n" + "\n".join([f"- {cls}" for cls in ALL_CLASSES])
# If you have actual descriptions, you'd format it like:
# CLASS_DEFINITIONS_STR = """
# Available record class codes:
# - RC_01: Description for Record Class 01...
# - RC_02: Description for Record Class 02...
# ...
# - RC_37: Description for Record Class 37...
# """

# --- 3. Data Preparation ---
# Create example training and development data for 37 classes
# In a real scenario, you'd load this from your dataset.
# For this example, we'll generate some synthetic data.

def generate_synthetic_data(num_examples: int, classes: List[str]) -> List[dspy.Example]:
    data = []
    for i in range(num_examples):
        # Pick a random class
        true_class = random.choice(classes)
        # Create a synthetic text. In a real scenario, this would be actual data.
        # This synthetic text is very simple; real text would be more complex.
        text_variations = [
            f"This document clearly pertains to {true_class}.",
            f"Data entry for item related to {true_class} classification.",
            f"Processing information for record class {true_class}.",
            f"Log entry type {true_class}.",
            f"Item categorized under {true_class}."
        ]
        text = random.choice(text_variations) + f" (Sample ID: {i})"
        data.append(dspy.Example(text=text, true_class=true_class).with_inputs("text"))
    return data

# Generate more diverse data if possible.
# For 37 classes, you'd ideally have several examples per class.
# Here, we generate a small dataset for demonstration.
# Training data: ~3 examples per class on average (if num_examples is around 3*37)
# You'll need much more for real-world performance.
NUM_TRAIN_EXAMPLES = 80 # Adjust as needed, more is better
NUM_DEV_EXAMPLES = 40   # Adjust as needed

train_data = generate_synthetic_data(NUM_TRAIN_EXAMPLES, ALL_CLASSES)
dev_data = generate_synthetic_data(NUM_DEV_EXAMPLES, ALL_CLASSES)

# Ensure all classes are represented in train/dev if possible, or handle in metrics
print(f"Generated {len(train_data)} training examples.")
print(f"Generated {len(dev_data)} development examples.")
# Optional: Check class distribution if needed
# from collections import Counter
# print("Train class distribution:", Counter(ex.true_class for ex in train_data))
# print("Dev class distribution:", Counter(ex.true_class for ex in dev_data))


# --- 4. DSPy Signature Definition ---
class MultiClassClassificationSignature(dspy.Signature):
    """Classify the input text into one of the predefined record class codes.
    Output only the record class code label."""

    text: str = dspy.InputField(desc="The text to classify.")
    predicted_class: str = dspy.OutputField(desc=f"One of: {', '.join(ALL_CLASSES)}")

# --- 5. DSPy Module ---
class SimpleClassifier(dspy.Module):
    def __init__(self, class_definitions_str: str):
        super().__init__()
        self.class_definitions_str = class_definitions_str
        instruction = (
            f"You are a precise classification model. Classify the input text into one of the following record class codes.\n"
            f"{self.class_definitions_str}\n"
            f"Respond with ONLY the record class code (e.g., 'RC_01', 'RC_25'). Do not add any other text or explanation."
        )
        self.classifier = dspy.Predict(MultiClassClassificationSignature.with_instructions(instruction))

    def forward(self, text: str) -> dspy.Prediction:
        return self.classifier(text=text)

# --- 6. Metric Function with Enhanced Debugging ---
metric_call_count = 0
def precision_accuracy_metric(gold: List[dspy.Example], preds: List[dspy.Prediction], trace=None) -> float:
    global metric_call_count
    metric_call_count += 1
    print(f"\n--- METRIC CALL #{metric_call_count} --- Trace: {trace is not None} ---")

    if not gold:
        print("METRIC WARNING: 'gold' (dev_data) list is empty. Returning 0.0")
        return 0.0
    if not preds:
        print("METRIC WARNING: 'preds' list is empty. Returning 0.0") # Should not happen if gold is not empty
        return 0.0

    gold_labels = []
    for i, g in enumerate(gold):
        if not isinstance(g, dspy.Example):
            print(f"METRIC ERROR: gold[{i}] is not a dspy.Example. Type: {type(g)}, Val: {g}")
            raise TypeError("Malformed gold data in metric.")
        if not hasattr(g, 'true_class') or g.true_class is None: # Check for None explicitly
            print(f"METRIC ERROR: gold[{i}] (Text: {getattr(g, 'text', 'N/A')}) missing 'true_class' or it's None. Val: {g}")
            raise ValueError("Gold example missing true_class or it's None.")
        if not isinstance(g.true_class, str) or not g.true_class.strip(): # Ensure it's a non-empty string
             print(f"METRIC ERROR: gold[{i}].true_class is not a valid string: '{g.true_class}' (Type: {type(g.true_class)})")
             raise ValueError("Gold true_class is not a valid string.")
        gold_labels.append(g.true_class)

    pred_labels = []
    for i, p in enumerate(preds):
        if not isinstance(p, dspy.Prediction):
            print(f"METRIC WARNING: preds[{i}] is not a dspy.Prediction. Type: {type(p)}, Val: {p}. Skipping.")
            # Decide how to handle: skip, add placeholder, or error
            # For now, let's try to make lists same length if gold_labels is longer. This is risky.
            # A better approach might be to ensure LLM always returns something or have a default.
            if len(pred_labels) < len(gold_labels): # If we skip, lists might mismatch
                 pred_labels.append("INVALID_PRED_TYPE_" + random.choice(ALL_CLASSES)) # Placeholder
            continue
        if not hasattr(p, 'predicted_class') or p.predicted_class is None:
            print(f"METRIC WARNING: preds[{i}] missing 'predicted_class' or it's None. Gold text: {gold[i].text if i < len(gold) else 'N/A'}. Using a placeholder.")
            pred_labels.append("MISSING_PRED_" + random.choice(ALL_CLASSES)) # Placeholder
            continue
        
        raw_pred = str(p.predicted_class).strip() # Ensure it's a string
        if not raw_pred: # Empty prediction
            print(f"METRIC WARNING: preds[{i}] resulted in empty string after strip. Gold text: {gold[i].text if i < len(gold) else 'N/A'}. Using a placeholder.")
            pred_labels.append("EMPTY_PRED_" + random.choice(ALL_CLASSES))
            continue

        # Simple normalization (you might need more sophisticated logic)
        # Try to find an exact match first
        best_match = raw_pred
        if raw_pred not in ALL_CLASSES:
            # If not exact, check if any known class is a substring of the prediction
            # Or if the prediction is a substring of a known class (less likely for your codes)
            found_substring_match = False
            for known_class in ALL_CLASSES:
                if known_class in raw_pred: # e.g., pred is "Category RC_01"
                    best_match = known_class
                    found_substring_match = True
                    break
            if not found_substring_match:
                 # If still no match, it's an unknown prediction. It will hurt precision for its true class.
                 # Keep `best_match = raw_pred` which is the original unknown prediction.
                 print(f"METRIC INFO: Prediction '{raw_pred}' not in ALL_CLASSES. Gold: {gold[i].true_class if i < len(gold) else 'N/A'}")
        
        pred_labels.append(best_match)

    print(f"METRIC INFO: Length of gold_labels: {len(gold_labels)}")
    print(f"METRIC INFO: Length of pred_labels: {len(pred_labels)}")

    # Ensure lists are of the same length before passing to sklearn.
    # This is a critical step. DSPy should ensure this, but defensive check is good.
    if len(gold_labels) != len(pred_labels):
        print(f"METRIC CRITICAL ERROR: Mismatch in lengths! gold_labels ({len(gold_labels)}), pred_labels ({len(pred_labels)})")
        # This should ideally not happen if DSPy's evaluation loop is correct.
        # If it does, we cannot reliably calculate metrics.
        # Option 1: Raise an error.
        # Option 2: Truncate to the shorter length (information loss).
        # Option 3: Return a very bad score.
        print("Gold Labels sample:", gold_labels[:5])
        print("Pred Labels sample:", pred_labels[:5])
        # For debugging, let's allow it to error out in sklearn if this happens, or return 0
        # raise ValueError("CRITICAL: Mismatch in length of gold and pred labels for metric calculation.")
        return 0.0 # Or some very low score to indicate a serious problem

    if not gold_labels: # Should be caught by `if not gold:` but as a safeguard
        print("METRIC WARNING: gold_labels list is empty before scikit-learn. Returning 0.0")
        return 0.0

    print(f"METRIC INFO: Unique gold labels: {sorted(list(set(gold_labels)))}")
    print(f"METRIC INFO: Unique pred labels: {sorted(list(set(pred_labels)))}")
    # Ensure all labels in pred_labels that are not in gold_labels are at least covered by ALL_CLASSES for the report
    # `labels=ALL_CLASSES` in precision_score and classification_report handles this.

    final_score = 0.0
    try:
        macro_precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0, labels=ALL_CLASSES)
        accuracy = accuracy_score(gold_labels, pred_labels)
        final_score = macro_precision

        # Only print full report if trace is None (typically final eval)
        if trace is None and metric_call_count % 1 == 0 : # Print every time for now during debugging
            print(f"METRIC Scores: Macro Precision: {macro_precision:.4f}, Accuracy: {accuracy:.4f}")
            print("Detailed Classification Report:")
            # Ensure no new labels in pred_labels that are not in ALL_CLASSES cause issues.
            # `labels=ALL_CLASSES` should handle this by restricting the report to these classes.
            # However, if pred_labels contains items NOT in ALL_CLASSES, they will be ignored for per-class metrics
            # unless they happen to match a gold_label that IS in ALL_CLASSES (which is fine).
            # If a pred_label is totally novel AND its corresponding gold_label is in ALL_CLASSES, it's a misclassification.
            report = classification_report(gold_labels, pred_labels, zero_division=0, labels=ALL_CLASSES, target_names=ALL_CLASSES)
            print(report)

    except ValueError as ve:
        print(f"METRIC ERROR during scikit-learn calculation: {ve}")
        print("This can happen if pred_labels contain values not in 'labels' and also not in 'gold_labels' set, "
              "or if either list is empty in an unexpected way.")
        print("Problematic Gold Labels:", gold_labels)
        print("Problematic Pred Labels:", pred_labels)
        traceback.print_exc()
        return 0.0 # Return a penalty score
    except Exception as e:
        print(f"METRIC UNEXPECTED ERROR during scikit-learn calculation: {e}")
        traceback.print_exc()
        return 0.0 # Return a penalty score

    print(f"--- METRIC RETURNING SCORE: {final_score:.4f} ---")
    return final_score


# --- 7. Optimizer Setup and Compilation ---
# ... (all previous code for BootstrapFewShot setup, including SimpleClassifier) ...

# --- 7. Optimizer Setup and Compilation (Corrected for BootstrapFewShot) ---
if __name__ == "__main__":
    # Instantiate the uncompiled module
    uncompiled_classifier = SimpleClassifier(class_definitions=CLASS_DEFINITIONS_STR) # Or class_definitions_str for the 37-class version

    # Test the uncompiled module on a dev example
    print("--- Testing Uncompiled Classifier ---")
    if dev_data: # Ensure dev_data exists
        example_dev_text = dev_data[0].text
        true_class_example = dev_data[0].true_class
        prediction = uncompiled_classifier(example_dev_text)
        print(f"Input: {example_dev_text}")
        print(f"True Class: {true_class_example}")
        print(f"Predicted class (uncompiled): {prediction.predicted_class}")
        # Inspect the prompt sent by the uncompiled module
        if hasattr(turbo, 'inspect_history'):
            turbo.inspect_history(n=0) # Clear history
            uncompiled_classifier(example_dev_text) # Make the call again
            if turbo.history:
                print("\nPrompt sent by UNCOMPILED module for this example:")
                print(turbo.history[0]['prompt'])
        print("-----------------------------------\n")
    else:
        print("Skipping uncompiled classifier test as dev_data is empty.")


    # Set up the BootstrapFewShot optimizer
    optimizer = dspy.BootstrapFewShot(
        metric=precision_accuracy_metric, # This metric function will use dev_data
        max_bootstrapped_demos=3,
        max_labeled_demos=16,
        # teacher_settings={'lm': turbo} # If your teacher model is different
    )

    print("--- Starting Compilation (Optimization with BootstrapFewShot) ---")
    # Ensure train_data is not empty
    if not train_data:
        print("Error: Training data is empty. Cannot compile.")
        exit()
    # Note: The `dev_data` is passed to the metric function internally by the optimizer
    # when it needs to evaluate a set of demonstrations.
    # The metric function itself needs access to dev_data (e.g., via global scope or closure).
    # In our current setup, `precision_accuracy_metric` uses `dev_data` which is in the global scope.
    # DSPy's BootstrapFewShot will call metric(train_data_subset_as_demos, dev_data_predictions)

    compiled_classifier = optimizer.compile(
        student=uncompiled_classifier, # The module we want to optimize
        trainset=train_data
        # NO valset argument here for BootstrapFewShot.compile()
    )
    print("--- Compilation Finished ---")

    # --- 8. Inspect the Optimized Prompt and Evaluate ---
    print("\n--- Evaluating Compiled Classifier (Optimized with BootstrapFewShot) ---")
    if dev_data:
        # The evaluate method *does* take a valset (or any dataset to evaluate on)
        compiled_classifier.evaluate(dev_data, metric=precision_accuracy_metric, display_progress=True, display_table=0)
    else:
        print("No dev data to evaluate compiled classifier.")

    # ... (rest of the inspection code for BootstrapFewShot: instructions, demos, final prompt example) ...

    print("\n--- Optimized Prompt Components (from BootstrapFewShot) ---")
    final_predictor = compiled_classifier.classifier # This is the dspy.Predict module
    print("Optimized Instructions (remains the same as defined in SimpleClassifier):")
    print(final_predictor.signature.instructions)

    if final_predictor.demos:
        print(f"\nOptimized Few-Shot Examples ({len(final_predictor.demos)} Demos selected by BootstrapFewShot):")
        for i, demo in enumerate(final_predictor.demos):
            print(f"Demo {i+1}:")
            demo_dict = demo.toDict()
            print(f"  Input ('text'): {demo_dict.get('text', 'N/A')}")
            print(f"  Output ('predicted_class'): {demo_dict.get('predicted_class', 'N/A')}")
    else:
        print("\nNo few-shot examples were selected by the optimizer.")

    print("\n--- Final Prompt Example (for first dev data point if available) ---")
    if dev_data:
        example_input = dev_data[0].text
        true_label_example = dev_data[0].true_class

        if hasattr(turbo, 'inspect_history'):
            turbo.inspect_history(n=0) # Clear history

        prediction_compiled = compiled_classifier(example_input)
        print(f"Input: {example_input}")
        print(f"True Class: {true_label_example}")
        print(f"Predicted class (compiled): {prediction_compiled.predicted_class}")

        if hasattr(turbo, 'inspect_history') and turbo.history:
            print("\nFull prompt sent to LLM for the above input using the COMPILED module:")
            final_prompt_sent = turbo.history[0]['prompt']
            print(final_prompt_sent)
        else:
            print("\n(Could not inspect LM history for the final prompt automatically.)")

    else:
        print("No dev data to show final prompt example.")
