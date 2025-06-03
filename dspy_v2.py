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
def precision_accuracy_metric(gold: List[dspy.Example | Any], preds: List[dspy.Prediction], trace=None) -> float:
    global metric_call_count
    metric_call_count += 1
    print(f"\n--- DEBUG: precision_accuracy_metric call #{metric_call_count} ---")
    print(f"Trace object: {trace}") # Trace is often None during optimizer, but not always

    if not gold:
        print("DEBUG METRIC: 'gold' list is empty or None. Returning 0.0")
        return 0.0

    print(f"DEBUG METRIC: Length of 'gold': {len(gold)}")
    print(f"DEBUG METRIC: Type of 'gold' list: {type(gold)}")
    print(f"DEBUG METRIC: Type of first element gold[0]: {type(gold[0])}")

    if not isinstance(gold[0], dspy.Example):
        print(f"DEBUG METRIC FATAL ERROR: gold[0] is NOT a dspy.Example. Value: {gold[0]}")
        print("This means the 'dev_data' passed to the optimizer (implicitly via global scope or explicitly via evaluate) is malformed.")
        raise TypeError(f"FATAL: Expected dspy.Example in gold data, got {type(gold[0])}: {gold[0]}")
    else:
        print(f"DEBUG METRIC: gold[0] is a dspy.Example. Keys: {gold[0].keys()}")
        if not hasattr(gold[0], 'true_class'):
            print(f"DEBUG METRIC FATAL ERROR: gold[0] (Value: {gold[0]}) LACKS 'true_class' attribute.")
            raise AttributeError(f"FATAL: Gold example {gold[0]} is missing 'true_class' field.")
        print(f"DEBUG METRIC: gold[0].true_class value: {gold[0].true_class}")

    gold_labels = []
    for i, g in enumerate(gold):
        print(f"DEBUG METRIC: Processing gold item #{i}, type: {type(g)}") # Print type of each item
        if not isinstance(g, dspy.Example): # Check every item, not just the first
            print(f"DEBUG METRIC FATAL ERROR: Gold item #{i} is NOT a dspy.Example. Value: {g}")
            raise TypeError(f"FATAL: Expected dspy.Example for gold item #{i}, got {type(g)}: {g}")
        if not hasattr(g, 'true_class'):
            print(f"DEBUG METRIC FATAL ERROR: Gold item #{i} (Value: {g}) LACKS 'true_class' attribute.")
            raise AttributeError(f"FATAL: Gold example #{i} {g} is missing 'true_class' field.")
        
        gold_labels.append(g.true_class) # <--- ERROR SITE IF g IS A STRING

    # ... (rest of your metric function: pred_labels, calculations, prints)
    pred_labels = []
    for p_idx, p in enumerate(preds):
        if not hasattr(p, 'predicted_class') or not isinstance(p.predicted_class, str):
            pred_labels.append("MALFORMED_PRED")
            continue
        raw_pred = p.predicted_class.strip().replace("'", "").replace('"', '').replace('.', '')
        if raw_pred in ALL_CLASSES: pred_labels.append(raw_pred)
        else:
            matched_class = next((kc for kc in ALL_CLASSES if kc in raw_pred), raw_pred)
            pred_labels.append(matched_class)

    macro_precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0, labels=ALL_CLASSES)
    accuracy = accuracy_score(gold_labels, pred_labels)
    if trace is None:
        print(f"Validation Metrics: Macro Precision: {macro_precision:.4f}, Accuracy: {accuracy:.4f}")
        # print(classification_report(gold_labels, pred_labels, zero_division=0, labels=ALL_CLASSES, target_names=ALL_CLASSES))
    print("--- END DEBUG: precision_accuracy_metric ---")
    return macro_precision

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
