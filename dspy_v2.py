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

# --- 6. Metric Function ---
def precision_accuracy_metric(gold: List[dspy.Example], preds: List[dspy.Prediction], trace=None) -> float:
    gold_labels = [g.true_class for g in gold]
    pred_labels = []

    for p in preds:
        raw_pred = p.predicted_class.strip() # Get the predicted class string
        # Simple cleaning: sometimes models add quotes or periods.
        raw_pred = raw_pred.replace("'", "").replace('"', '').replace('.', '')

        # Check if the raw prediction is one of the known classes
        if raw_pred in ALL_CLASSES:
            pred_labels.append(raw_pred)
        else:
            # If not an exact match, try to find a known class within the prediction string
            # This is a basic attempt; more sophisticated matching might be needed for messy outputs
            matched_class = None
            for known_class in ALL_CLASSES:
                if known_class in raw_pred:
                    matched_class = known_class
                    break
            if matched_class:
                pred_labels.append(matched_class)
            else:
                # If no known class is found, append the raw prediction.
                # This might lead to Scikit-learn treating it as a new, incorrect class.
                # Or, you could assign a default "unknown" or one of the classes if appropriate.
                pred_labels.append(raw_pred) # Or handle as 'unknown'
                # print(f"Warning: Prediction '{raw_pred}' not in known classes. Gold: {gold[len(pred_labels)-1].true_class if len(pred_labels)-1 < len(gold) else 'N/A'}")


    # Calculate macro-averaged precision (KPI)
    # `zero_division=0` means if a class has no predicted samples, its precision is 0.
    # `labels=ALL_CLASSES` ensures all predefined classes are considered.
    macro_precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0, labels=ALL_CLASSES)
    accuracy = accuracy_score(gold_labels, pred_labels)

    print(f"\n--- Validation Metrics ---")
    print(f"KPI (Macro Precision): {macro_precision:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("Detailed Classification Report (may be long for 37 classes, showing sample):")
    # Use `labels=ALL_CLASSES` and `target_names=ALL_CLASSES`
    # The report can be very long for 37 classes, so you might want to summarize it
    # or only print it if a flag is set.
    try:
        report = classification_report(
            gold_labels,
            pred_labels,
            zero_division=0,
            labels=ALL_CLASSES,
            target_names=ALL_CLASSES
        )
        print(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("This can happen if predicted labels contain values not in 'labels' and not in 'gold_labels'.")
        print(f"Unique Gold Labels: {sorted(list(set(gold_labels)))}")
        print(f"Unique Pred Labels: {sorted(list(set(pred_labels)))}")


    print("-------------------------\n")
    return macro_precision

# --- 7. Optimizer Setup and Compilation ---
if __name__ == "__main__":
    # Instantiate the uncompiled module
    uncompiled_classifier = SimpleClassifier(class_definitions_str=CLASS_DEFINITIONS_STR)

    print("--- Testing Uncompiled Classifier (on one dev example) ---")
    if dev_data:
        example_dev_text = dev_data[0].text
        prediction = uncompiled_classifier(example_dev_text)
        print(f"Input: {example_dev_text}")
        print(f"True Class: {dev_data[0].true_class}")
        print(f"Predicted class (uncompiled): {prediction.predicted_class}")

        if hasattr(turbo, 'inspect_history'):
            turbo.inspect_history(n=0)
            uncompiled_classifier(example_dev_text)
            if turbo.history:
                print("\nPrompt sent by UNCOMPILED module for this example:")
                print(turbo.history[0]['prompt'])
            else:
                print("No history found for uncompiled prompt inspection.")
        print("-----------------------------------\n")
    else:
        print("No dev data to test uncompiled classifier.")


    # Set up the optimizer
    # With 37 classes, BootstrapFewShot might benefit from:
    # - More `max_bootstrapped_demos` if your context window allows and you have enough diverse train data.
    # - A larger `max_labeled_demos` pool to choose from.
    optimizer = dspy.BootstrapFewShot(
        metric=precision_accuracy_metric,
        max_bootstrapped_demos=3,  # Try 2-5 demos; more might hit context limits with 37 classes
        max_labeled_demos=16,      # Consider up to this many from training set (e.g., ~0.5 per class avg)
                                  # Ensure this is not much larger than your trainset size.
        # teacher_settings={'lm': turbo} # If teacher is different
    )

    print("--- Starting Compilation (Optimization) ---")
    # Ensure trainset is not empty
    if not train_data:
        print("Error: Training data is empty. Cannot compile.")
        exit()
    if not dev_data:
        print("Error: Development data is empty. Cannot evaluate during compilation.")
        exit()

    compiled_classifier = optimizer.compile(
        student=uncompiled_classifier,
        trainset=train_data,
        valset=dev_data
    )
    print("--- Compilation Finished ---")

    # --- 8. Inspect the Optimized Prompt and Evaluate ---
    print("\n--- Evaluating Compiled Classifier ---")
    if dev_data:
        compiled_classifier.evaluate(dev_data, metric=precision_accuracy_metric, display_progress=True, display_table=0)
    else:
        print("No dev data to evaluate compiled classifier.")


    print("\n--- Optimized Prompt Components ---")
    final_predictor = compiled_classifier.classifier
    print("Optimized Instructions (defined in SimpleClassifier, includes class definitions):")
    print(final_predictor.signature.instructions)

    if final_predictor.demos:
        print(f"\nOptimized Few-Shot Examples ({len(final_predictor.demos)} Demos):")
        for i, demo in enumerate(final_predictor.demos):
            print(f"Demo {i+1}:")
            demo_dict = demo.toDict()
            print(f"  Input ('text'): {demo_dict.get('text', 'N/A')}")
            print(f"  Output ('predicted_class'): {demo_dict.get('predicted_class', 'N/A')}")
    else:
        print("\nNo few-shot examples were selected/generated by the optimizer.")

    print("\n--- Final Prompt Example (for first dev data point if available) ---")
    if dev_data:
        example_input = dev_data[0].text
        true_label_example = dev_data[0].true_class

        if hasattr(turbo, 'inspect_history'):
            turbo.inspect_history(n=0)

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

        print("\n--- Conceptual assembly of the full prompt (for reference) ---")
        # This is a manual reconstruction to show how DSPy builds it
        conceptual_prompt = f"{final_predictor.signature.instructions}\n\n---\n\n"
        for demo in final_predictor.demos:
            demo_dict = demo.toDict()
            conceptual_prompt += f"Text: {demo_dict.get('text', 'N/A')}\n" # Use your input field name
            conceptual_prompt += f"Predicted Class: {demo_dict.get('predicted_class', 'N/A')}\n\n---\n\n" # Use your output field name
        conceptual_prompt += f"Text: {example_input}\n" # Current input
        conceptual_prompt += f"Predicted Class:" # LLM completes from here
        print(conceptual_prompt)
        print("------------------------------------------------------------------")

    else:
        print("No dev data to show final prompt example.")
