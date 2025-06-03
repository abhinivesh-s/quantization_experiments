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
# --- Metric Function - Adaptable to Single or Multiple Examples ---
metric_call_count = 0
def precision_accuracy_metric(
    gold: Union[dspy.Example, List[dspy.Example]], # Can be a single Example or a List
    preds: Union[dspy.Prediction, List[dspy.Prediction]], # Can be a single Prediction or a List
    trace=None
) -> float:
    global metric_call_count
    metric_call_count += 1
    current_call_id = f"MetricCall-{metric_call_count}"
    print(f"\n--- {current_call_id} --- ENTER --- Trace: {trace is not None} ---")

    # --- Normalize inputs to always be lists ---
    # This is a key change to handle both single and list inputs.
    if isinstance(gold, dspy.Example):
        gold_list = [gold]
        print(f"  {current_call_id}: Received single dspy.Example for 'gold'. Converted to list.")
    elif isinstance(gold, list):
        gold_list = gold
    else:
        print(f"  {current_call_id}: ERROR - 'gold' is of unexpected type: {type(gold)}. Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Invalid Gold Type) ---")
        return 0.0

    if isinstance(preds, dspy.Prediction):
        preds_list = [preds]
        print(f"  {current_call_id}: Received single dspy.Prediction for 'preds'. Converted to list.")
    elif isinstance(preds, list):
        preds_list = preds
    else:
        print(f"  {current_call_id}: ERROR - 'preds' is of unexpected type: {type(preds)}. Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Invalid Preds Type) ---")
        return 0.0
    
    print(f"  {current_call_id}: Normalized Gold length: {len(gold_list)}, Normalized Preds length: {len(preds_list)}")

    # --- Initial Checks (using normalized lists) ---
    if not gold_list:
        print(f"  {current_call_id}: WARNING - Normalized `gold_list` is empty. Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Empty Gold List) ---")
        return 0.0
    
    # Check for the "string 'text' as gold[0]" specific issue, if it's still a concern
    # This check should be against gold_list[0] now
    if isinstance(gold_list[0], str):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  {current_call_id}: METRIC DETECTED UNEXPECTED `gold_list[0]` TYPE: {type(gold_list[0])}, VALUE: '{gold_list[0]}'")
        if gold_list[0] == "text" and len(gold_list) == 1:
            print("    This matches a suspected DSPy issue. Returning 0.0.")
            print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Suspected DSPy String Issue) ---")
            return 0.0
        else:
            print("    `gold_list[0]` is a string, but not the specific 'text' case. Returning 0.0.")
            print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Unexpected String in Gold) ---")
            return 0.0
    
    if not isinstance(gold_list[0], dspy.Example):
        print(f"  {current_call_id}: ERROR - `gold_list[0]` is not dspy.Example (Type: {type(gold_list[0])}). Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Malformed Gold after String Check) ---")
        return 0.0

    if len(gold_list) != len(preds_list):
        print(f"  {current_call_id}: CRITICAL - LENGTH MISMATCH! Gold: {len(gold_list)}, Preds: {len(preds_list)}. Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Length Mismatch) ---")
        return 0.0

    # --- Gold Label Extraction ---
    gold_labels = []
    try:
        for i, g in enumerate(gold_list):
            if not isinstance(g, dspy.Example) or not hasattr(g, 'true_class') or \
               not isinstance(g.true_class, str) or not g.true_class.strip():
                print(f"  {current_call_id}: ERROR - Malformed dspy.Example in gold_list at index {i}. Returning 0.0")
                print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Bad Gold Item) ---")
                return 0.0
            gold_labels.append(g.true_class)
    except Exception as e_gold_proc:
        print(f"  {current_call_id}: CRITICAL ERROR processing gold labels: {e_gold_proc}")
        traceback.print_exc()
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Gold Processing Error) ---")
        return 0.0

    # --- Prediction Label Extraction & Normalization ---
    pred_labels = []
    for i, p in enumerate(preds_list):
        if not isinstance(p, dspy.Prediction) or not hasattr(p, 'predicted_class') or p.predicted_class is None:
            pred_labels.append("INVALID_PRED_" + random.choice(ALL_CLASSES)) # Placeholder
            continue
        raw_pred = str(p.predicted_class).strip()
        if not raw_pred: pred_labels.append("EMPTY_PRED_" + random.choice(ALL_CLASSES)); continue
        best_match = raw_pred
        if raw_pred not in ALL_CLASSES: best_match = next((kc for kc in ALL_CLASSES if kc in raw_pred), raw_pred)
        pred_labels.append(best_match)
    
    # --- Scikit-learn Calculation ---
    if not gold_labels:
        print(f"  {current_call_id}: WARNING - gold_labels list became empty. Returning 0.0")
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Empty Gold Labels) ---")
        return 0.0
        
    final_score = 0.0
    try:
        # When len(gold_labels) == 1, macro precision might behave poorly or give warnings.
        # sklearn's precision_score with average='macro' on a single sample:
        # If correct: P=1 for that class, 0 for others. Macro P = 1/N_classes.
        # If incorrect: P=0 for true class, P=0 for predicted (if different & no other TP/FP), 0 for others. Macro P = 0.
        # So, for a single item, it's essentially 1/N_classes if correct, 0 if incorrect.
        # This might not be the ideal "score" for a single instance if optimizers expect values closer to 1.
        
        if len(gold_labels) == 1:
            print(f"  {current_call_id}: Calculating score for a single example.")
            # For a single example, macro precision is 1/num_classes if correct, 0 if incorrect.
            # Or, simply use accuracy for a single item.
            is_correct = (gold_labels[0] == pred_labels[0])
            single_item_score = 1.0 if is_correct else 0.0
            final_score = single_item_score # Using simple accuracy for single item scoring
            print(f"    Single item: Gold='{gold_labels[0]}', Pred='{pred_labels[0]}'. Correct: {is_correct}. Score: {final_score:.4f}")
        else:
            # Normal calculation for lists
            macro_precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0, labels=ALL_CLASSES)
            final_score = macro_precision
            if final_score == 0.0:
                print(f"  {current_call_id}: >>> KPI (Macro Precision for list) calculated as 0.0 <<<")

        if trace is None or metric_call_count % 1 == 0 or len(gold_labels) > 1: # Log more for lists or final
             if len(gold_labels) > 1: # Only print accuracy if we have multiple items for it to make sense
                accuracy = accuracy_score(gold_labels, pred_labels)
                print(f"  {current_call_id}: Scores (list) - Macro Precision: {final_score:.4f}, Accuracy: {accuracy:.4f}")
            # if final_score == 0.0 and len(gold_labels) > 1 and (trace is None):
            #     print("    Classification Report (when list KPI is 0.0):")
            #     print(classification_report(gold_labels, pred_labels, zero_division=0, labels=ALL_CLASSES, target_names=ALL_CLASSES))
            
    except ValueError as ve: # sklearn error
        print(f"  {current_call_id}: ERROR during scikit-learn (ValueError): {ve}")
        traceback.print_exc()
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Sklearn ValueError) ---")
        return 0.0
    except Exception as e_sklearn: # other sklearn errors
        print(f"  {current_call_id}: UNEXPECTED ERROR during scikit-learn: {e_sklearn}")
        traceback.print_exc()
        print(f"--- {current_call_id} --- EXITING WITH SCORE: 0.0 (Sklearn Other Error) ---")
        return 0.0

    print(f"--- {current_call_id} --- EXITING WITH SCORE: {final_score:.4f} ---")
    return final_score
# --- End Metric Function ---

# --- End Metric Function ---


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- YOUR PERFECTED DATA LOADING AND SANITY CHECKS FOR train_data and dev_data HERE ---
    # ... (ensure train_data and dev_data are correctly populated lists of dspy.Example)
    # Example placeholder:
    train_df = pd.DataFrame({'text': [f"t{i}" for i in range(5)], 'RCC': [random.choice(ALL_CLASSES) for _ in range(5)]})
    dev_df = pd.DataFrame({'text': [f"d{i}" for i in range(3)], 'RCC': [random.choice(ALL_CLASSES) for _ in range(3)]})
    train_data = [dspy.Example(text=row['text'], true_class=row['RCC']).with_inputs('text') for _, row in train_df.iterrows()]
    dev_data = [dspy.Example(text=row['text'], true_class=row['RCC']).with_inputs('text') for _, row in dev_df.iterrows()]
    # --- END DATA LOADING ---

    if not train_data or not dev_data:
        print("CRITICAL: train_data or dev_data is empty. Exiting.")
        exit()

    # 1. Create an instance of your UNCOMPILED module
    uncompiled_classifier = SimpleClassifier(class_definitions_str=CLASS_DEFINITIONS_STR)

    # 2. OPTIONAL: Evaluate the UNCOMPILED module using dspy.evaluate.Evaluate
    # This gives you a baseline.
    if dev_data:
        print("\n--- Evaluating UNCOMPILED Classifier (using dspy.evaluate.Evaluate) ---")
        from dspy.evaluate import Evaluate # Make sure to import
        evaluator_for_uncompiled = Evaluate(
            devset=dev_data,
            metric=precision_accuracy_metric,
            num_threads=1, # Adjust as needed
            display_progress=True,
            display_table=0
        )
        uncompiled_score = evaluator_for_uncompiled(uncompiled_classifier)
        print(f"Score for UNCOMPILED classifier: {uncompiled_score}")
        print("----------------------------------------------------------------------")


    # 3. Set up the optimizer
    optimizer = dspy.BootstrapFewShot(
        metric=precision_accuracy_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
    )

    print("\n--- Starting Compilation (BootstrapFewShot) ---")
    print(f"Length of dev_data (used as 'gold' in metric for full eval): {len(dev_data)}")
    try:
        # 4. COMPILE the module
        compiled_classifier = optimizer.compile(
            student=uncompiled_classifier, # Pass the uncompiled instance
            trainset=train_data
        )
        print("--- Compilation Finished ---")
    except Exception as e:
        print(f"\n!!! ERROR DURING optimizer.compile() !!!")
        # ... (your existing error handling) ...
        exit("Halting due to compilation error.")

    # 5. Evaluate the COMPILED module using ITS OWN .evaluate() method
    print("\n--- Evaluating COMPILED Classifier (using compiled_classifier.evaluate()) ---")
    if dev_data and compiled_classifier: # Ensure compilation was successful
        # The 'compiled_classifier' object returned by optimizer.compile() HAS an .evaluate() method
        compiled_score_dict = compiled_classifier.evaluate(
            dev_data, # The dataset to evaluate on
            metric=precision_accuracy_metric,
            display_progress=True,
            display_table=5 # Show some examples of passes/fails
        )
        # compiled_classifier.evaluate() often returns a dictionary of scores or detailed results
        # The exact return type can vary slightly or might just print.
        # The primary effect is that your metric function is called and its prints are shown.
        # If your metric returns a float, that's usually the main score.
        # print(f"Evaluation results from compiled_classifier.evaluate(): {compiled_score_dict}")
    elif not compiled_classifier:
        print("Skipping evaluation of compiled classifier because compilation failed.")
    else:
        print("No dev_data to evaluate the compiled classifier.")

    # --- Inspecting Optimized Prompt Components (on compiled_classifier) ---
    if compiled_classifier:
        print("\n--- Optimized Prompt Components (from compiled_classifier) ---")
        # Access the underlying predictor, which might be nested if you have a complex module.
        # For SimpleClassifier, it's straightforward:
        final_predictor = compiled_classifier.classifier # Or however you access the dspy.Predict instance
        
        print("Optimized Instructions (potentially modified if your optimizer changes them):")
        # For BootstrapFewShot, instructions usually remain as defined unless your module structure allows modification
        # or if using an optimizer like MIPRO.
        print(final_predictor.extended_signature.instructions if hasattr(final_predictor, 'extended_signature') else final_predictor.signature.instructions)


        if final_predictor.demos:
            print(f"\nOptimized Few-Shot Examples ({len(final_predictor.demos)} Demos):")
            for i, demo in enumerate(final_predictor.demos):
                print(f"Demo {i+1}: {demo.toDict()}")
        else:
            print("\nNo few-shot examples were selected/generated by the optimizer for the final predictor.")
    else:
        print("Skipping prompt inspection because compilation failed.")
