import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import random
import os

# --- 1. Configuration for Gemini 1.5 Pro via GCP Vertex AI ---
# IMPORTANT: Replace these with your actual GCP project ID and region.
# Ensure your environment is authenticated to GCP (e.g., via `gcloud auth application-default login`)

GCP_PROJECT_ID = "your-gcp-project-id"  # e.g., "my-awesome-project-12345"
VERTEX_LOCATION = "us-central1"       # e.g., "us-central1", "europe-west1"
GEMINI_MODEL_NAME = "gemini-1.5-pro-preview-0514" # Check Google's docs for the latest stable model name

# Configure Dspy to use Google Vertex AI
# dspy.configure sets the default language model for all dspy.Predict calls
print(f"Configuring Dspy with Gemini 1.5 Pro ({GEMINI_MODEL_NAME}) on Vertex AI in {VERTEX_LOCATION} for project {GCP_PROJECT_ID}...")
dspy.configure(
    lm=dspy.Google(
        model=GEMINI_MODEL_NAME,
        project_id=GCP_PROJECT_ID,
        location=VERTEX_LOCATION
    )
)
print("Dspy configuration complete.")


# --- 2. Data Preparation ---
# Generate dummy data for 40 classes, 30,000 documents
def generate_dummy_data(num_classes=40, docs_per_class=750): # 40 * 750 = 30,000 documents
    data = []
    class_definitions_map = {}
    
    for i in range(num_classes):
        class_name = f"Category_{chr(65+i) if i < 26 else chr(65+i-26)}{i}" # A, B, ..., Z, AA, BB...
        definition = f"This class '{class_name}' pertains to the domain of {random.choice(['science', 'history', 'technology', 'art', 'finance', 'nature', 'health', 'sports', 'politics', 'education'])} focusing on specific aspects related to item {i*100}, concept '{class_name.lower()}_theory', and applications in {random.choice(['industry', 'academia', 'daily life'])}. It also includes topics such as {random.choice(['research', 'development', 'analysis', 'synthesis'])} of {class_name.lower()} materials."
        class_definitions_map[class_name] = definition
        
        for j in range(docs_per_class):
            doc_text = f"Document {j+1} for {class_name}. It heavily discusses aspects of '{class_name.lower()}_theory', {random.choice(['recent findings', 'historical context', 'future implications'])} in {definition.split('domain of ')[1].split(' focusing')[0].strip()}, and how it impacts {definition.split('applications in ')[1].split('.')[0].strip()}. Keywords: {class_name.lower()}, {random.choice(['data', 'model', 'system', 'theory'])}, {random.choice(['analysis', 'prediction', 'management'])}. This text is clearly related to the core definition of '{class_name}'."
            data.append({
                'document_text': doc_text,
                'ground_truth_label': class_name
            })
            
    random.shuffle(data) # Shuffle the data
    return pd.DataFrame(data), class_definitions_map

print("Generating dummy data...")
df, all_class_definitions = generate_dummy_data(num_classes=40, docs_per_class=750)
all_class_names = list(all_class_definitions.keys())

# Split data into training and test sets
# Using a smaller dev set for Dspy optimization to save time/cost.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth_label'])
# Use a smaller subset of training data for Dspy's internal optimization process
# A sample of 500-1000 is often sufficient for BootstrapFewShot.
dev_df_for_optimizer = train_df.sample(n=min(1000, len(train_df)), random_state=42)

# Convert dataframes to Dspy Example objects
train_examples = [
    dspy.Example(document_text=row['document_text'], ground_truth_label=row['ground_truth_label']).with_fields(['document_text', 'ground_truth_label'])
    for index, row in train_df.iterrows()
]
test_examples = [
    dspy.Example(document_text=row['document_text'], ground_truth_label=row['ground_truth_label']).with_fields(['document_text', 'ground_truth_label'])
    for index, row in test_df.iterrows()
]
dev_examples_for_optimizer = [
    dspy.Example(document_text=row['document_text'], ground_truth_label=row['ground_truth_label']).with_fields(['document_text', 'ground_truth_label'])
    for index, row in dev_df_for_optimizer.iterrows()
]

print(f"Total documents: {len(df)}")
print(f"Number of training examples: {len(train_examples)}")
print(f"Number of test examples: {len(test_examples)}")
print(f"Number of development examples for optimizer: {len(dev_examples_for_optimizer)}")
print(f"Number of classes: {len(all_class_names)}")


# --- 3. Dspy Module Definition ---
# Helper to format class definitions for the prompt
def format_class_definitions(definitions_map):
    formatted_str = "Here are the definitions for each class:\n"
    for class_name, definition in definitions_map.items():
        formatted_str += f"- **{class_name}**: {definition}\n"
    return formatted_str

class_definitions_prompt_str = format_class_definitions(all_class_definitions)

class MulticlassDocumentClassifier(dspy.Module):
    def __init__(self, all_class_names, all_class_definitions_str):
        super().__init__()
        self.all_class_names = all_class_names
        self.all_class_definitions_str = all_class_definitions_str # Pre-formatted string

        # Define the basic classification signature.
        # We pass the full class definitions and valid labels string dynamically.
        self.classify = dspy.Predict(
            dspy.Signature(
                "document_text, class_definitions, valid_labels -> ground_truth_label",
                "Classifies a given document into one of the specified categories based on its content and provided definitions."
            )
        )

    def forward(self, document_text):
        # The class_definitions_str is already pre-formatted and big, passed directly.
        # The valid_labels string is also built dynamically for clarity and strictness.
        valid_labels_prompt = f"Choose ONLY ONE of the following valid categories: {', '.join(self.all_class_names)}." \
                              f"Your final answer MUST be one of these exact category names, with no extra text or explanations." \
                              f"If the document does not clearly fit any category, select the MOST RELEVANT one. Do NOT output a new category."

        prediction = self.classify(
            document_text=document_text,
            class_definitions=self.all_class_definitions_str,
            valid_labels=valid_labels_prompt
        )
        
        # Robustly clean the predicted label. LLMs sometimes add prefixes/suffixes.
        # This cleaning logic is crucial for accurate metric calculation.
        raw_prediction = prediction.ground_truth_label
        cleaned_prediction = raw_prediction.strip()
        
        # Remove common prefixes/suffixes, case-insensitively
        prefixes_to_remove = ["label:", "category:", "predicted label:", "classification:", "the category is", "the correct category is", "i classify this as"]
        for prefix in prefixes_to_remove:
            if cleaned_prediction.lower().startswith(prefix.lower()):
                cleaned_prediction = cleaned_prediction[len(prefix):].strip()
                break # Remove only one prefix
        
        # Remove trailing punctuation and quotes
        cleaned_prediction = cleaned_prediction.rstrip('.,;"\'').strip()

        # Attempt to find an exact match in the valid class names (case-sensitive first, then case-insensitive)
        if cleaned_prediction in self.all_class_names:
            return cleaned_prediction
        
        for class_name in self.all_class_names:
            if cleaned_prediction.lower() == class_name.lower():
                return class_name # Return the exact case-sensitive class name
        
        # More flexible matching for partial or slightly off predictions
        # This is a heuristic and can sometimes lead to incorrect assignments.
        # Only use if you're willing to trade strictness for finding a match.
        for class_name in self.all_class_names:
            if class_name.lower() in cleaned_prediction.lower():
                # If the predicted string contains a valid class name, and isn't too long after it
                # Example: "This is Category_A" -> "Category_A"
                # You might need to adjust the length threshold (e.g., +20 chars)
                if len(cleaned_prediction) - len(class_name) < 20: # Heuristic for brevity
                    return class_name
        
        # If after all cleaning, it's still not a valid class name, return it as is.
        # The metric will then mark it as incorrect.
        # You could also map it to a specific "INVALID_PREDICTION" placeholder here.
        # For strictness, let's return it as is.
        return cleaned_prediction


# Instantiate the classifier with pre-formatted definitions
my_classifier = MulticlassDocumentClassifier(all_class_names, class_definitions_prompt_str)


# --- 4. Metric Definition ---
def dspy_metric(example, prediction, trace=None):
    gold_label = example.ground_truth_label
    predicted_label = prediction # The forward method already returns the cleaned label string
    
    # Crucially, check if the predicted_label is one of the valid class names.
    # This ensures that hallucinated or improperly formatted labels are counted as incorrect.
    if predicted_label not in all_class_names:
        # print(f"Warning: Predicted label '{predicted_label}' is not a valid class name for document: '{example.document_text[:50]}...' (Gold: {gold_label})")
        return False # Penalize predictions that are not in the valid set
        
    return gold_label == predicted_label # Exact match accuracy


# --- 5. Optimize the Prompt with Dspy ---
# Compile the program. Dspy will try to optimize the prompt (e.g., add few-shot examples)
# based on the dev_examples_for_optimizer and the dspy_metric.

print("\nStarting prompt optimization (BootstrapFewShot)...")
# Note: For 40 classes, few-shot examples can quickly consume context.
# max_bootstrapped_demos=1 means it will try to find one good example to prepend to the prompt.
# If your context window is very large, you could try 2 or 3.
# max_labeled_demos: The maximum number of labeled examples to sample from the `trainset`
#                    to consider as potential few-shot demonstrations.
# teacher_lm/student_lm: Use the same configured LLM for both.
optimizer = BootstrapFewShot(
    metric=dspy_metric,
    max_bootstrapped_demos=1, # Number of few-shot examples to include in the prompt
    max_labeled_demos=5,      # Max training examples to consider for each demo
    teacher_lm=dspy.settings.lm,
    trainset=dev_examples_for_optimizer
)

optimized_classifier = optimizer.compile(my_classifier, trainset=dev_examples_for_optimizer)
print("Prompt optimization complete!")

# You can inspect the optimized program's demonstrations:
if hasattr(optimized_classifier.classify, 'demos') and optimized_classifier.classify.demos:
    print("\nOptimized Few-Shot Demos:")
    for demo in optimized_classifier.classify.demos:
        print(f"  Input: {demo.document_text[:80]}...")
        print(f"  Output: {demo.ground_truth_label}")
else:
    print("\nNo few-shot demonstrations were added by the optimizer (or demos attribute not found).")


# --- 6. Evaluate the Optimized Prompt ---
print("\nEvaluating optimized classifier on the test set...")

evaluator = Evaluate(
    prog=optimized_classifier,
    metric=dspy_metric,
    devset=test_examples, # Evaluate on the held-out test set
    num_threads=1,        # Adjust for parallel processing, be mindful of API rate limits
    display_progress=True,
    display_table=0       # Set to 0 to avoid printing individual prediction tables
)

test_accuracy_dspy_eval = evaluator.evaluate()

print(f"\nFinal Test Set Accuracy (via Dspy Evaluate): {test_accuracy_dspy_eval}")

# For a more detailed scikit-learn classification report
y_true = []
y_pred = []

print("\nRunning full prediction on test set for detailed scikit-learn report...")
# It's important to use the *optimized* classifier for this
for i, example in enumerate(test_examples):
    try:
        # Pass document_text to the forward method
        predicted_label_cleaned = optimized_classifier(document_text=example.document_text)
        y_true.append(example.ground_truth_label)
        y_pred.append(predicted_label_cleaned)
    except Exception as e:
        print(f"Error predicting for example {i}: {e}")
        y_true.append(example.ground_truth_label)
        y_pred.append("ERROR_PREDICTION") # Mark as an error to see if it impacts overall metrics

# Ensure y_pred contains only valid labels for scikit-learn's classification_report
# Or, if you want to see how often it predicts invalid labels, make them part of your 'labels'
y_pred_for_report = [p if p in all_class_names else 'INVALID_PREDICTION' for p in y_pred]

# Dynamically add 'INVALID_PREDICTION' to labels if it appeared
report_labels = list(all_class_names)
if 'INVALID_PREDICTION' in y_pred_for_report:
    report_labels.append('INVALID_PREDICTION')
report_labels = sorted(list(set(report_labels))) # Ensure unique and sorted for consistency

print("\n--- Scikit-learn Classification Report ---")
try:
    # Use labels parameter to ensure all known classes are in the report,
    # even if no instances were predicted for them.
    # And include 'INVALID_PREDICTION' if it occurred.
    print(classification_report(y_true, y_pred_for_report, labels=report_labels, zero_division=0))
except ValueError as e:
    print(f"Error generating classification report: {e}")
    print("This usually happens if `y_true` or `y_pred` contain labels not in `labels` argument.")
    print(f"Unique y_true labels: {set(y_true)}")
    print(f"Unique y_pred labels (for report): {set(y_pred_for_report)}")
    print(f"Labels used in report: {set(report_labels)}")

print(f"Overall Accuracy (sklearn, using cleaned predictions): {accuracy_score(y_true, y_pred_for_report)}")

# --- Optional: Save and Load the Optimized Program ---
# This allows you to use the optimized prompt without re-running optimization
# optimized_classifier.save("multiclass_classifier_optimized.dspy")
# print("\nOptimized classifier saved to multiclass_classifier_optimized.dspy")

# To load:
# loaded_classifier = MulticlassDocumentClassifier(all_class_names, class_definitions_prompt_str)
# loaded_classifier.load("multiclass_classifier_optimized.dspy")
# print("Optimized classifier loaded.")

# --- Example Inference with the Optimized Classifier ---
print("\n--- Example Inference with Optimized Classifier ---")
example_document = df.sample(1, random_state=random.randint(0, 1000)).iloc[0]
print(f"Document for inference: {example_document['document_text'][:200]}...")
print(f"True Label: {example_document['ground_truth_label']}")

predicted_label = optimized_classifier(document_text=example_document['document_text'])
print(f"Predicted Label: {predicted_label}")
