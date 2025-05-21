import dspy
from dspy.teleprompt import BootstrapFewShot, SignatureOptimizer
from dspy.evaluate import Evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import random
import os

# --- 1. Configuration ---
# Set your API key for the LLM.
# Make sure to set this environment variable or replace "sk-YOUR_OPENAI_API_KEY"
# For OpenAI
# os.environ["OPENAI_API_KEY"] = "sk-YOUR_OPENAI_KEY"
# dspy.configure(lm=dspy.OpenAI(model="gpt-4o"))

# For Google (Gemini) - requires a custom wrapper if not directly supported by Dspy's core
# As of my last update, Dspy's direct Gemini integration might require a custom wrapper.
# If you have a custom dspy.LM for Gemini, you'd configure it here.
# For example:
# class GeminiLM(dspy.LM):
#     def __init__(self, model_name="gemini-pro"):
#         super().__init__()
#         import google.generativeai as genai
#         genai.configure(api_key="YOUR_GEMINI_API_KEY") # Ensure this is your actual API key
#         self.model = genai.GenerativeModel(model_name)
#         self.history = []

#     def _create_completion(self, prompt, **kwargs):
#         response = self.model.generate_content(prompt)
#         return response.text

#     def __call__(self, prompt, **kwargs):
#         return self._create_completion(prompt, **kwargs)

# dspy.configure(lm=GeminiLM(model_name="gemini-pro"))

# FOR DEMONSTRATION PURPOSES, I'll use OpenAI as it's directly integrated.
# PLEASE REPLACE WITH YOUR ACTUAL API KEY AND MODEL.
dspy.configure(lm=dspy.OpenAI(model="gpt-4o", api_key="sk-YOUR_OPENAI_API_KEY"))


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
dev_df_for_optimizer = train_df.sample(n=min(1000, len(train_df)), random_state=42) # Max 1000 examples for optimization

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
        # The valid_labels string is also built dynamically for clarity.
        valid_labels_prompt = f"Choose ONLY ONE of the following valid categories: {', '.join(self.all_class_names)}." \
                              f"Do NOT include any other text or explanations in your final answer."

        prediction = self.classify(
            document_text=document_text,
            class_definitions=self.all_class_definitions_str,
            valid_labels=valid_labels_prompt
        )
        
        # Robustly clean the predicted label. LLMs sometimes add prefixes/suffixes.
        # This cleaning logic is crucial for accurate metric calculation.
        raw_prediction = prediction.ground_truth_label
        cleaned_prediction = raw_prediction.strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["label:", "category:", "predicted label:", "classification:"]
        for prefix in prefixes_to_remove:
            if cleaned_prediction.lower().startswith(prefix):
                cleaned_prediction = cleaned_prediction[len(prefix):].strip()
                break # Remove only one prefix
        
        # Remove trailing punctuation and quotes
        cleaned_prediction = cleaned_prediction.rstrip('.,;"\'').strip()

        # If the LLM returns something like "The category is Class_X", extract "Class_X"
        for class_name in self.all_class_names:
            if class_name.lower() in cleaned_prediction.lower() and len(cleaned_prediction.lower()) < len(class_name.lower()) + 20: # prevent matching within longer strings
                # This is a heuristic. More complex regex might be needed for very messy outputs.
                return class_name # Return the exact class name
                
        # As a last resort, check if the raw cleaned prediction is an exact match
        if cleaned_prediction in self.all_class_names:
            return cleaned_prediction

        # If after all cleaning, it's still not a valid class name, try to find the closest match (optional)
        # Or, just return it as is, and the metric will mark it as incorrect.
        # For simplicity and strictness, we'll return it as is, letting the metric fail.
        # A more advanced approach might map to 'UNCATEGORIZED' or 'INVALID_PREDICTION'
        return cleaned_prediction


# Instantiate the classifier with pre-formatted definitions
my_classifier = MulticlassDocumentClassifier(all_class_names, class_definitions_prompt_str)


# --- 4. Metric Definition ---
def dspy_metric(example, prediction, trace=None):
    gold_label = example.ground_truth_label
    predicted_label = prediction # The forward method already returns the cleaned label string
    
    # Check if the predicted_label is one of the valid class names.
    # This ensures that hallucinated labels are counted as incorrect.
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
# if hasattr(optimized_classifier.classify, 'demos') and optimized_classifier.classify.demos:
#     print("\nOptimized Few-Shot Demos:")
#     for demo in optimized_classifier.classify.demos:
#         print(f"  Input: {demo.document_text[:80]}...")
#         print(f"  Output: {demo.ground_truth_label}")
# else:
#     print("\nNo few-shot demonstrations were added by the optimizer (or demos attribute not found).")


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
        y_pred.append("ERROR_PREDICTION") # Mark as an error

# Ensure y_pred contains only valid labels for scikit-learn's classification_report
# Or, if you want to see how often it predicts invalid labels, make them part of your 'labels'
unique_pred_labels = set(y_pred)
combined_labels_for_report = sorted(list(set(all_class_names) | unique_pred_labels)) # Include all potential labels

print("\n--- Scikit-learn Classification Report ---")
try:
    print(classification_report(y_true, y_pred, labels=all_class_names, zero_division=0))
except ValueError as e:
    print(f"Error generating classification report: {e}")
    print("This usually happens if `y_true` or `y_pred` contain labels not in `labels` argument.")
    print(f"Unique y_true labels: {set(y_true)}")
    print(f"Unique y_pred labels: {set(y_pred)}")
    print(f"Expected labels (all_class_names): {set(all_class_names)}")

print(f"Overall Accuracy (sklearn): {accuracy_score(y_true, y_pred)}")

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
