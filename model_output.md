Introduction - hello

This document details the output format of the deployed AI model. The model is a LinearSVC classifier, wrapped in a Calibrated Classifier, that takes TF-IDF embeddings of text documents as input and outputs predicted probabilities for each of the 37 classes, along with the single most likely class prediction.

Model Description

Base Model: LinearSVC (Support Vector Classifier)

Calibration Method: Calibrated Classifier (likely using isotonic or sigmoid calibration)

Input: TF-IDF embeddings of a text document

Output:

Predicted probabilities for each of the 37 classes.

The class with the highest predicted probability (the top predicted class).

Number of Classes: 37

Output Format

The model's output for a single input document will be presented as follows:

Predicted Class: This indicates the class that the model predicts as the most likely for the input document. This is the class corresponding to the highest predicted probability.

Class Probabilities: This provides the predicted probability for each of the 37 classes. The probabilities are presented in a tabular format for clarity. The sum of all probabilities for a given document should ideally be close to 1.0.

Example Output

Below is an example of how the model's output might appear for a hypothetical input document.

Input Document (Hypothetical): [Insert a brief description or excerpt of the hypothetical input document here, if helpful for context]

Predicted Class: Class_X (Replace 'Class_X' with the actual predicted class name)

Class Probabilities:

Class Name	Predicted Probability
Class_1	0.012
Class_2	0.005
Class_3	0.028
...	...
Class_X	0.850
...	...
Class_35	0.003
Class_36	0.015
Class_37	0.010

Note: The table above is illustrative. The actual class names and probabilities will vary based on the input document and the model's predictions.

Interpretation of Output

The "Predicted Class" is the model's single best guess for the document's category.

The "Predicted Probability" for each class indicates the model's confidence that the document belongs to that specific class. A higher probability suggests greater confidence.

The calibration step helps to ensure that the predicted probabilities are more reliable and better reflect the true likelihood of the document belonging to a particular class. For example, a predicted probability of 0.85 for a class suggests that in 85% of cases where the model predicts this class with a probability of 0.85, the document truly belongs to that class.

Further Considerations

Class Names: Ensure the "Class Name" column in the table uses the exact names or identifiers for your 37 classes.

Probability Formatting: The probabilities are typically represented as decimal values between 0.0 and 1.0.

Confidence Thresholds: Depending on the application, you might establish confidence thresholds for using the model's predictions. For instance, you might only accept predictions where the top probability is above a certain value.

Error Handling: (Optional, but good to mention) Briefly describe any error handling mechanisms or what the output might look like if the model encounters an issue with the input (e.g., inability to process the input).

Conclusion

This document outlines the standard output format for the Calibrated LinearSVC model. Understanding this format is crucial for interpreting the model's predictions and utilizing them effectively.

How to incorporate this into a Word Document:

Create a New Word Document: Start with a blank Word document.

Add Title: Type the document title at the top, e.g., "AI Model Output Documentation - Calibrated LinearSVC (TF-IDF)".

Sections: Use headings (like the bolded section titles in the example) to structure the document.

Text: Copy and paste the descriptive text from the example.

Table:

In the "Example Output" section, click where you want the table to appear.

Go to the "Insert" tab in Word.

Click "Table".

Select "Insert Table...".

Set the "Number of columns" to 2.

Set the "Number of rows" to 38 (1 header row + 37 rows for classes).

Click "OK".

Enter the column headers: "Class Name" and "Predicted Probability".

Fill in the table with hypothetical class names and probabilities, ensuring to highlight the probability for the predicted class as shown in the example.

Formatting: Use bolding for section titles and table headers. Consider using a consistent font and spacing.

Save: Save your Word document.

This structured approach makes it easy for anyone reviewing the document to understand the model's output at a glance. Remember to replace the placeholder information (like "Class_X" and the hypothetical probabilities) with details relevant to your specific model and classes.




************************************************************************************************

# **Model Output Documentation**

## **Model Description**

This model is a **Linear Support Vector Classifier (LinearSVC)** wrapped in **CalibratedClassifierCV** to enable probability estimates. It is trained on **TF-IDF embeddings** of documents for multiclass classification.

* **Input:** TF-IDF vectorized representation of a document.
* **Output:**

  * Predicted class label (the one with the highest probability)
  * Class-wise probability distribution over 37 possible classes

## **Classes**

The model predicts one of **37 predefined classes**, each representing a specific category relevant to the business problem. The classes are indexed or named (e.g., `class_0`, `class_1`, ..., `class_36`) depending on the training label format.

---

## **Sample Output**

| Document ID | Predicted Class | Top 5 Classes with Probabilities                                                      |
| ----------- | --------------- | ------------------------------------------------------------------------------------- |
| doc\_001    | class\_12       | class\_12 (0.41), class\_5 (0.22), class\_9 (0.13), class\_3 (0.10), class\_17 (0.05) |
| doc\_002    | class\_3        | class\_3 (0.62), class\_1 (0.15), class\_7 (0.08), class\_8 (0.06), class\_22 (0.03)  |
| doc\_003    | class\_7        | class\_7 (0.35), class\_12 (0.30), class\_0 (0.11), class\_4 (0.10), class\_18 (0.07) |

> **Note:** The predicted class is the one with the **highest probability**.

---

## **Technical Notes**

* **Probability Calibration:** Since `LinearSVC` does not natively support probabilities, `CalibratedClassifierCV` is used to provide reliable probability estimates.
* **Top-K Interpretation:** Often, it’s useful to consider the top-N predicted classes for downstream tasks or human review. The above table shows Top-5 probabilities for illustration.

## **Output Format (as Python dict or JSON)**

If exporting output programmatically, the result per document may look like:

```json
{
  "document_id": "doc_001",
  "predicted_class": "class_12",
  "probabilities": {
    "class_0": 0.01,
    "class_1": 0.00,
    ...
    "class_12": 0.41,
    ...
    "class_36": 0.00
  }
}
```

---

Would you like me to create a downloadable Word (.docx) version of this document?




### **Model Evaluation Reports**

After generating predictions, the model is evaluated using several standard classification metrics **calculated per class**. These metrics provide insight into the performance of the classifier across the 37 classes.

#### **1. Precision**

* **Definition:** Precision measures the proportion of predicted positive instances that are actually correct.
* **Formula:**

  $$
  \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
  $$
* **Interpretation:** High precision means fewer false positives.

#### **2. Recall (Sensitivity)**

* **Definition:** Recall measures the proportion of actual positives that are correctly identified.
* **Formula:**

  $$
  \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  $$
* **Interpretation:** High recall means fewer false negatives.

#### **3. F1 Score**

* **Definition:** The F1 Score is the harmonic mean of precision and recall.
* **Formula:**

  $$
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$
* **Interpretation:** F1 balances precision and recall, especially useful when classes are imbalanced.

#### **4. Accuracy**

* **Definition:** Accuracy is the proportion of total predictions that were correct.
* **Formula (per class):**

  $$
  \text{Accuracy}_{\text{class}} = \frac{\text{TP + TN}}{\text{TP + FP + FN + TN}}
  $$
* **Interpretation:** Overall correctness. Less informative for imbalanced data unless reported per class.

#### **5. Support**

* **Definition:** Support is the number of true instances for each class in the reference (true) labels.
* **Interpretation:** Indicates how many examples were available to evaluate a class, which helps understand reliability.

---

You would typically display these metrics in a table like the one below:

| Class     | Precision | Recall | F1 Score | Accuracy | Support |
| --------- | --------- | ------ | -------- | -------- | ------- |
| class\_0  | 0.85      | 0.78   | 0.81     | 0.97     | 120     |
| class\_1  | 0.65      | 0.72   | 0.68     | 0.94     | 95      |
| ...       | ...       | ...    | ...      | ...      | ...     |
| class\_36 | 0.90      | 0.87   | 0.88     | 0.99     | 108     |

You can also calculate:

* **Macro average:** Unweighted average over classes.
* **Weighted average:** Average weighted by support (more useful with class imbalance).
* **Micro average:** Global metrics by aggregating TP, FP, FN across all classes.

Would you like help generating this table programmatically from your model predictions?



Certainly! Here’s a Word-friendly version of the model evaluation metric definitions. You can copy-paste this directly into your document with clean formatting:

---

### **Model Evaluation Reports**

After generating predictions, the model is evaluated using standard classification metrics for each of the 37 classes. These include:

---

**1. Precision**

* **Definition:** Precision measures the proportion of predicted positive instances that are actually correct.
* **Formula:**
  `Precision = True Positives / (True Positives + False Positives)`
* **Interpretation:** High precision means that when the model predicts a class, it's usually correct.

---

**2. Recall (Sensitivity)**

* **Definition:** Recall measures the proportion of actual positive instances that the model correctly identified.
* **Formula:**
  `Recall = True Positives / (True Positives + False Negatives)`
* **Interpretation:** High recall means the model captures most of the actual instances of a class.

---

**3. F1 Score**

* **Definition:** The F1 Score is the harmonic mean of precision and recall, balancing both metrics.
* **Formula:**
  `F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`
* **Interpretation:** Useful when you need a balance between precision and recall, especially with imbalanced classes.

---

**4. Accuracy (Per Class)**

* **Definition:** Accuracy measures the proportion of all predictions that were correct, per class.
* **Formula:**
  `Accuracy = (True Positives + True Negatives) / (Total Predictions)`
* **Interpretation:** Represents overall correctness but may be misleading in the case of class imbalance.

---

**5. Support**

* **Definition:** Support is the number of actual instances of each class in the dataset.
* **Interpretation:** Indicates how many examples of a class were available to evaluate the model's performance. Higher support generally means more reliable metrics.

---

You can also compute **macro**, **micro**, and **weighted** averages of these metrics:

* **Macro Average:** Unweighted mean of the metric across all classes.
* **Weighted Average:** Average weighted by the number of instances (support) per class.
* **Micro Average:** Aggregates contributions of all classes to compute a global average.



The output probability array from the model corresponds to the order of classes as returned by the model object's .classes_ attribute. Each index in the probability vector aligns with the class label at the same position in model.classes_.




