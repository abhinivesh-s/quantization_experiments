Got it! Here's the **Markdown-formatted version** of the earlier response explaining **weighted accuracy** in multiclass classification:

---

## 🎯 Weighted Accuracy in Multiclass Classification

**Weighted accuracy** adjusts for class imbalance by assigning weights to each class — usually based on how many samples there are (support). It gives a better view of overall model performance when classes are imbalanced.

---

### 📘 Formula

\[
\text{Weighted Accuracy} = \sum_{i=1}^{C} w_i \cdot \text{Accuracy}_i
\]

Where:
- \( C \) = number of classes  
- \( w_i \) = weight for class \( i \), typically \( \frac{n_i}{N} \), where \( n_i \) is the number of true samples in class \( i \), and \( N \) is the total number of samples  
- \( \text{Accuracy}_i \) = accuracy for class \( i \): proportion of correctly classified samples among all true samples of class \( i \)

Alternatively (simpler):

\[
\text{Weighted Accuracy} = \frac{\sum_{i=1}^{C} \text{TP}_i}{\sum_{i=1}^{C} n_i}
\]

Where:
- \( \text{TP}_i \) = true positives for class \( i \) (correct predictions)
- \( n_i \) = total true samples in class \( i \)

---

### 🛠️ In Practice (Using `sklearn`)

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Standard accuracy
accuracy_score(y_true, y_pred)

# Balanced accuracy (equal weight per class)
balanced_accuracy_score(y_true, y_pred)
```

### ✅ To compute support-weighted accuracy manually:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
correct_per_class = np.diag(cm)
support_per_class = cm.sum(axis=1)

weighted_accuracy = np.sum(correct_per_class) / np.sum(support_per_class)
```

---

### 🧠 Notes

- `accuracy_score`: raw accuracy (correct / total)
- `balanced_accuracy_score`: average of per-class recall (equal weight per class)
- Manual weighted accuracy: accounts for actual class distribution (weighted by support)

---

Let me know if you want help adapting this to your dataset or metrics DataFrame!
