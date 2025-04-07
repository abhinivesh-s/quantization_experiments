Absolutely! Here's the full explanation of **weighted accuracy in multiclass classification**, formatted for **GitHub Markdown** (no LaTeX, fully GitHub-compatible):

---

## 🎯 Weighted Accuracy in Multiclass Classification

**Weighted accuracy** adjusts for class imbalance by assigning weights to each class — usually based on how many samples are in each class (i.e., class support). It gives a more realistic view of model performance when the data is imbalanced.

---

### 📘 Formula

```
Weighted Accuracy = Σ (wᵢ × Accuracyᵢ)

Where:
  C           = number of classes
  wᵢ          = weight for class i (typically nᵢ / N)
  Accuracyᵢ   = accuracy for class i (correct predictions / true samples)
  nᵢ          = number of true samples in class i
  N           = total number of samples across all classes
```

Alternatively, you can compute it as:

```
Weighted Accuracy = Total correct predictions / Total samples
                  = Σ TPᵢ / Σ nᵢ
```

Where:
- TPᵢ is the number of correctly predicted samples for class i  
- nᵢ is the number of true samples in class i

---

### 🛠️ In Practice (with scikit-learn)

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Regular accuracy
accuracy = accuracy_score(y_true, y_pred)

# Balanced accuracy (equal weight per class)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
```

---

### ✅ To compute support-weighted accuracy manually:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
correct_per_class = np.diag(cm)         # Diagonal = correctly predicted samples per class
support_per_class = cm.sum(axis=1)      # Total true samples per class

weighted_accuracy = np.sum(correct_per_class) / np.sum(support_per_class)
```

---

### 🧠 Notes

- `accuracy_score`: raw accuracy (correct predictions / total samples)
- `balanced_accuracy_score`: average of per-class recall (all classes weighted equally)
- Manual weighted accuracy: lets you account for true class distribution (support-based)

---

Let me know if you'd like a version of this for binary classification, or want it embedded in a notebook/script!
