from sklearn.metrics import classification_report
import pandas as pd

# Dummy data: holdout missing class '1'
y_val_true = [0, 1, 1, 0]
y_val_pred = [0, 1, 0, 0]

y_holdout_true = [0, 0, 0, 0]
y_holdout_pred = [0, 0, 0, 0]

# Generate reports
val_report = pd.DataFrame(classification_report(y_val_true, y_val_pred, output_dict=True)).transpose()
holdout_report = pd.DataFrame(classification_report(y_holdout_true, y_holdout_pred, output_dict=True)).transpose()

# Add suffixes
val_report = val_report.add_suffix('_val')
holdout_report = holdout_report.add_suffix('_holdout')

# Robust merge using outer join to preserve all classes
combined = pd.merge(val_report, holdout_report, left_index=True, right_index=True, how='outer')

# Reset index and rename for clarity
combined.reset_index(inplace=True)
combined.rename(columns={'index': 'class'}, inplace=True)

# Optional: fill missing values for cleaner display
combined.fillna(0, inplace=True)  # or use `np.nan` if you prefer to indicate missing data explicitly

# Optional: round metrics
combined = combined.round(3)

print(combined)
