# scripts/combine_labels.py

import pandas as pd

# Paths to input CSVs
ham_labels = pd.read_csv("datasets/labels.csv")
isic_labels = pd.read_csv("datasets/isic_labels.csv")

# Combine both
combined = pd.concat([ham_labels, isic_labels], ignore_index=True)

# Sanity check
print("HAM count:", len(ham_labels))
print("ISIC count:", len(isic_labels))
print("Total combined:", len(combined))
print("Label distribution:\n", combined['dx'].value_counts())

# Save
combined.to_csv("datasets/combined_labels.csv", index=False)
print("✅ Saved as datasets/combined_labels.csv")
