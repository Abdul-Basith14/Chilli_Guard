"""
Scratch Script: test_models.py
Evaluates multiple preprocessing pipelines and classification models on FTIR training set using GroupKFold.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

proc_dir = Path("data/dataset_III_ftir/processed")
X_raw = np.load(proc_dir / "X_raw.npy")
wavenumbers = np.load(proc_dir / "wavenumbers.npy")
df_splits = pd.read_csv(proc_dir / "dataset_III_splits.csv")

train_mask = (df_splits['split'] == 'train').values
val_mask = (df_splits['split'] == 'val').values
test_mask = (df_splits['split'] == 'test').values

X_train = X_raw[train_mask]
y_train = df_splits.loc[train_mask, 'label'].values
groups_train = df_splits.loc[train_mask, 'group_id'].values

X_val = X_raw[val_mask]
y_val = df_splits.loc[val_mask, 'label'].values

X_test = X_raw[test_mask]
y_test = df_splits.loc[test_mask, 'label'].values

print(f"Train samples: {len(X_train)} (Pure: {sum(y_train==0)}, Adult: {sum(y_train==1)}) across {len(set(groups_train))} groups")
print(f"Val samples  : {len(X_val)} (Pure: {sum(y_val==0)}, Adult: {sum(y_val==1)})")
print(f"Test samples : {len(X_test)} (Pure: {sum(y_test==0)}, Adult: {sum(y_test==1)})")

# Preprocessing Transformers
def snv_transform(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

def sg1_transform(X):
    sg = savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=1)
    return snv_transform(sg)

def sg2_transform(X):
    sg = savgol_filter(X, window_length=15, polyorder=2, deriv=2, axis=1)
    return snv_transform(sg)

preprocessors = {
    "Raw + SNV": snv_transform,
    "SG 1st Deriv + SNV": sg1_transform,
    "SG 2nd Deriv + SNV": sg2_transform,
}

classifiers = {
    "SVM (RBF, C=10)": SVC(kernel='rbf', C=10.0, random_state=42),
    "SVM (Linear, C=1.0)": SVC(kernel='linear', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

print("\n=================================================================")
print("  CROSS-VALIDATION RESULTS (GroupKFold n_splits=5 on Train Set)")
print("=================================================================")
gkf = GroupKFold(n_splits=5)

results = []
for prep_name, prep_fn in preprocessors.items():
    X_tr_prep = prep_fn(X_train)
    for clf_name, clf in classifiers.items():
        cv_scores = []
        cv_f1s = []
        for train_idx_cv, val_idx_cv in gkf.split(X_tr_prep, y_train, groups_train):
            X_tr_fold, X_val_fold = X_tr_prep[train_idx_cv], X_tr_prep[val_idx_cv]
            y_tr_fold, y_val_fold = y_train[train_idx_cv], y_train[val_idx_cv]
            
            clf.fit(X_tr_fold, y_tr_fold)
            preds = clf.predict(X_val_fold)
            
            cv_scores.append(accuracy_score(y_val_fold, preds))
            cv_f1s.append(f1_score(y_val_fold, preds, average='macro', zero_division=0))
            
        mean_acc = np.mean(cv_scores)
        mean_f1 = np.mean(cv_f1s)
        results.append({
            "Preprocessor": prep_name,
            "Classifier": clf_name,
            "CV_Acc": mean_acc,
            "CV_Macro_F1": mean_f1,
        })

df_res = pd.DataFrame(results).sort_values(by="CV_Macro_F1", ascending=False)
print("\nTop Models Ranked by Group-CV Macro-F1:")
print(df_res.to_string(index=False))

# Evaluate top model on Validation set and Test set
best_row = df_res.iloc[0]
print(f"\n=================================================================")
print(f"  WINNING MODEL SELECTION: [{best_row['Preprocessor']}] + [{best_row['Classifier']}]")
print(f"=================================================================")

best_prep_fn = preprocessors[best_row['Preprocessor']]
best_clf = classifiers[best_row['Classifier']]

X_tr_best = best_prep_fn(X_train)
X_val_best = best_prep_fn(X_val)
X_test_best = best_prep_fn(X_test)

best_clf.fit(X_tr_best, y_train)

val_preds = best_clf.predict(X_val_best)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
print(f"\nValidation Set Performance:")
print(f"  Val Accuracy : {val_acc*100:.2f}%")
print(f"  Val Macro-F1 : {val_f1:.4f}")

test_preds = best_clf.predict(X_test_best)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
print(f"\nHeld-Out Test Set Performance:")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Macro-F1 : {test_f1:.4f}")
