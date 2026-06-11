# RANDOM FOREST

# Data processing
import pandas as pd
import numpy as np
import random as rn

# Modeling 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

# Hyperparameter tuning
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Model evaluation
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.inspection import permutation_importance

import os
import joblib

###------READ PARQUET-----###
train_df=pd.read_parquet('wcv_train')

######################################################################################

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)
rn.seed(seed)

# Make separate data set for Y and X
Y_df = train_df['missed']
X_df = train_df.drop(['missed'], axis=1)

# Build the pipeline
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced'))
])

# Define parameter grid
rf_param_grid = {
    'rf__max_depth': [10, 20, 50, 70, 80, 100, 150],
    'rf__max_features': [2, 4, 6, 'sqrt', 'log2'],
    'rf__min_samples_leaf': [2, 5, 7, 10, 15],
    'rf__min_samples_split': [2, 5, 7, 10, 12, 15, 18, 20],
    'rf__n_estimators': [100, 200, 500, 1000, 1500, 2000]
}

# Set up GroupKFold
gkf = GroupKFold(n_splits=5)
groups = X_df['MRN']
X_model = X_df.drop(columns=['MRN']) # Drop MRN from features for modeling

# Set up RandomizedSearchCV
rf_grid_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=rf_param_grid,
    n_iter=300,
    cv=gkf,
    n_jobs=-1,
    random_state=42,
    verbose=3,
    scoring='average_precision' #options: accuracy, recall, precision, average_precision, balanced_accuracy, f1, roc_auc
)

# Fit with groups
rf_grid_search.fit(X_model, Y_df, groups=groups)
rf_best_grid = rf_grid_search.best_estimator_

#Uncomment to display best hyperparameters
print(rf_grid_search.best_params_)

##SAVE MODEL
joblib.dump(rf_best_grid, "rf_model.pkl")

###-----OUT-OF-FOLD PERFORMANCE-----###
# Get out-of-fold predicted probabilities
y_pred_proba = cross_val_predict(
    rf_best_grid, 
    X_model,
    Y_df,
    groups=groups,
    cv=gkf,
    method='predict_proba',
    n_jobs=-1
)

yscore = y_pred_proba[:, 1]
fpr, tpr, thresh = roc_curve(Y_df, yscore)
roc_auc = auc(fpr, tpr)

####EXPORT OOF SCORES###
oof_df = pd.DataFrame({
    'MRN': groups,
    'true_label': Y_df,
    'pred_proba': yscore
})
oof_df.to_parquet("oof_rf.parquet", index=False)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'OOF Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('Random_Forest_MOF.png')

# For confusion matrix and other metrics
y_hat = cross_val_predict(
    rf_best_grid,
    X_model,
    Y_df,
    groups=groups,
    cv=gkf,
    method='predict',
    n_jobs=-1
)

###-----OOF PERFORMANCE-----###
cm = confusion_matrix(Y_df, y_hat)
tn, fp, fn, tp = cm.ravel()

# Metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
accuracy    = (tp + tn) / (tp + tn + fp + fn)
ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

print("\n--- Out-of-Fold Performance ---")
print(f"{'Metric':<15}{'Value':>10}")
print("-" * 25)
print(f"{'Sensitivity':<15}{sensitivity:>10.3f}")
print(f"{'Specificity':<15}{specificity:>10.3f}")
print(f"{'Accuracy':<15}{accuracy:>10.3f}")
print(f"{'PPV':<15}{ppv:>10.3f}")
print(f"{'NPV':<15}{npv:>10.3f}")

###-----HOLD OUT SET PERFORMANCE-----###
# Drop MRN from test features
test_df=pd.read_parquet('wcv_test')
Y_test = test_df['missed']
X_test = test_df.drop(['missed', 'MRN'], axis=1)

# Predict probabilities and labels
y_test_proba = rf_best_grid.predict_proba(X_test)[:, 1]
y_test_pred  = rf_best_grid.predict(X_test)

# ROC Curve and AUC
fpr_test, tpr_test, _ = roc_curve(Y_test, y_test_proba)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend(loc='lower right')   # fixed from pltplt.legend
plt.show()

print(f"Test ROC AUC: {roc_auc_test:.4f}")

###-----TEST SET PERFORMANCE-----###
cm_test = confusion_matrix(Y_test, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()

# Metrics
sensitivity_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
specificity_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
accuracy_test    = (tp + tn) / (tp + tn + fp + fn)
ppv_test         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  
npv_test         = tn / (tn + fn) if (tn + fn) > 0 else 0.0 

print("\n--- Test Set Performance ---")
print(f"{'Metric':<15}{'Value':>10}")
print("-" * 25)
print(f"{'Sensitivity':<15}{sensitivity_test:>10.3f}")
print(f"{'Specificity':<15}{specificity_test:>10.3f}")
print(f"{'Accuracy':<15}{accuracy_test:>10.3f}")
print(f"{'PPV':<15}{ppv_test:>10.3f}")
print(f"{'NPV':<15}{npv_test:>10.3f}")

###-----GINI IMPORTANCE-----###
importances = rf_best_grid.named_steps['rf'].feature_importances_

feature_names = X_model.columns.tolist()

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

###-----PERMUTATION IMPORTANCE-----###
result = permutation_importance(
    rf_best_grid,         
    X_model, 
    Y_df, 
    n_repeats=30,     # number of shuffles per feature
    random_state=42,
    scoring='average_precision'         # same metric as grid search
)

feature_names = X_model.columns.tolist()

# Build DataFrame for permutation importance
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

print(perm_importance_df)

# Plot top 20 permutation importance
plt.figure(figsize=(10, 6))
perm_importance_df.head(20).plot(kind='barh', x='Feature', y='Importance Mean', legend=False)
plt.title('Top 20 Permutation Importances')
plt.gca().invert_yaxis()
plt.show()

# Plot top 20 features importances
plt.figure(figsize=(10, 6))
importance_df.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Evaluate predicted probabilities distribution
plt.figure(figsize=(10, 6))
sns.histplot(yscore, bins=50, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities for Positive Class')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


###### F1 SCORE VS CLASSIFICATION THRESHOLD (OOF)
# OOF probabilities
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []

for t in thresholds:
    y_pred_t = (yscore >= t).astype(int)    # OOF predictions at threshold t
    tn, fp, fn, tp = confusion_matrix(Y_df, y_pred_t).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f1_scores.append(f1)

# Find best threshold (OOF)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# Plot F1 vs threshold (OOF)
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, color='tab:blue', lw=2)
plt.axvline(best_threshold, color='tab:red', linestyle='--',
            label=f'Best Threshold = {best_threshold:.2f} (F1 = {best_f1:.3f})')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Classification Threshold (OOF)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best OOF Threshold: {best_threshold:.3f}, Best OOF F1 Score: {best_f1:.3f}")

#### DIAGNOSTICS TABLE BY THRESHOLD (OOF)

# Define thresholds to compare (OOF)
thresholds_to_compare = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9]
results = []

for t in thresholds_to_compare:
    y_pred_t = (yscore >= t).astype(int)    # OOF predictions at threshold t
    tn, fp, fn, tp = confusion_matrix(Y_df, y_pred_t).ravel()
    
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1           = (2 * ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    results.append([t, sensitivity, specificity, balanced_acc, ppv, npv, f1])

# Create DataFrame (OOF)
df_compare_oof = pd.DataFrame(
    results,
    columns=['Threshold', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'PPV', 'NPV', 'F1']
)
print("\n--- OOF Metrics at Different Thresholds ---")
print(df_compare_oof.to_string(index=False, float_format="%.3f"))

out_path = "thresholds_oof.xlsx"
if os.path.exists(out_path):
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_compare_oof.to_excel(writer, sheet_name="rf_oof", index=False)
else:
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        df_compare_oof.to_excel(writer, sheet_name="rf_oof", index=False)



###### F1 SCORE VS CLASSIFICATION THRESHOLD (TEST / HOLDOUT)

# TEST probabilities: y_test_proba (shape: [n_samples,]) and TEST labels: Y_test
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores_test = []

for t in thresholds:
    y_pred_t = (y_test_proba >= t).astype(int)    # TEST predictions at threshold t
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_t).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    f1_scores_test.append(f1)

# Find best threshold (TEST)
best_idx_test       = np.argmax(f1_scores_test)
best_threshold_test = thresholds[best_idx_test]
best_f1_test        = f1_scores_test[best_idx_test]

# Plot F1 vs threshold (TEST)
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores_test, color='tab:green', lw=2)
plt.axvline(best_threshold_test, color='tab:red', linestyle='--',
            label=f'Best Threshold = {best_threshold_test:.2f} (F1 = {best_f1_test:.3f})')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Classification Threshold (Test / Holdout)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best TEST Threshold: {best_threshold_test:.3f}, Best TEST F1 Score: {best_f1_test:.3f}")

#### DIAGNOSTICS TABLE BY THRESHOLD (TEST / HOLDOUT)

# Define thresholds to compare (TEST)
thresholds_to_compare = [ 0.10, 0.20, 0.24, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9]
results_test = []

for t in thresholds_to_compare:
    y_pred_t = (y_test_proba >= t).astype(int)                 # TEST predictions at threshold t
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_t).ravel()
    
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1           = (2 * ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    results_test.append([t, sensitivity, specificity, balanced_acc, ppv, npv, f1])

# Create DataFrame (TEST)
df_compare_test = pd.DataFrame(
    results_test,
    columns=['Threshold', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'PPV', 'NPV', 'F1']
)
print("\n--- TEST Metrics at Different Thresholds ---")
print(df_compare_test.to_string(index=False, float_format="%.3f"))

# Write to the same workbook 
if os.path.exists(out_path):
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_compare_test.to_excel(writer, sheet_name="rf_test", index=False)
else:
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        df_compare_test.to_excel(writer, sheet_name="rf_test", index=False)

# --- SHAP Beeswarm Plots for Random Forest 
rf_model = rf_best_grid.named_steps['rf']
feature_names = X_model.columns.tolist()

# Created a masker because the normal explainer wasnt working for RF
masker = shap.maskers.Independent(X_model, max_samples=100)
explainer = shap.Explainer(rf_model, masker)

def save_beeswarm_new_api(X, fname, max_display=30, sample_size=1000):
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    shap_values = explainer(X_sample)
    shap_values_pos = shap_values[:, :, 1]
    tmp = shap.Explanation(shap_values_pos.values,
                           base_values=shap_values_pos.base_values,
                           data=X_sample.values,
                           feature_names=feature_names)
    
    # Plot beeswarm
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(tmp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved SHAP beeswarm plot to {fname}")

# OOF and Test
save_beeswarm_new_api(X_model, "rf_shap_beeswarm_oof.png")
save_beeswarm_new_api(X_test, "rf_shap_beeswarm_test.png")