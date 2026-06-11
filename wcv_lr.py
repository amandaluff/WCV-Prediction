# LOGISTIC REGRESSION

#Load the necessary packages
# Data processing
import pandas as pd
import numpy as np
import random as rn

# Modeling 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

#save model
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

# Build the pipeline: tofloat -> scaler -> SMOTE -> LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        solver='liblinear', 
        class_weight='balanced',             # handle imbalance
        max_iter=1000,
        random_state=42
    ))
])

# Define parameter grid for Logistic Regression
lr_param_grid = {
    'lr__C': [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 3, 10],
    'lr__penalty': ['l2', 'l1']
}

# Set up GroupKFold
gkf = GroupKFold(n_splits=5)
groups = X_df['MRN']
X_model = X_df.drop(columns=['MRN']) # Drop MRN from features for modeling

# Set up RandomizedSearchCV
lr_grid_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=lr_param_grid,
    n_iter=50,  # logistic grid smaller than this - will stop at full grid
    cv=gkf,
    n_jobs=-1,
    random_state=42,
    verbose=3,
    scoring='average_precision'
)

# Fit with groups
lr_grid_search.fit(X_model, Y_df, groups=groups)
lr_best_grid = lr_grid_search.best_estimator_
print(lr_grid_search.best_params_)

###SAVE MODEL
joblib.dump(lr_best_grid, "lr_model.pkl")

###-----OUT-OF-FOLD PERFORMANCE-----###
# Get out-of-fold predicted probabilities
y_pred_proba = cross_val_predict(
    lr_best_grid, 
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
oof_df.to_parquet("oof_logistic.parquet", index=False)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'OOF Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('Logistic_Regression_OOF.png')

# For confusion matrix and other metrics
y_hat = cross_val_predict(
    lr_best_grid,
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
y_test_proba = lr_best_grid.predict_proba(X_test)[:, 1]
y_test_pred  = lr_best_grid.predict(X_test)

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
plt.legend(loc='lower right')
plt.show()

print(f"Test ROC AUC: {roc_auc_test:.4f}")

###-----TEST SET PERFORMANCE-----###
cm_test = confusion_matrix(Y_test, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()

# Metrics
sensitivity_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
specificity_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
accuracy_test    = (tp + tn) / (tp + tn + fp + fn)
ppv_test         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
npv_test         = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

print("\n--- Test Set Performance ---")
print(f"{'Metric':<15}{'Value':>10}")
print("-" * 25)
print(f"{'Sensitivity':<15}{sensitivity_test:>10.3f}")
print(f"{'Specificity':<15}{specificity_test:>10.3f}")
print(f"{'Accuracy':<15}{accuracy_test:>10.3f}")
print(f"{'PPV':<15}{ppv_test:>10.3f}")
print(f"{'NPV':<15}{npv_test:>10.3f}")

###-----COEFFICIENT IMPORTANCE-----###
# Coefficients as importance
lr = lr_best_grid.named_steps['lr']
coef = lr.coef_.ravel()          # signed effect on log-odds
feature_names = X_model.columns.tolist()

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coef': coef,
    'AbsCoef': np.abs(coef)
}).sort_values(by='AbsCoef', ascending=False)

print(coef_df)

ax = plt.gca()
coef_df.head(20).plot(kind='barh', x='Feature', y='AbsCoef', legend=False, ax=ax)
plt.title('Top 20 Features by |Coefficient| (Logistic Regression)')
plt.gca().invert_yaxis()
plt.show()

###-----PERMUTATION IMPORTANCE-----###
result = permutation_importance(
    lr_best_grid,
    X_model,
    Y_df,
    n_repeats=30,
    random_state=42,
    scoring='average_precision'
)

feature_names = X_model.columns.tolist()

# Build DataFrame for permutation importance
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

print(perm_importance_df)

ax = plt.gca()
perm_importance_df.head(20).plot(kind='barh', x='Feature', y='Importance Mean', legend=False, ax=ax)
plt.title('Top 20 Permutation Importances (Logistic Regression)')
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

thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []

for t in thresholds:
    y_pred_t = (yscore >= t).astype(int) # OOF predictions at threshold t
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
thresholds_to_compare = [0.10, 0.20, 0.30, 0.40, 0.50, 0.58, 0.6, 0.7, 0.8, 0.9]
results = []

for t in thresholds_to_compare:
    y_pred_t = (yscore >= t).astype(int) 
    tn, fp, fn, tp = confusion_matrix(Y_df, y_pred_t).ravel()
    
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0  
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
        df_compare_oof.to_excel(writer, sheet_name="lr_oof", index=False)
else:
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        df_compare_oof.to_excel(writer, sheet_name="lr_oof", index=False)



###### F1 SCORE VS CLASSIFICATION THRESHOLD (TEST / HOLDOUT)

# TEST probabilities
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores_test = []

for t in thresholds:
    y_pred_t = (y_test_proba >= t).astype(int)
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
thresholds_to_compare = [0.10, 0.20, 0.30, 0.40, 0.50, 0.57, 0.6, 0.7, 0.8, 0.9]
results_test = []

for t in thresholds_to_compare:
    y_pred_t = (y_test_proba >= t).astype(int) 
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_t).ravel()
    
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0
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
        df_compare_test.to_excel(writer, sheet_name="lr_test", index=False)
else:
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        df_compare_test.to_excel(writer, sheet_name="lr_test", index=False)


# --- SHAP beeswarm for regularized Logistic Regression: OOF and Test, nonzero coefficients only ---

# Extract fitted model 
lr = lr_best_grid.named_steps['lr']
scaler = lr_best_grid.named_steps['scaler']

# Coefficients and feature names
feature_names = X_model.columns.tolist()
coef = lr.coef_.ravel()

# Nonzero coefficient mask
coef_tol = 0.0
nz_mask = np.abs(coef) > coef_tol
nz_idx = np.where(nz_mask)[0]
nz_features = [feature_names[i] for i in nz_idx]

# Transform OOF and Test features
X_model_scaled = scaler.transform(X_model)
X_test_scaled = scaler.transform(X_test)

# Wrap in DataFrames
X_model_scaled_df = pd.DataFrame(X_model_scaled, columns=feature_names, index=X_model.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

# Background sample 
bg_size = min(2000, len(X_model_scaled_df))
background = shap.sample(X_model_scaled_df, bg_size, random_state=42)

explainer = shap.LinearExplainer(lr, background)

def _compute_shap(X_scaled_df):
    vals = explainer.shap_values(X_scaled_df)
    return vals, False

def _subset_to_nonzero(values, X_scaled_df):
    """Subset SHAP values and X to the nonzero coefficient features."""
    if isinstance(values, shap.Explanation):
        return values[:, nz_idx], X_scaled_df[nz_features]
    else:
        return values[:, nz_idx], X_scaled_df[nz_features]

def save_beeswarm_lr(X_scaled_df, fname, max_display=30):
    """Compute SHAP values and save a beeswarm limited to nonzero features."""
    values, _ = _compute_shap(X_scaled_df)
    values_nz, X_nz = _subset_to_nonzero(values, X_scaled_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        values_nz,
        X_nz,
        show=False,
        max_display=min(max_display, len(nz_features))
        )
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved SHAP beeswarm (nonzero coefficients) to {fname}")

# OOF beeswarm
save_beeswarm_lr(X_model_scaled_df, "shap_beeswarm_lr_oof.png", max_display=30)

# Test beeswarm
save_beeswarm_lr(X_test_scaled_df, "shap_beeswarm_lr_test.png", max_display=30)

