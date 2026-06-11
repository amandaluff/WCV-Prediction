##COMBINED DIAGNOSTIC PLOTS FOR ALL MODELS

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# OOF prediction files
oof_files = {
    'Logistic Regression': 'oof_logistic.parquet',
    'XGBoost': 'oof_xgb.parquet',
    'Random Forest': 'oof_rf.parquet'
}

# Saved models for test set
model_files = {
    'Logistic Regression': 'lr_model.pkl',
    'XGBoost': 'xgb_model.pkl',
    'Random Forest': 'rf_model.pkl'
}

# Test data
test_df = pd.read_parquet('wcv_test')
Y_test = test_df['missed']
X_test = test_df.drop(['missed', 'MRN'], axis=1)

# -----------------------------
# ROC CURVES
# -----------------------------
def plot_roc_curves(oof_files, model_files, X_test, Y_test):
    # OOF ROC
    plt.figure(figsize=(8, 6))
    for name, path in oof_files.items():
        df = pd.read_parquet(path)
        fpr, tpr, _ = roc_curve(df['true_label'], df['pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.title('ROC Curve Comparison (Out-of-Fold)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('ROC_OOF.png', dpi=300)
    plt.show()

    # Test ROC
    plt.figure(figsize=(8, 6))
    for name, path in model_files.items():
        model = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.title('ROC Curve Comparison (Test)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('ROC_Test.png', dpi=300)
    plt.show()

# -----------------------------
# PRECISION-RECALL CURVES
# -----------------------------
def plot_pr_curves(oof_files, model_files, X_test, Y_test):
    # OOF PR
    plt.figure(figsize=(8, 6))
    for name, path in oof_files.items():
        df = pd.read_parquet(path)
        precision, recall, _ = precision_recall_curve(df['true_label'], df['pred_proba'])
        ap = average_precision_score(df['true_label'], df['pred_proba'])
        plt.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.2f})')
    #plt.title('Precision-Recall Curve (Out-of-Fold)')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PR_OOF.png', dpi=300)
    plt.show()

    # Test PR
    plt.figure(figsize=(8, 6))
    for name, path in model_files.items():
        model = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, y_proba)
        ap = average_precision_score(Y_test, y_proba)
        plt.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.2f})')
    #plt.title('Precision-Recall Curve (Test)')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PR_Test.png', dpi=300)
    plt.show()

# -----------------------------
# RUN ALL PLOTS
# -----------------------------
if __name__ == "__main__":
    plot_roc_curves(oof_files, model_files, X_test, Y_test)
    plot_pr_curves(oof_files, model_files, X_test, Y_test)

# -----------------------------
# PRINT AVERAGE PRECISION
# -----------------------------
def print_average_precision(oof_files, model_files, X_test, Y_test):
    print("\nAverage Precision Scores:")
    
    # OOF AP
    print("\nOOF:")
    for name, path in oof_files.items():
        df = pd.read_parquet(path)
        ap = average_precision_score(df['true_label'], df['pred_proba'])
        print(f"{name}: {ap:.4f}")
    
    # Test AP
    print("\nTest:")
    for name, path in model_files.items():
        model = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(Y_test, y_proba)
        print(f"{name}: {ap:.4f}")

# -----------------------------
# RUN ALL
# -----------------------------
if __name__ == "__main__":
    plot_roc_curves(oof_files, model_files, X_test, Y_test)
    plot_pr_curves(oof_files, model_files, X_test, Y_test)
    print_average_precision(oof_files, model_files, X_test, Y_test)
