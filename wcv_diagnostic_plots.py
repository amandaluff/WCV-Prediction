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
# PERFORMANCE TABLES WITH 95% CIs
# -----------------------------
from sklearn.metrics import roc_auc_score

METRIC_ORDER = [
    'AUC', 'Average Precision', 'Accuracy', 'Balanced Accuracy',
    'Sensitivity (Recall)', 'Specificity', 'PPV (Precision)', 'NPV', 'F1'
]

def compute_metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec    = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv     = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv     = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    bal_acc = (sens + spec) / 2
    acc     = (tp + tn) / (tp + tn + fp + fn)

    return {
        'AUC':                  roc_auc_score(y_true, y_proba),
        'Average Precision':    average_precision_score(y_true, y_proba),
        'Accuracy':             acc,
        'Balanced Accuracy':    bal_acc,
        'Sensitivity (Recall)': sens,
        'Specificity':          spec,
        'PPV (Precision)':      ppv,
        'NPV':                  npv,
        'F1':                   f1
    }

def find_best_f1_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    return thresholds[np.argmax(f1)]


def bootstrap_metrics(y_true, y_proba, threshold, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    boot_records = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_records.append(
            compute_metrics_at_threshold(y_true[idx], y_proba[idx], threshold)
        )

    boot_df = pd.DataFrame(boot_records)
    point   = compute_metrics_at_threshold(y_true, y_proba, threshold)

    row = {}
    for m in METRIC_ORDER:
        lo = boot_df[m].quantile(0.025)
        hi = boot_df[m].quantile(0.975)
        row[m] = f"{point[m]:.4f} ({lo:.4f}, {hi:.4f})"
    row['Threshold'] = f"{threshold:.4f}"
    return row


def build_performance_table(oof_files, model_files, X_test, Y_test,
                            use_f1_threshold=True, n_boot=2000):
    rows = []
    index_tuples = []

    # Internal validation (OOF)
    for name, path in oof_files.items():
        df = pd.read_parquet(path)
        y_true  = df['true_label'].values
        y_proba = df['pred_proba'].values

        thresh = find_best_f1_threshold(y_true, y_proba) if use_f1_threshold else 0.5
        rows.append(bootstrap_metrics(y_true, y_proba, thresh, n_boot))
        index_tuples.append(
            ('Internal Validation (Practice A, Out-of-Fold)', name)
        )

    # External validation (Test)
    for name, path in model_files.items():
        model   = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_true  = Y_test.values

        thresh = find_best_f1_threshold(y_true, y_proba) if use_f1_threshold else 0.5
        rows.append(bootstrap_metrics(y_true, y_proba, thresh, n_boot))
        index_tuples.append(('External Validation (Practice B)', name))

    col_order = ['Threshold'] + METRIC_ORDER
    result = pd.DataFrame(rows, columns=col_order)
    result.index = pd.MultiIndex.from_tuples(
        index_tuples, names=['Validation Set', 'Model']
    )
    return result

# -----------------------------
# RUN ALL
# -----------------------------

if __name__ == "__main__":
    plot_roc_curves(oof_files, model_files, X_test, Y_test)
    plot_pr_curves(oof_files, model_files, X_test, Y_test)
    print_average_precision(oof_files, model_files, X_test, Y_test)

    # Table 1: F1-maximizing threshold
    table_f1 = build_performance_table(
        oof_files, model_files, X_test, Y_test,
        use_f1_threshold=True, n_boot=2000
    )
    print("\n" + "=" * 130)
    print("Table 1: Performance at F1-Maximizing Threshold (95% CI)")
    print("=" * 130)
    print(table_f1.to_string())
    table_f1.to_csv('performance_f1_threshold.csv')

    # Table 2: Fixed 0.5 threshold
    table_50 = build_performance_table(
        oof_files, model_files, X_test, Y_test,
        use_f1_threshold=False, n_boot=2000
    )
    print("\n" + "=" * 130)
    print("Table 2: Performance at 0.5 Threshold (95% CI)")
    print("=" * 130)
    print(table_50.to_string())
    table_50.to_csv('performance_05_threshold.csv')
