# Subgroup performance assessment for fairness evaluation
# Computes AUC and average precision by race/ethnicity, language, and insurance
# for logistic regression, random forest, and XGBoost in the external validation set

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from fairlearn.metrics import MetricFrame

# Paths and settings
TEST_PATH = "wcv_test"
DEMOG_PATH = "wcv_demo"

MODEL_PATHS = {
    "lr": "lr_model.pkl",
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl"
}

SUBGROUP_COLS = ["race_ethnicity", "language", "medicaid"]

# Load data
test_df = pd.read_parquet(TEST_PATH)
demog_df = pd.read_parquet(DEMOG_PATH)

# Score models
X_test = test_df.drop(["MRN", "missed"], axis=1)
pred_df = test_df[["MRN", "missed", "medicaid"]].copy()

for name, path in MODEL_PATHS.items():
    model = joblib.load(path)
    pred_df[f"prob_{name}"] = model.predict_proba(X_test)[:, 1]

# Merge demographics
demog_merge = demog_df[["MRN", "race", "ethnicity", "language"]].drop_duplicates(subset=["MRN"])
df = pred_df.merge(demog_merge, on="MRN", how="left")

# Create combined race/ethnicity variable
race_eth_map = {
    "White": "NH White",
    "Black": "NH Black",
    "Asian": "NH Asian",
    "Another Race": "Another NH Race",
    "Not Reported": "NH Not Reported"
}

def assign_race_eth(row):
    if row["ethnicity"] == "Hispanic or Latino":
        return "Hispanic"
    return race_eth_map.get(row["race"], "NH Not Reported")

df["race_ethnicity"] = df.apply(assign_race_eth, axis=1)

# Define metrics
def auc_metric(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

def ap_metric(y_true, y_prob):
    return average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

def n_obs(y_true, y_prob):
    return len(y_true)

METRICS = {"n": n_obs, "AUC": auc_metric, "AveragePrecision": ap_metric}

# Run subgroup analysis
results = []
for name in MODEL_PATHS:
    for sg in SUBGROUP_COLS:
        mf = MetricFrame(metrics=METRICS, y_true=df["missed"].values,
                         y_pred=df[f"prob_{name}"].values, sensitive_features=df[sg].values)
        out = mf.by_group.reset_index()
        out = out.rename(columns={out.columns"group"})
        out["model"] = name
        out["group_var"] = sg
        results.append(out)

fairness_table = pd.concat(results, ignore_index=True)
fairness_table = fairness_table[["model", "group_var", "group", "n", "AUC", "AveragePrecision"]]
fairness_table["n"] = fairness_table["n"].astype(int)
fairness_table["AUC"] = fairness_table["AUC"].round(4)
fairness_table["AveragePrecision"] = fairness_table["AveragePrecision"].round(4)
fairness_table.to_csv("subgroup_performance_table.csv", index=False)