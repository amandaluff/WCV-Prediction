import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

TEST_PATH = "wcv_test"
DEMOG_PATH = "wcv_demo"

MODEL_PATHS = {
    "lr": "lr_model.pkl",
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl",
}

SUBGROUP_COLS = ["race_ethnicity", "language", "medicaid"]

# Load and score
test_df = pd.read_parquet(TEST_PATH)
demog_df = pd.read_parquet(DEMOG_PATH)

X_test = test_df.drop(["MRN", "missed"], axis=1)
pred_df = test_df[["MRN", "missed", "medicaid"]].copy()

for name, path in MODEL_PATHS.items():
    model = joblib.load(path)
    pred_df[f"prob_{name}"] = model.predict_proba(X_test)[:, 1]

demog_merge = demog_df[["MRN", "race", "ethnicity", "language"]].drop_duplicates(
    subset=["MRN"]
)
df = pred_df.merge(demog_merge, on="MRN", how="left")

race_eth_map = {
    "White": "NH White",
    "Black": "NH Black",
    "Asian": "NH Asian",
    "Another Race": "Another NH Race",
    "Not Reported": "NH Not Reported",
}


def assign_race_eth(row):
    if row["ethnicity"] == "Hispanic or Latino":
        return "Hispanic"
    return race_eth_map.get(row["race"], "NH Not Reported")


df["race_ethnicity"] = df.apply(assign_race_eth, axis=1)


# Bootstrap within each subgroup
def bootstrap_subgroup(y_true, y_proba, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    if len(np.unique(y_true)) < 2:
        return {"AUC": "—", "AveragePrecision": "—"}

    point_auc = roc_auc_score(y_true, y_proba)
    point_ap = average_precision_score(y_true, y_proba)

    boot_auc, boot_ap = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_auc.append(roc_auc_score(y_true[idx], y_proba[idx]))
        boot_ap.append(average_precision_score(y_true[idx], y_proba[idx]))

    if len(boot_auc) < 100:
        return {
            "AUC": f"{point_auc:.4f} (unstable)",
            "AveragePrecision": f"{point_ap:.4f} (unstable)",
        }

    return {
        "AUC": (
            f"{point_auc:.4f} "
            f"({np.percentile(boot_auc, 100 * alpha / 2):.4f}, "
            f"{np.percentile(boot_auc, 100 * (1 - alpha / 2)):.4f})"
        ),
        "AveragePrecision": (
            f"{point_ap:.4f} "
            f"({np.percentile(boot_ap, 100 * alpha / 2):.4f}, "
            f"{np.percentile(boot_ap, 100 * (1 - alpha / 2)):.4f})"
        ),
    }


results = []
for name in MODEL_PATHS:
    for sg in SUBGROUP_COLS:
        for level, sub_df in df.groupby(sg):
            metrics = bootstrap_subgroup(
                sub_df["missed"].values, sub_df[f"prob_{name}"].values
            )
            results.append(
                {
                    "model": name,
                    "group_var": sg,
                    "group": level,
                    "n": len(sub_df),
                    "AUC": metrics["AUC"],
                    "AveragePrecision": metrics["AveragePrecision"],
                }
            )

fairness_table = pd.DataFrame(results)
fairness_table = fairness_table[
    ["model", "group_var", "group", "n", "AUC", "AveragePrecision"]
]
fairness_table.to_csv("subgroup_performance_table.csv", index=False)
print(fairness_table.to_string(index=False))
