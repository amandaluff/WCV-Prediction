# RANDOM FOREST

import pandas as pd
import numpy as np
import random as rn
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# -----------------------------
# DATA
# -----------------------------
seed = 123
np.random.seed(seed)
rn.seed(seed)

train_df = pd.read_parquet('wcv_train')
Y_df = train_df['missed']
X_df = train_df.drop(['missed'], axis=1)

groups = X_df['MRN']
X_model = X_df.drop(columns=['MRN'])

test_df = pd.read_parquet('wcv_test')
Y_test = test_df['missed']
X_test = test_df.drop(['missed', 'MRN'], axis=1)

# -----------------------------
# PIPELINE + TUNING
# -----------------------------
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced'))
])

rf_param_grid = {
    'rf__max_depth': [10, 20, 50, 70, 80, 100, 150],
    'rf__max_features': [2, 4, 6, 'sqrt', 'log2'],
    'rf__min_samples_leaf': [2, 5, 7, 10, 15],
    'rf__min_samples_split': [2, 5, 7, 10, 12, 15, 18, 20],
    'rf__n_estimators': [100, 200, 500, 1000, 1500, 2000]
}

gkf = GroupKFold(n_splits=5)

rf_grid_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=rf_param_grid,
    n_iter=300,
    cv=gkf,
    n_jobs=-1,
    random_state=42,
    verbose=3,
    scoring='average_precision'
)

rf_grid_search.fit(X_model, Y_df, groups=groups)
rf_best_grid = rf_grid_search.best_estimator_
print(rf_grid_search.best_params_)

joblib.dump(rf_best_grid, "rf_model.pkl")

# -----------------------------
# OUT-OF-FOLD PREDICTIONS
# -----------------------------
y_pred_proba = cross_val_predict(
    rf_best_grid,
    X_model,
    Y_df,
    groups=groups,
    cv=gkf,
    method='predict_proba',
    n_jobs=-1
)

oof_df = pd.DataFrame({
    'MRN': groups,
    'true_label': Y_df,
    'pred_proba': y_pred_proba[:, 1]
})
oof_df.to_parquet("oof_rf.parquet", index=False)

# -----------------------------
# GINI IMPORTANCE
# -----------------------------
feature_names = X_model.columns.tolist()
importances = rf_best_grid.named_steps['rf'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

fig, ax = plt.subplots(figsize=(10, 6))
importance_df.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False, ax=ax)
ax.set_title('Top 20 Gini Importances (Random Forest)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('rf_gini_importance.png', dpi=300)
plt.show()

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
result = permutation_importance(
    rf_best_grid,
    X_model,
    Y_df,
    n_repeats=30,
    random_state=42,
    scoring='average_precision'
)

perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

print(perm_importance_df)

fig, ax = plt.subplots(figsize=(10, 6))
perm_importance_df.head(20).plot(kind='barh', x='Feature', y='Importance Mean', legend=False, ax=ax)
ax.set_title('Top 20 Permutation Importances (Random Forest)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('rf_perm_importance.png', dpi=300)
plt.show()

# -----------------------------
# SHAP BEESWARM
# -----------------------------
rf_model = rf_best_grid.named_steps['rf']

masker = shap.maskers.Independent(X_model, max_samples=100)
explainer = shap.Explainer(rf_model, masker)


def save_beeswarm_rf(X, fname, max_display=30, sample_size=1000):
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    shap_values = explainer(X_sample)
    shap_pos = shap_values[:, :, 1]
    explanation = shap.Explanation(
        shap_pos.values,
        base_values=shap_pos.base_values,
        data=X_sample.values,
        feature_names=feature_names
    )
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")


save_beeswarm_rf(X_model, "rf_shap_beeswarm_oof.png")
save_beeswarm_rf(X_test, "rf_shap_beeswarm_test.png")
