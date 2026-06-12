# XGBoost

import pandas as pd
import numpy as np
import random as rn
import joblib
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier
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
    ('xgb', XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ))
])

xgb_param_grid = {
    'xgb__max_depth': [3, 5, 7, 9, 11],
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__n_estimators': [100, 300, 500, 800, 1000, 1200],
    'xgb__subsample': [0.4, 0.6, 0.8],
    'xgb__colsample_bytree': [0.4, 0.5, 0.6],
    'xgb__min_child_weight': [2, 3, 5, 8],
    'xgb__gamma': [0, 1, 2, 3, 4, 5],
    'xgb__reg_alpha': [0, 0.5, 1, 2, 3],
    'xgb__reg_lambda': [0, 0.5, 1, 2, 3],
    'xgb__scale_pos_weight': [0.5, 1, 2, 5]
}

gkf = GroupKFold(n_splits=5)

xgb_grid_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=xgb_param_grid,
    n_iter=300,
    cv=gkf,
    n_jobs=-1,
    random_state=42,
    verbose=3,
    scoring='average_precision'
)

xgb_grid_search.fit(X_model, Y_df, groups=groups)
xgb_best_grid = xgb_grid_search.best_estimator_
print(xgb_grid_search.best_params_)

joblib.dump(xgb_best_grid, "xgb_model.pkl")

# -----------------------------
# OUT-OF-FOLD PREDICTIONS
# -----------------------------
y_pred_proba = cross_val_predict(
    xgb_best_grid,
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
oof_df.to_parquet("oof_xgb.parquet", index=False)

# -----------------------------
# GINI IMPORTANCE
# -----------------------------
feature_names = X_model.columns.tolist()
importances = xgb_best_grid.named_steps['xgb'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

fig, ax = plt.subplots(figsize=(10, 6))
importance_df.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False, ax=ax)
ax.set_title('Top 20 Feature Importances (XGBoost)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_gini_importance.png', dpi=300)
plt.show()

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
result = permutation_importance(
    xgb_best_grid,
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
ax.set_title('Top 20 Permutation Importances (XGBoost)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_perm_importance.png', dpi=300)
plt.show()

# -----------------------------
# SHAP BEESWARM
# -----------------------------
xgb_model = xgb_best_grid.named_steps['xgb']

background = shap.sample(X_model, min(200, len(X_model)), random_state=42)
explainer = shap.TreeExplainer(
    xgb_model,
    data=background,
    feature_names=feature_names,
    model_output="probability"
)


def save_beeswarm_xgb(X, fname, max_display=30):
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        max_display=max_display
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")


save_beeswarm_xgb(X_model, "shap_beeswarm_xgb_oof.png")
save_beeswarm_xgb(X_test, "shap_beeswarm_xgb_test.png")
