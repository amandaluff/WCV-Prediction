# LOGISTIC REGRESSION

import pandas as pd
import numpy as np
import random as rn
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.inspection import permutation_importance
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
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

lr_param_grid = {
    'lr__C': [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 3, 10],
    'lr__penalty': ['l2', 'l1']
}

gkf = GroupKFold(n_splits=5)

lr_grid_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=lr_param_grid,
    n_iter=50,
    cv=gkf,
    n_jobs=-1,
    random_state=42,
    verbose=3,
    scoring='average_precision'
)

lr_grid_search.fit(X_model, Y_df, groups=groups)
lr_best_grid = lr_grid_search.best_estimator_
print(lr_grid_search.best_params_)

joblib.dump(lr_best_grid, "lr_model.pkl")

# -----------------------------
# OUT-OF-FOLD PREDICTIONS
# -----------------------------
y_pred_proba = cross_val_predict(
    lr_best_grid,
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
oof_df.to_parquet("oof_logistic.parquet", index=False)

# -----------------------------
# COEFFICIENT IMPORTANCE
# -----------------------------
lr = lr_best_grid.named_steps['lr']
coef = lr.coef_.ravel()
feature_names = X_model.columns.tolist()

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coef': coef,
    'AbsCoef': np.abs(coef)
}).sort_values(by='AbsCoef', ascending=False)

print(coef_df)

fig, ax = plt.subplots(figsize=(10, 6))
coef_df.head(20).plot(kind='barh', x='Feature', y='AbsCoef', legend=False, ax=ax)
ax.set_title('Top 20 Features by |Coefficient| (Logistic Regression)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('lr_coef_importance.png', dpi=300)
plt.show()

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
result = permutation_importance(
    lr_best_grid,
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
ax.set_title('Top 20 Permutation Importances (Logistic Regression)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('lr_perm_importance.png', dpi=300)
plt.show()

# -----------------------------
# SHAP BEESWARM (NONZERO COEFFICIENTS)
# -----------------------------
scaler = lr_best_grid.named_steps['scaler']

coef_tol = 0.0
nz_idx = np.where(np.abs(coef) > coef_tol)[0]
nz_features = [feature_names[i] for i in nz_idx]

X_model_scaled = pd.DataFrame(
    scaler.transform(X_model), columns=feature_names, index=X_model.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=feature_names, index=X_test.index
)

background = shap.sample(X_model_scaled, min(2000, len(X_model_scaled)), random_state=42)
explainer = shap.LinearExplainer(lr, background)


def save_beeswarm_lr(X_scaled_df, fname, max_display=30):
    values = explainer.shap_values(X_scaled_df)
    values_nz = values[:, nz_idx]
    X_nz = X_scaled_df[nz_features]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        values_nz, X_nz, show=False,
        max_display=min(max_display, len(nz_features))
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")


save_beeswarm_lr(X_model_scaled, "shap_beeswarm_lr_oof.png")
save_beeswarm_lr(X_test_scaled, "shap_beeswarm_lr_test.png")
