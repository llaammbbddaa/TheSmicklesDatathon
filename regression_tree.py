"""
Regression Tree (CART) to predict Mortality Rate (%)
=====================================================
Dataset : globalHealthStats.csv  (~1 M rows, 22 columns)
Target  : Mortality Rate (%)
Method  : scikit-learn DecisionTreeRegressor with GridSearchCV

Steps
-----
1. Load & inspect data
2. Preprocessing  – handle missing values, encode categoricals, scale numerics
3. Train / validation split  +  cross-validation
4. Hyperparameter tuning  (max_depth, min_samples_leaf, criterion)
5. Performance metrics  (MAE, RMSE, R²)
6. Feature importance  +  partial-dependence plots (top features)
7. Save the final model to disk  (joblib)
8. Visualize the full decision tree as an image
"""

# ── Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import joblib, os, warnings
warnings.filterwarnings("ignore")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("=" * 60)
print("STEP 1 – Load data")
print("=" * 60)

df = pd.read_csv("globalHealthStats.csv")
print(f"Shape: {df.shape}")
print(f"Columns:\n{list(df.columns)}\n")
print(df.head(3))
print(df.info())

# =====================================================================
# 2. PREPROCESSING
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 – Preprocessing")
print("=" * 60)

# -- 2a. Target column -------------------------------------------------
TARGET = "Mortality Rate (%)"
assert TARGET in df.columns, f"Target column '{TARGET}' not found!"

# -- 2b. Feature columns (all except target) ---------------------------
feature_cols = [
    "Country", "Year", "Disease Name", "Disease Category",
    "Prevalence Rate (%)", "Incidence Rate (%)", "Age Group", "Gender",
    "Population Affected", "Healthcare Access (%)", "Doctors per 1000",
    "Hospital Beds per 1000", "Treatment Type",
    "Average Treatment Cost (USD)", "Availability of Vaccines/Treatment",
    "Recovery Rate (%)", "DALYs", "Improvement in 5 Years (%)",
    "Per Capita Income (USD)", "Education Index", "Urbanization Rate (%)",
]
# Keep only columns that actually exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"Using {len(feature_cols)} feature columns.")

X = df[feature_cols].copy()
y = df[TARGET].copy()

# -- 2c. Missing values ------------------------------------------------
print(f"\nMissing values BEFORE handling:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
print(f"Target missing: {y.isnull().sum()}")

# Drop rows where target is missing
mask = y.notna()
X, y = X[mask], y[mask]

# Numeric cols → fill with median; categorical cols → fill with mode
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
    if X[c].isnull().any():
        X[c].fillna(X[c].median(), inplace=True)
for c in cat_cols:
    if X[c].isnull().any():
        X[c].fillna(X[c].mode()[0], inplace=True)

print(f"Missing values AFTER handling: {X.isnull().sum().sum()}")

# -- 2d. Encode categorical features (Label Encoding) ------------------
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    label_encoders[c] = le
    print(f"  Encoded '{c}' → {len(le.classes_)} classes")

# -- 2e. Scaling (tree models are invariant to monotone transforms,
#         but we scale anyway for completeness; store scaler for reuse) -
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
print(f"\nScaled feature matrix shape: {X_scaled.shape}")

# NOTE: Decision trees do NOT require scaling. We keep the *unscaled*
# version for training so that splits remain interpretable, but the
# scaler is saved alongside the model for pipelines that might need it.
X_train_raw = X  # unscaled – used for training

# =====================================================================
# 3. TRAIN / VALIDATION SPLIT + CROSS-VALIDATION
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 – Train / Validation split + Cross-validation")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw, y, test_size=0.2, random_state=42
)
print(f"Train size : {X_train.shape[0]:,}")
print(f"Val   size : {X_val.shape[0]:,}")

# Quick baseline – default tree with 3-fold CV on a subsample
baseline = DecisionTreeRegressor(random_state=42)
# Use a 100 k subsample for fast CV (dataset is 800 k)
SUB_N = min(100_000, len(X_train))
rng = np.random.RandomState(42)
sub_idx = rng.choice(X_train.index, size=SUB_N, replace=False)
X_sub, y_sub = X_train.loc[sub_idx], y_train.loc[sub_idx]

cv_scores = cross_val_score(baseline, X_sub, y_sub, cv=3,
                            scoring="neg_mean_absolute_error")
print(f"\nBaseline 3-fold CV MAE (100k sample): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =====================================================================
# 4. HYPERPARAMETER TUNING  (GridSearchCV on subsample)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 – Hyperparameter tuning (GridSearchCV on 100k subsample)")
print("=" * 60)

param_grid = {
    "criterion":       ["squared_error", "friedman_mse"],
    "max_depth":       [5, 10, 15, 20, None],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
grid.fit(X_sub, y_sub)

print(f"\nBest params : {grid.best_params_}")
print(f"Best CV MAE : {-grid.best_score_:.4f}")

# Refit best model on the FULL training set
print("\nRefitting best model on full training set …")
best_tree = DecisionTreeRegressor(random_state=42, **grid.best_params_)
best_tree.fit(X_train, y_train)
print("Done.")

# =====================================================================
# 5. PERFORMANCE METRICS  (MAE, RMSE, R²)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5 – Performance metrics")
print("=" * 60)

y_pred_train = best_tree.predict(X_train)
y_pred_val   = best_tree.predict(X_val)

def report(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return mae, rmse, r2

train_metrics = report(y_train, y_pred_train, "Train")
val_metrics   = report(y_val,   y_pred_val,   "Val  ")

# =====================================================================
# 6. FEATURE IMPORTANCE + PARTIAL DEPENDENCE PLOTS
# =====================================================================
print("\n" + "=" * 60)
print("STEP 6 – Feature importance & partial dependence plots")
print("=" * 60)

importances = best_tree.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 features:")
print(feat_imp.head(10).to_string())

# -- 6a. Bar plot of feature importances --------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.head(15).plot.barh(ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("Top 15 Feature Importances (Decision Tree)")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close(fig)
print(f"  → Saved {OUT_DIR}/feature_importance.png")

# -- 6b. Partial Dependence Plots for top 4 features --------------------
top_features = feat_imp.head(4).index.tolist()
top_indices  = [list(X_train.columns).index(f) for f in top_features]

fig, ax = plt.subplots(figsize=(14, 8))
display = PartialDependenceDisplay.from_estimator(
    best_tree, X_val, features=top_indices,
    feature_names=list(X_train.columns),
    ax=ax, kind="average", grid_resolution=50,
)
fig.suptitle("Partial Dependence – Top 4 Features", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/partial_dependence.png", dpi=150)
plt.close(fig)
print(f"  → Saved {OUT_DIR}/partial_dependence.png")

# =====================================================================
# 7. SAVE THE FINAL MODEL TO DISK
# =====================================================================
print("\n" + "=" * 60)
print("STEP 7 – Save model artefacts")
print("=" * 60)

joblib.dump(best_tree,        f"{OUT_DIR}/best_regression_tree.joblib")
joblib.dump(scaler,           f"{OUT_DIR}/scaler.joblib")
joblib.dump(label_encoders,   f"{OUT_DIR}/label_encoders.joblib")
print(f"  → Model saved to {OUT_DIR}/best_regression_tree.joblib")
print(f"  → Scaler saved to {OUT_DIR}/scaler.joblib")
print(f"  → Encoders saved to {OUT_DIR}/label_encoders.joblib")

# =====================================================================
# 8. VISUALIZE THE FULL DECISION TREE
# =====================================================================
print("\n" + "=" * 60)
print("STEP 8 – Full decision tree visualization")
print("=" * 60)

# Determine tree depth for sizing
depth = best_tree.get_depth()
n_leaves = best_tree.get_n_leaves()
print(f"  Tree depth  : {depth}")
print(f"  Num leaves  : {n_leaves}")

# For very large trees, cap figure size to avoid memory issues
fig_w = min(max(n_leaves * 2, 40), 300)
fig_h = min(max(depth * 3, 15), 80)

fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=80)
plot_tree(
    best_tree,
    feature_names=list(X_train.columns),
    filled=True,
    rounded=True,
    fontsize=6,
    ax=ax,
    impurity=False,
    proportion=False,
)
ax.set_title("Full Regression Tree (CART)", fontsize=16)
plt.tight_layout()
tree_path = f"{OUT_DIR}/full_decision_tree.png"
fig.savefig(tree_path, dpi=80, bbox_inches="tight")
plt.close(fig)
print(f"  → Saved {tree_path}")

# Also export a compact text representation
text_rep = export_text(best_tree, feature_names=list(X_train.columns), max_depth=5)
with open(f"{OUT_DIR}/tree_text_rules.txt", "w") as f:
    f.write(text_rep)
print(f"  → Saved {OUT_DIR}/tree_text_rules.txt (top-5 depth text rules)")

print("\n" + "=" * 60)
print("ALL DONE ✓")
print("=" * 60)
