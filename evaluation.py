from environment_setup import *
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---- Load data and encoders ----
encoder = joblib.load("encoder.joblib")
label_encoder = joblib.load("label_encoder.joblib")
X_train, X_test, y_train, y_test = joblib.load("train_test_data.joblib")

print("Data and encoders loaded")

# ---- Subsample for hyperparameter tuning ----
X_subsample, _, y_subsample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

# ---- Parameter grid ----
param_grid = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0]
}

# ---- XGBoost classifier for grid search ----
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False
)

print("Tuning XGBoost parameters...")

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

# ---- Fit grid search ----
grid_search.fit(X_subsample, y_subsample)
print("Best Parameters:", grid_search.best_params_)

# ---- Train final model ----
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    **grid_search.best_params_
)

model.fit(X_train, y_train)
print("Model training complete!")

# ---- Save final model ----
joblib.dump(model, "xgb_final_model.joblib")
print("Final model saved as xgb_final_model.joblib")

# ==========================
# ---- Evaluation & Results ----
# ==========================

# ---- Make predictions ----
y_pred = model.predict(X_test)

# ---- Convert predictions back to original labels ----
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# ---- Evaluate performance ----
print(" Model Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels))

# ---- Feature Importance ----
feature_names = encoder.get_feature_names_out()
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(np.array(feature_names)[top_idx], importances[top_idx], color=ACCENT_COLORS[3])
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_title("Top Feature Importances (XGBoost)", fontsize=14, weight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("visualizations/feature_importance.png", facecolor=BACKGROUND_COLOR)
plt.show()

# ---- Confusion Matrix ----
conf_mat = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat,
            annot=True,
            fmt="d",
            cmap="BuPu",
            linewidths=0.5,
            linecolor="gray",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={"label": "Count"},
            ax=ax)

fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_title("Confusion Matrix", fontsize=14, weight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("visualizations/confusion_matrix.png", facecolor=BACKGROUND_COLOR)
plt.show()
