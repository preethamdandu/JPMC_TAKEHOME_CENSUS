# Generates all plots for the report (saves to outputs/plots/).
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_loader import (
    load_census_data, clean_label, get_feature_columns, get_numeric_columns,
)

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, "outputs", "plots")
os.makedirs(out_dir, exist_ok=True)

# Load the saved RF pipeline and redo the same 80/20 split (same random_state)
with open(os.path.join(base, "outputs", "classifier_model.pkl"), "rb") as f:
    pipe = pickle.load(f)

data_path = os.path.join(base, "data", "census-bureau.data")
columns_path = os.path.join(base, "data", "census-bureau.columns")
df_full = load_census_data(data_path, columns_path)
df_full["label_binary"] = clean_label(df_full["label"])
feature_cols = get_feature_columns(df_full)
X_all = df_full[feature_cols].copy()
y_all = df_full["label_binary"]
for c in get_numeric_columns(df_full, feature_cols):
    if c in X_all.columns and X_all[c].dtype == object:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

# ---- Figure 1: Top 10 feature importances ----
rf = pipe.named_steps["classifier"]
names = pipe.named_steps["preprocess"].get_feature_names_out()
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head(10)
imp = imp.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy", alpha=0.8)
ax.set_xlabel("Importance")
ax.set_title("Top 10 Feature Importances (Random Forest)")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig1_feature_importance.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig1_feature_importance.png")

# ---- Figure 2: >50K rate by segment ----
df_seg = pd.read_csv(os.path.join(base, "outputs", "segment_assignments.csv"))
grp = df_seg.groupby("segment")["label_binary"].agg(["mean", "count"]).sort_index()
grp["pct"] = 100 * grp["mean"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(grp.index.astype(str), grp["pct"], color="steelblue", edgecolor="navy", alpha=0.8)
ax.set_xlabel("Segment")
ax.set_ylabel("% >50K")
ax.set_title("Income >50K Rate by Segment (K-Prototypes k=6)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig2_segment_rates.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig2_segment_rates.png")

# ---- Figure 3: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix (Random Forest, Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig3_confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig3_confusion_matrix.png")

# Save confusion matrix to metrics file (only if not already there)
metrics_path = os.path.join(base, "outputs", "classifier_metrics.txt")
with open(metrics_path, "r") as f:
    existing = f.read()
if "Confusion Matrix" not in existing:
    with open(metrics_path, "a") as f:
        f.write("\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"              <=50K   >50K\n")
        f.write(f"  <=50K      {cm[0,0]:>6d}  {cm[0,1]:>6d}\n")
        f.write(f"  >50K       {cm[1,0]:>6d}  {cm[1,1]:>6d}\n")
    print("Appended confusion matrix to classifier_metrics.txt")
else:
    print("Confusion matrix already in classifier_metrics.txt, skipping")

# ---- Figure 4: ROC Curve ----
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"RF (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Random Forest, Test Set)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig4_roc_curve.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig4_roc_curve.png")

# ---- Figure 5: Precision-Recall Curve ----
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, color="steelblue", lw=2, label=f"RF (AP = {ap:.4f})")
baseline = y_test.mean()
ax.axhline(y=baseline, color="gray", ls="--", lw=1, alpha=0.6, label=f"Baseline ({baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (Random Forest, Test Set)")
ax.legend(loc="upper right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig5_pr_curve.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig5_pr_curve.png")

# Print a few threshold checkpoints for the report
print("\nPrecision / Recall at selected thresholds:")
print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    idx = np.searchsorted(thresholds, t)
    if idx < len(precision) and idx < len(recall):
        print(f"  {t:>10.2f}  {precision[idx]:>10.4f}  {recall[idx]:>10.4f}")

# ---- Figure 6: Class distribution (EDA) ----
counts = y_all.value_counts().sort_index()
labels = ["\u226450K", ">50K"]
fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(labels, counts.values, color=["steelblue", "coral"], edgecolor="navy", alpha=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"{val:,}\n({100 * val / counts.sum():.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Target Class Distribution")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig6_class_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved fig6_class_distribution.png")
