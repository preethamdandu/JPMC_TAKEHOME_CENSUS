# Standalone RF script (comparison version; final model is in src/train_classifier.py).
import os
import sys
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base, "src"))
import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from data_loader import load_census_data, clean_label, get_feature_columns, get_numeric_columns, get_categorical_columns

def build_preprocessor(df, feature_cols):
    num_cols = [c for c in get_numeric_columns(df, feature_cols) if c in df.columns]
    cat_cols = [c for c in get_categorical_columns(df, feature_cols) if c in df.columns]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="census-bureau.data")
    p.add_argument("--columns", default="census-bureau.columns")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--no-grid", action="store_true")
    args = p.parse_args()
    data_path = os.path.join(base, "data", args.data)
    columns_path = os.path.join(base, "data", args.columns)
    out_dir = os.path.join(base, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    df = load_census_data(data_path, columns_path)
    df["label_binary"] = clean_label(df["label"])
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["label_binary"]
    for c in get_numeric_columns(df, feature_cols):
        if c in X.columns and X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    preprocessor = build_preprocessor(X_train, feature_cols)
    pipe = Pipeline([("preprocess", preprocessor), ("classifier", RandomForestClassifier(random_state=args.random_state, class_weight="balanced"))])
    if args.no_grid:
        pipe.fit(X_train, y_train)
        best_pipe = pipe
    else:
        grid = GridSearchCV(pipe, {"classifier__n_estimators": [100, 200], "classifier__max_depth": [15, 25, None], "classifier__min_samples_leaf": [2, 5]}, cv=StratifiedKFold(5, shuffle=True, random_state=args.random_state), scoring="roc_auc", n_jobs=-1, refit=True, verbose=1)
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        print("Best params:", grid.best_params_)
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0), "f1": f1_score(y_test, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0}
    for k, v in metrics.items(): print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
    with open(os.path.join(out_dir, "classifier_rf_model.pkl"), "wb") as f:
        pickle.dump(best_pipe, f)
    with open(os.path.join(out_dir, "classifier_rf_metrics.txt"), "w") as f:
        f.write("Test set metrics (Random Forest):\n")
        for k, v in metrics.items(): f.write(f"  {k}: {v:.4f}\n")
    print("Saved: outputs/classifier_rf_model.pkl, classifier_rf_metrics.txt")

if __name__ == "__main__":
    main()
