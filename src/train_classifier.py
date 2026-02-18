# Train RF classifier (<=50K vs >50K) with grid search.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from data_loader import (
    load_census_data, clean_label, get_feature_columns,
    get_numeric_columns, get_categorical_columns,
)


def build_preprocessor(df, feature_cols):
    num_cols = [c for c in get_numeric_columns(df, feature_cols) if c in df.columns]
    cat_cols = [c for c in get_categorical_columns(df, feature_cols) if c in df.columns]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),  # median — capital gains is very skewed
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # ignore unseen cats at predict time
        ]), cat_cols))
    if not transformers:
        raise ValueError("No valid feature columns.")
    return ColumnTransformer(transformers, remainder="drop")


def main():
    parser = argparse.ArgumentParser(description="Obj 1: Train best classifier (Random Forest)")
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-grid", action="store_true", help="Skip grid search (faster)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", args.data)
    columns_path = os.path.join(base, "data", args.columns)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_census_data(data_path, columns_path)
    df["label_binary"] = clean_label(df["label"])
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["label_binary"]
    # some numeric columns load as object dtype if they have mixed content
    for c in get_numeric_columns(df, feature_cols):
        if c in X.columns and X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    print("Stratified 80/20 split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train, feature_cols)
    # balanced weights so the model doesn't just predict <=50K for everything (94% of data)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(random_state=args.random_state, class_weight="balanced")),
    ])

    if args.no_grid:
        print("Training (no grid)...")
        pipe.fit(X_train, y_train)
        best_pipe = pipe
    else:
        print("Grid search (CV ROC-AUC)...")
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [15, 25, None],
            "classifier__min_samples_leaf": [2, 5],
        }
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True, verbose=1)
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        print("Best params:", grid.best_params_)
        print("Best CV ROC-AUC:", round(grid.best_score_, 4))

    print("\nHoldout evaluation...")
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0,
    }
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # print top features — useful for sanity checking the model
    rf = best_pipe.named_steps["classifier"]
    if hasattr(rf, "feature_importances_"):
        try:
            names = best_pipe.named_steps["preprocess"].get_feature_names_out()
            if len(names) == len(rf.feature_importances_):
                imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
                print("\nTop 10 feature importances:")
                print(imp.head(10).to_string())
        except Exception:
            pass

    with open(os.path.join(out_dir, "classifier_model.pkl"), "wb") as f:
        pickle.dump(best_pipe, f)
    with open(os.path.join(out_dir, "classifier_metrics.txt"), "w") as f:
        f.write("Test set metrics (Random Forest):\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n" + classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
    print(f"Saved: {out_dir}/classifier_model.pkl, classifier_metrics.txt")


if __name__ == "__main__":
    main()
