# K-Prototypes segmentation (k=6, mixed numeric + categorical).
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import (
    load_census_data, clean_label, get_feature_columns,
    get_numeric_columns, get_categorical_columns,
)
try:
    from kmodes.kprototypes import KPrototypes
except ImportError:
    raise ImportError("Install kmodes: pip install kmodes")

# drop categoricals with too many unique values (e.g. state) — they dominate the distance
MAX_CATEGORIES = 50


def prepare_mixed_features(df, feature_cols):
    num_cols = [c for c in get_numeric_columns(df, feature_cols) if c in df.columns]
    cat_cols = [c for c in get_categorical_columns(df, feature_cols) if c in df.columns]
    X_num = df[num_cols].copy()
    for c in num_cols:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
    X_num = X_num.fillna(X_num.median())
    X_cat = pd.DataFrame(index=df.index)
    for c in cat_cols:
        n_unique = df[c].nunique()
        if n_unique <= MAX_CATEGORIES and n_unique > 1:
            X_cat[c] = df[c].fillna("__missing__").astype(str)
    valid = ~(X_num.isna().any(axis=1) | np.isinf(X_num).any(axis=1))
    if X_cat.shape[1] > 0:
        valid = valid & (~X_cat.isna().any(axis=1))
    X_num = X_num.loc[valid]
    X_cat = X_cat.loc[valid] if X_cat.shape[1] > 0 else pd.DataFrame(index=X_num.index)
    return X_num, X_cat, num_cols, list(X_cat.columns), valid


def main():
    parser = argparse.ArgumentParser(description="Obj 2: Best segmentation (K-Prototypes k=6)")
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--init", default="Cao", choices=["Cao", "Huang", "random"])
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", args.data)
    columns_path = os.path.join(base, "data", args.columns)
    out_dir = os.path.join(base, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_census_data(data_path, columns_path)
    df["label_binary"] = clean_label(df["label"])
    feature_cols = get_feature_columns(df)
    X_num, X_cat, num_cols, cat_cols, valid = prepare_mixed_features(df, feature_cols)
    df_sub = df.loc[valid].copy()

    if args.sample is not None:
        n = min(args.sample, len(df_sub))
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(df_sub.index, size=n, replace=False)
        X_num = X_num.loc[idx]
        X_cat = X_cat.loc[idx] if X_cat.shape[1] > 0 else X_cat
        df_sub = df_sub.loc[idx]
        print(f"Using sample of {n:,} rows.")

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    X_num_scaled = pd.DataFrame(X_num_scaled, index=X_num.index, columns=num_cols)
    if X_cat.shape[1] == 0:
        raise ValueError("K-Prototypes needs at least one categorical column.")
    # K-Prototypes wants numerics first, then categoricals — tell it which columns are categorical
    X = pd.concat([X_num_scaled, X_cat], axis=1)
    categorical_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    print(f"Fitting K-Prototypes k={args.n_clusters}...")
    # n_init=3: run 3 times with different seeds, pick the best (avoids bad initialization)
    kp = KPrototypes(n_clusters=args.n_clusters, init=args.init, random_state=args.random_state, verbose=0, n_init=3)
    kp.fit(X, categorical=categorical_indices)
    df_sub["segment"] = kp.labels_

    print("\nSegment sizes:")
    for seg, count in df_sub["segment"].value_counts().sort_index().items():
        print(f"  Segment {seg}: {count:,} ({100*count/len(df_sub):.1f}%)")
    # post-hoc: how does income break down by segment? (labels not used during clustering)
    print("\n>50K rate by segment:")
    for seg in sorted(df_sub["segment"].unique()):
        print(f"  Segment {seg}: {df_sub.loc[df_sub['segment']==seg,'label_binary'].mean():.2%}")
    key_cols = [c for c in ["age", "education", "marital stat", "sex", "full or part time employment stat"] if c in df_sub.columns]
    print("\nSegment profiles (key variables):")
    for seg in sorted(df_sub["segment"].unique()):
        sub = df_sub[df_sub["segment"] == seg]
        print(f"  --- Segment {seg} (n={len(sub):,}) ---")
        for col in key_cols:
            if col in sub.columns and sub[col].dtype in (np.int64, np.float64, "int64", "float64"):
                print(f"    {col}: mean = {sub[col].mean():.2f}")
            else:
                print(f"    {col}: top = {dict(sub[col].value_counts().head(3))}")

    model_file = os.path.join(out_dir, "segmentation_model.pkl")
    assignments_file = os.path.join(out_dir, "segment_assignments.csv")
    with open(model_file, "wb") as f:
        pickle.dump({
            "kprototypes": kp, "scaler": scaler,
            "numeric_columns": num_cols, "categorical_columns": cat_cols,
            "categorical_indices": categorical_indices, "n_clusters": args.n_clusters,
        }, f)
    df_sub[["segment", "label_binary"]].to_csv(assignments_file, index=False)
    print(f"\nSaved: {model_file}, {assignments_file}")


if __name__ == "__main__":
    main()
