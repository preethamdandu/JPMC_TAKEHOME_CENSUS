# Runs K-Means, PCA+K-Means, K-Prototypes at k=4,5,6 on a 30K sample.
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_loader import (
    load_census_data, clean_label, get_feature_columns,
    get_numeric_columns, get_categorical_columns,
)

try:
    from kmodes.kprototypes import KPrototypes
except ImportError:
    KPrototypes = None
    print("Warning: kmodes not installed. K-Prototypes runs will be skipped.")

warnings.filterwarnings("ignore")

MAX_CATEGORIES = 50
RANDOM_STATE = 42
K_VALUES = [4, 5, 6]
SAMPLE_SIZE = 30000


def load_and_prep(base):
    data_path = os.path.join(base, "data", "census-bureau.data")
    columns_path = os.path.join(base, "data", "census-bureau.columns")
    df = load_census_data(data_path, columns_path)
    df["label_binary"] = clean_label(df["label"])
    feature_cols = get_feature_columns(df)

    num_cols = [c for c in get_numeric_columns(df, feature_cols) if c in df.columns]
    cat_cols = [c for c in get_categorical_columns(df, feature_cols) if c in df.columns]

    X_num = df[num_cols].copy()
    for c in num_cols:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
    X_num = X_num.fillna(X_num.median())

    X_cat = pd.DataFrame(index=df.index)
    for c in cat_cols:
        if df[c].nunique() <= MAX_CATEGORIES and df[c].nunique() > 1:
            X_cat[c] = df[c].fillna("__missing__").astype(str)

    valid = ~(X_num.isna().any(axis=1) | np.isinf(X_num).any(axis=1))
    if X_cat.shape[1] > 0:
        valid = valid & ~X_cat.isna().any(axis=1)

    X_num = X_num.loc[valid]
    X_cat = X_cat.loc[valid]
    labels = df.loc[valid, "label_binary"].values

    return X_num, X_cat, num_cols, list(X_cat.columns), labels


def compute_spread(segment_labels, income_labels):
    """max - min of >50K rate across segments."""
    df_tmp = pd.DataFrame({"seg": segment_labels, "y": income_labels})
    rates = df_tmp.groupby("seg")["y"].mean()
    return rates.max() - rates.min(), rates


def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    return km.labels_


def run_pca_kmeans(X_scaled, k, n_components=10):
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_pca)
    return km.labels_


def run_kprototypes(X_num_scaled, X_cat, cat_indices, k):
    if KPrototypes is None:
        return None
    X = pd.concat([X_num_scaled, X_cat], axis=1)
    kp = KPrototypes(n_clusters=k, init="Cao", random_state=RANDOM_STATE, verbose=0, n_init=3)
    kp.fit(X, categorical=cat_indices)
    return kp.labels_


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Loading data...")
    X_num, X_cat, num_cols, cat_cols, income = load_and_prep(base)

    # Sample for speed — K-Prototypes on 199K rows takes hours
    n = len(X_num)
    if n > SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=SAMPLE_SIZE, replace=False)
        X_num = X_num.iloc[idx].reset_index(drop=True)
        X_cat = X_cat.iloc[idx].reset_index(drop=True)
        income = income[idx]
        print(f"Sampled {SAMPLE_SIZE:,} rows from {n:,} for comparison.")

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # For K-Means: label-encode categoricals and stack with scaled numerics
    X_cat_encoded = X_cat.copy()
    for c in cat_cols:
        le = LabelEncoder()
        X_cat_encoded[c] = le.fit_transform(X_cat[c])
    X_kmeans = np.hstack([X_num_scaled, X_cat_encoded.values])

    # For K-Prototypes: scaled numerics + raw categoricals
    X_num_scaled_df = pd.DataFrame(X_num_scaled, index=X_num.index, columns=num_cols)
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    rows = []
    for k in K_VALUES:
        # K-Means
        print(f"  K-Means k={k}...", end=" ", flush=True)
        seg = run_kmeans(X_kmeans, k)
        spread, rates = compute_spread(seg, income)
        rows.append({"method": "K-Means", "k": k, "spread": round(spread, 4),
                      "max_rate": round(rates.max(), 4), "min_rate": round(rates.min(), 4)})
        print(f"spread={spread:.4f}")

        # PCA + K-Means
        print(f"  PCA+K-Means k={k}...", end=" ", flush=True)
        seg = run_pca_kmeans(X_kmeans, k)
        spread, rates = compute_spread(seg, income)
        rows.append({"method": "PCA + K-Means", "k": k, "spread": round(spread, 4),
                      "max_rate": round(rates.max(), 4), "min_rate": round(rates.min(), 4)})
        print(f"spread={spread:.4f}")

        # K-Prototypes
        if KPrototypes is not None:
            print(f"  K-Prototypes k={k}...", end=" ", flush=True)
            seg = run_kprototypes(X_num_scaled_df, X_cat, cat_indices, k)
            if seg is not None:
                spread, rates = compute_spread(seg, income)
                rows.append({"method": "K-Prototypes", "k": k, "spread": round(spread, 4),
                              "max_rate": round(rates.max(), 4), "min_rate": round(rates.min(), 4)})
                print(f"spread={spread:.4f}")

    results = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print("Segmentation Comparison: >50K Spread by Method and k")
    print("=" * 55)
    for _, r in results.iterrows():
        print(f"  {r['method']:<16s} k={r['k']}  spread={r['spread']:.4f}  "
              f"(max {r['max_rate']:.4f}, min {r['min_rate']:.4f})")

    best = results.loc[results["spread"].idxmax()]
    print(f"\nBest: {best['method']} k={best['k']} with spread {best['spread']:.4f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "segmentation_comparison_results.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
