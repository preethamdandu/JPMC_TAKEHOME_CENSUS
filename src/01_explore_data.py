# EDA: target balance, missing values, numeric distributions.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from data_loader import load_census_data, clean_label, get_feature_columns, get_numeric_columns

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_census_data(
        os.path.join(base, "data", "census-bureau.data"),
        os.path.join(base, "data", "census-bureau.columns"),
    )
    df["label_binary"] = clean_label(df["label"])
    feature_cols = get_feature_columns(df)
    num_cols = get_numeric_columns(df, feature_cols)

    # how bad is the imbalance?
    print("=== Target ===")
    print(df["label_binary"].value_counts())
    print("Proportion >50K:", df["label_binary"].mean().round(4))

    # quick look at numeric distributions — any weird ranges or all-zero columns?
    print("\n=== Numeric (sample) ===")
    if num_cols:
        sub = df[num_cols].apply(pd.to_numeric, errors="coerce")
        print(sub.describe().T[["count", "mean", "std"]].head(10))

    # which columns have the most missing? helps decide imputation
    print("\n=== Missing (top 10) ===")
    print(df[feature_cols].isna().sum().sort_values(ascending=False).head(10))

    if "education" in df.columns:
        print("\n=== Education value counts ===")
        print(df["education"].value_counts().head(10))

if __name__ == "__main__":
    main()
