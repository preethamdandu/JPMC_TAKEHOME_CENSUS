# Loads census data (no header row — column names come from a separate file).
import os
import pandas as pd
import numpy as np

# These show up as missing in the raw data instead of blank/NaN
NA_VALUES = (
    "?",
    "Not in universe",
    "Not in universe or children",
    "Not in universe under 1 year old",
)


def load_columns(columns_path: str) -> list:
    with open(columns_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_census_data(data_path: str, columns_path: str, na_values: tuple = NA_VALUES) -> pd.DataFrame:
    columns = load_columns(columns_path)
    return pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=na_values,
        skipinitialspace=True,
        low_memory=False,
    )


def clean_label(series: pd.Series) -> pd.Series:
    # Label column has values like " 50000+." and " - 50000." — match on "50000+"
    raw = series.astype(str).str.strip()
    return (raw.str.contains("50000\\+", regex=True)).astype(int)


def get_feature_columns(df: pd.DataFrame) -> list:
    # weight is for population sampling, not prediction; label is the target
    exclude = {"weight", "label", "label_binary"}
    return [c for c in df.columns if c not in exclude]


def get_numeric_columns(df: pd.DataFrame, feature_cols: list) -> list:
    return [
        c for c in feature_cols
        if c in df.columns and (df[c].dtype in ("int64", "float64") or np.issubdtype(df[c].dtype, np.number))
    ]


def get_categorical_columns(df: pd.DataFrame, feature_cols: list) -> list:
    num = set(get_numeric_columns(df, feature_cols))
    return [c for c in feature_cols if c in df.columns and c not in num]


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", "census-bureau.data")
    columns_path = os.path.join(base, "data", "census-bureau.columns")
    df = load_census_data(data_path, columns_path)
    print("Loaded shape:", df.shape)
    df["label_binary"] = clean_label(df["label"])
    print("Proportion >50K:", df["label_binary"].mean().round(4))
    print("Feature columns:", len(get_feature_columns(df)))
