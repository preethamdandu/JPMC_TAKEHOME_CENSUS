# Quick summary of the final segmentation output — segment sizes and >50K rates.
import os
import pandas as pd
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(base, "outputs", "segment_assignments.csv")
if not os.path.isfile(path):
    print("No outputs/segment_assignments.csv. Run src/train_segmentation.py first.")
    exit(1)
df = pd.read_csv(path)
n = len(df)
print("Segment sizes and >50K rate:")
for seg in sorted(df["segment"].unique()):
    sub = df[df["segment"] == seg]
    rate = sub["label_binary"].mean()
    print(f"  Segment {seg}: {len(sub):,} ({100*len(sub)/n:.1f}%)  >50K rate: {rate:.2%}")
spread = df.groupby("segment")["label_binary"].mean()
print(f"\n>50K spread (max-min): {spread.max() - spread.min():.4f}")
