# Side-by-side comparison of LR / RF / XGBoost metrics. Reads from outputs/ after running each.
import os
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base, "outputs")

def parse_metrics(path):
    if not os.path.isfile(path):
        return None
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if line.startswith(name + ":"):
                    try:
                        metrics[name] = float(line.split(":")[1].strip())
                    except ValueError:
                        pass
                    break
    return metrics if len(metrics) == 5 else None

files = [
    ("classifier_metrics.txt", "Random Forest (best)"),
    ("classifier_rf_metrics.txt", "Random Forest"),
    ("classifier_xgb_metrics.txt", "XGBoost"),
]
results = [(label, parse_metrics(os.path.join(out_dir, f))) for f, label in files]
if not any(r[1] for r in results):
    print("No metrics found in outputs/. Run src/train_classifier.py and/or alternatives/train_classifier_*.py first.")
    exit(1)
print("Metric         " + "".join(f" {r[0]:<24}" for r in results if r[1]))
print("-" * 80)
for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    row = f"{key:<14}"
    for _, m in results:
        row += f" {m.get(key, float('nan')):<24.4f}" if m else " —"
    print(row)
