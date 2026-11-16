import csv
import os
def log_results(metrics: dict, filepath: str = "results/metrics.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)