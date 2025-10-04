# evaluation/metrics_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "evaluation/results/"
os.makedirs("evaluation/plots", exist_ok=True)

def plot_metrics(file, title):
    df = pd.read_csv(file)
    metrics = ["precision", "recall", "false_alarm_rate"]

    for metric in metrics:
        plt.figure(figsize=(6,5))
        plt.bar(df["model"], df[metric], color="skyblue")
        plt.title(f"{title} - {metric.upper()}")
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.ylim(0,2)
        plt.tight_layout()
        out_path = f"evaluation/plots/{title}_{metric}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Saved {out_path}")

if __name__ == "__main__":
    plot_metrics("evaluation/results/pir_metrics.csv", "PIR")
    # Add for contact_metrics.csv, env_metrics.csv if available
