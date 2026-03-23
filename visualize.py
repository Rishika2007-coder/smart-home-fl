import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from model import EnergyModel, ActivityModel, load_client_data

os.makedirs("results", exist_ok=True)
sns.set_theme(style="whitegrid")


# ── Plot 1: Accuracy per FL round ──────────────────────────────────────
def plot_accuracy():
    with open("logs/metrics.json") as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(metrics["rounds"], metrics["accuracy"],
                 marker="o", color="#185FA5", linewidth=2, markersize=6)
    axes[0].set_title("Activity Detection Accuracy per FL Round",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.4)

    # Energy MSE
    axes[1].plot(metrics["rounds"], metrics["energy_mse"],
                 marker="s", color="#D85A30", linewidth=2, markersize=6)
    axes[1].set_title("Energy Prediction MSE per FL Round",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("results/fl_rounds.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/fl_rounds.png")


# ── Plot 2: Energy prediction vs actual ───────────────────────────────
def plot_energy_predictions():
    energy_model   = EnergyModel()
    activity_model = ActivityModel()

    (X_train, y_e_train, y_a_train,
     X_test,  y_e_test,  y_a_test) = load_client_data(1)

    energy_model.eval()
    with torch.no_grad():
        predictions = energy_model(X_test).numpy().flatten()
    actuals = y_e_test.numpy().flatten()

    plt.figure(figsize=(10, 4))
    x = range(len(actuals[:100]))
    plt.plot(x, actuals[:100],     label="Actual",    color="#185FA5", linewidth=1.5)
    plt.plot(x, predictions[:100], label="Predicted",
             color="#D85A30", linestyle="--", linewidth=1.5)
    plt.title("Energy Consumption: Actual vs Predicted (Home 1)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Sample")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/energy_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/energy_prediction.png")


# ── Plot 3: Confusion matrix for activity detection ───────────────────
def plot_confusion_matrix():
    from sklearn.metrics import confusion_matrix

    energy_model   = EnergyModel()
    activity_model = ActivityModel()

    (X_train, y_e_train, y_a_train,
     X_test,  y_e_test,  y_a_test) = load_client_data(1)

    activity_model.eval()
    with torch.no_grad():
        preds  = torch.argmax(activity_model(X_test), dim=1).numpy()
    actual = y_a_test.numpy()

    cm     = confusion_matrix(actual, preds)
    labels = ["Sleeping", "Active", "Away"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Activity Detection Confusion Matrix",
              fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/confusion_matrix.png")


# ── Plot 4: Data distribution across 5 homes ─────────────────────────
def plot_data_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]

    for home_id in range(1, 6):
        df = pd.read_csv(f"data/client_{home_id}.csv")
        axes[0].hist(df["energy"], bins=20, alpha=0.6,
                     label=f"Home {home_id}", color=colors[home_id - 1])
        axes[1].hist(df["temperature"], bins=20, alpha=0.6,
                     label=f"Home {home_id}", color=colors[home_id - 1])

    axes[0].set_title("Energy Distribution per Home",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Energy (kWh)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Temperature Distribution per Home",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/data_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/data_distribution.png")


if __name__ == "__main__":
    print("Generating all visualizations...\n")
    plot_energy_predictions()
    plot_confusion_matrix()
    plot_data_distribution()

    # Only plot FL metrics if training has been done
    import os
    if os.path.exists("logs/metrics.json"):
        plot_accuracy()
    else:
        print("Skipped FL rounds plot (run server.py first)")

    print("\nAll charts saved to results/")