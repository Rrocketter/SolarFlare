"""
solar_flare_analysis.py

A comprehensive script for:
1. Comparing model performance with NOAA/SWPC benchmarks
2. Generating visualizations for presentations
3. Calculating socioeconomic impact

Requirements:
- Python 3.10+
- Required packages: pandas, numpy, matplotlib, scikit-learn, seaborn
- Input files:
  - model_predictions.csv (your model's outputs)
  - noaa_historical.csv (download from SWPC archive)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from matplotlib.backends.backend_pdf import PdfPages

# Configuration
CONFIG = {
    "data_paths": {
        "model": "model_predictions.csv",
        "noaa": "noaa_historical.csv"
    },
    "cost_factors": {
        "false_alarm": 250000,  # USD per false alarm
        "missed_event": 7800000,  # USD per missed event
        "satellite_hr": 1500  # USD per satellite per hour
    },
    "visualization": {
        "style": "seaborn",
        "colors": {"noaa": "#1f77b4", "model": "#ff7f0e"},
        "dpi": 300
    }
}


def load_data():
    """Load and merge datasets"""
    model = pd.read_csv(CONFIG["data_paths"]["model"], parse_dates=["timestamp"])
    noaa = pd.read_csv(CONFIG["data_paths"]["noaa"], parse_dates=["timestamp"])

    # Merge on nearest timestamp (within 5 minutes)
    merged = pd.merge_asof(
        model.sort_values("timestamp"),
        noaa.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("5min"),
        suffixes=("_model", "_noaa")
    )

    return merged


def calculate_metrics(df):
    """Calculate performance metrics"""
    metrics = {}

    # Threat Score (TS) = TSS
    tn, fp, fn, tp = confusion_matrix(
        df["flare_observed"], df["flare_pred_model"]
    ).ravel()
    metrics["tss"] = tp / (tp + fn + fp)

    # False Alarm Ratio
    metrics["far"] = fp / (fp + tp)

    # Lead Time improvement
    metrics["lead_time_improvement"] = (
            df["lead_time_model"] - df["lead_time_noaa"]
    ).mean()

    # Economic impact
    savings = (
            (CONFIG["cost_factors"]["false_alarm"] * (metrics["far"] * len(df))) +
            (CONFIG["cost_factors"]["missed_event"] * (1 - metrics["tss"]) * len(df)) +
            (CONFIG["cost_factors"]["satellite_hr"] * metrics["lead_time_improvement"] * 4500)
    )
    metrics["annual_savings"] = savings

    return metrics


def plot_comparison(metrics, filename):
    """Generate performance comparison visualization"""
    plt.style.use(CONFIG["visualization"]["style"])
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # TSS Comparison
    ax[0, 0].bar(["NOAA", "Model"], [0.62, metrics["tss"]],
                 color=[CONFIG["visualization"]["colors"]["noaa"],
                        CONFIG["visualization"]["colors"]["model"]])
    ax[0, 0].set_title("True Skill Statistic (TSS) Comparison")

    # Lead Time Distribution
    ax[0, 1].hist(
        [merged["lead_time_noaa"], merged["lead_time_model"]],
        label=["NOAA", "Model"],
        color=[CONFIG["visualization"]["colors"]["noaa"],
               CONFIG["visualization"]["colors"]["model"]],
        bins=15
    )
    ax[0, 1].set_title("Lead Time Distribution")
    ax[0, 1].legend()

    # Economic Impact Breakdown
    categories = ["False Alarm Reduction", "Missed Event Prevention", "Lead Time Value"]
    values = [
        CONFIG["cost_factors"]["false_alarm"] * (0.31 - metrics["far"]) * len(merged),
        CONFIG["cost_factors"]["missed_event"] * (metrics["tss"] - 0.62) * len(merged),
        CONFIG["cost_factors"]["satellite_hr"] * metrics["lead_time_improvement"] * 4500
    ]
    ax[1, 0].pie(values, labels=categories, autopct="%1.1f%%")
    ax[1, 0].set_title("Economic Impact Breakdown")

    # FAR Comparison
    ax[1, 1].barh(["NOAA", "Model"], [0.31, metrics["far"]],
                  color=[CONFIG["visualization"]["colors"]["noaa"],
                         CONFIG["visualization"]["colors"]["model"]])
    ax[1, 1].set_title("False Alarm Ratio (FAR) Comparison")

    plt.tight_layout()
    plt.savefig(filename, dpi=CONFIG["visualization"]["dpi"])
    plt.close()


def plot_physical_interpretation(filename):
    """Generate physics interpretation visualization (example)"""
    # Example data - replace with actual Grad-CAM and magnetogram data
    magnetogram = np.random.randn(256, 256)
    gradcam = np.abs(np.random.randn(256, 256))

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Magnetogram
    im1 = ax[0].imshow(magnetogram, cmap="gray")
    plt.colorbar(im1, ax=ax[0])
    ax[0].set_title("HMI Magnetogram")

    # Grad-CAM Overlay
    ax[1].imshow(magnetogram, cmap="gray")
    im2 = ax[1].imshow(gradcam, cmap="hot", alpha=0.5)
    plt.colorbar(im2, ax=ax[1])
    ax[1].set_title("Model Attention (Grad-CAM)")

    plt.savefig(filename, dpi=CONFIG["visualization"]["dpi"])
    plt.close()


def generate_report(metrics):
    """Generate PDF report with all visualizations"""
    with PdfPages("solar_flare_analysis_report.pdf") as pdf:
        # Title Page
        plt.figure(figsize=(11, 8.5))
        plt.text(0.5, 0.5, "Solar Flare Prediction Analysis Report\n\n"
                           f"Annual Estimated Savings: ${metrics['annual_savings'] / 1e6:.2f}M",
                 ha="center", va="center", size=24)
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # Performance Comparison
        plot_comparison(metrics, "temp_comparison.png")
        plt.figure(figsize=(11, 8.5))
        plt.imshow(plt.imread("temp_comparison.png"))
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # Physical Interpretation
        plot_physical_interpretation("temp_physics.png")
        plt.figure(figsize=(11, 8.5))
        plt.imshow(plt.imread("temp_physics.png"))
        plt.axis("off")
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    # Load and prepare data
    merged = load_data()

    # Calculate metrics
    metrics = calculate_metrics(merged)
    print(f"Key Metrics:\n{'-' * 20}")
    print(f"TSS Improvement: {metrics['tss'] - 0.62:.2%}")
    print(f"FAR Reduction: {0.31 - metrics['far']:.2%}")
    print(f"Lead Time Improvement: {metrics['lead_time_improvement']:.1f} hours")
    print(f"Estimated Annual Savings: ${metrics['annual_savings'] / 1e6:.2f}M\n")

    # Generate outputs
    plot_comparison(metrics, "performance_comparison.png")
    plot_physical_interpretation("physical_interpretation.png")
    generate_report(metrics)

    print("Analysis complete. Generated files:")
    print("- performance_comparison.png")
    print("- physical_interpretation.png")
    print("- solar_flare_analysis_report.pdf")