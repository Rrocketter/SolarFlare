import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import joblib

# Set up output directory
output_dir = 'visualization_outputs'
os.makedirs(output_dir, exist_ok=True)


# This function assumes you've already run the main code and have the model outputs
def visualize_results(model_dir='models'):
    """Generate and save visualization outputs for solar flare prediction model"""
    print(f"Generating visualizations from {model_dir} - saving to {output_dir}")

    # Load model comparison data if available
    comparison_path = os.path.join(model_dir, 'model_comparison.csv')
    if os.path.exists(comparison_path):
        model_comparison = pd.read_csv(comparison_path)

        # Visualize model performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'tss', 'hss', 'roc_auc', 'pr_auc']

        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        x = np.arange(len(metrics))
        width = 0.35

        # Get model names
        models = model_comparison['model'].unique()
        colors = ['#2C7BB6', '#D7191C']

        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = model_comparison[model_comparison['model'] == model]
            values = [model_data[metric].values[0] for metric in metrics]
            ax.bar(x + (i * width), values, width, label=model, color=colors[i], alpha=0.8)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300)
        plt.close()

        # Create radar chart for model comparison
        plt.figure(figsize=(10, 10))

        # Number of variables
        N = len(metrics)

        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Plot for each model
        ax = plt.subplot(111, polar=True)

        for i, model in enumerate(models):
            model_data = model_comparison[model_comparison['model'] == model]
            values = [model_data[metric].values[0] for metric in metrics]
            values += values[:1]  # Close the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Set labels and ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)

        # Draw y-axis labels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Metrics', size=15, y=1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_radar.png'), dpi=300)
        plt.close()
    else:
        print(f"Warning: Model comparison file not found at {comparison_path}")

    # Load feature importance data if available
    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    if os.path.exists(importance_path):
        feature_importance = pd.read_csv(importance_path)

        # Create horizontal bar chart with viridis colormap
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance)), feature_importance['Importance'])

        # Color bars by importance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(feature_importance['Importance'][i] / max(feature_importance['Importance'])))

        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance Analysis')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_horizontal.png'), dpi=300)
        plt.close()

        # Create pie chart for feature importance
        plt.figure(figsize=(10, 10))
        plt.pie(feature_importance['Importance'],
                labels=feature_importance['Feature'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(feature_importance),
                colors=plt.cm.viridis(np.linspace(0, 1, len(feature_importance))))
        plt.title('Feature Contribution (Pie Chart)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_pie.png'), dpi=300)
        plt.close()
    else:
        print(f"Warning: Feature importance file not found at {importance_path}")

    # Try to load and visualize the labeled dataset
    labeled_path = os.path.join(model_dir, 'labeled_dataset.csv')
    if os.path.exists(labeled_path):
        labeled_data = pd.read_csv(labeled_path)

        # Convert timestamp to datetime if needed
        if 'timestamp' in labeled_data.columns:
            labeled_data['timestamp'] = pd.to_datetime(labeled_data['timestamp'])

        # Count flares by month if timestamp is available
        if 'timestamp' in labeled_data.columns and 'label' in labeled_data.columns:
            labeled_data['month'] = labeled_data['timestamp'].dt.strftime('%Y-%m')

            # Count flares by month
            flare_counts = labeled_data.groupby('month')['label'].sum().reset_index()
            flare_counts.columns = ['month', 'flare_count']

            # Count total observations by month
            total_counts = labeled_data.groupby('month').size().reset_index()
            total_counts.columns = ['month', 'total_count']

            # Merge the two counts
            monthly_data = pd.merge(flare_counts, total_counts, on='month')
            monthly_data['flare_percentage'] = (monthly_data['flare_count'] / monthly_data['total_count']) * 100

            # Plot monthly flare counts
            plt.figure(figsize=(14, 6))
            plt.bar(monthly_data['month'], monthly_data['flare_count'], color='#D55E00')
            plt.xlabel('Month')
            plt.ylabel('Number of Major Flares')
            plt.title('Major Flare Counts by Month')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'monthly_flare_counts.png'), dpi=300)
            plt.close()

            # Plot monthly flare percentages
            plt.figure(figsize=(14, 6))
            plt.bar(monthly_data['month'], monthly_data['flare_percentage'], color='#0072B2')
            plt.xlabel('Month')
            plt.ylabel('Percentage of Observations with Major Flares')
            plt.title('Major Flare Percentage by Month')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'monthly_flare_percentages.png'), dpi=300)
            plt.close()

        # Get feature columns
        feature_columns = [col for col in labeled_data.columns if col not in ['timestamp', 'label', 'month']]

        if feature_columns and 'label' in labeled_data.columns:
            # Create violin plots for each feature by class
            for feature in feature_columns:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x='label', y=feature, data=labeled_data)
                plt.title(f'Distribution of {feature} by Flare Occurrence')
                plt.xlabel('Flare Occurred (1) vs No Flare (0)')
                plt.ylabel(feature)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'violin_{feature}.png'), dpi=300)
                plt.close()

            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            corr_matrix = labeled_data[feature_columns + ['label']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300)
            plt.close()

            # Create pairplot for features
            if len(feature_columns) <= 5:  # Only do this for a reasonable number of features
                plt.figure(figsize=(15, 15))
                sns.pairplot(labeled_data[feature_columns + ['label']], hue='label', diag_kind='kde')
                plt.savefig(os.path.join(output_dir, 'feature_pairplot.png'), dpi=300)
                plt.close()
    else:
        print(f"Warning: Labeled dataset not found at {labeled_path}")

    # Try to load the models to generate ROC and PR curves
    try:
        # Load SVM model
        svm_model_path = os.path.join(model_dir, 'svm_model.pkl')
        if os.path.exists(svm_model_path):
            svm_model = joblib.load(svm_model_path)

            # Load RF model
            rf_model_path = os.path.join(model_dir, 'rf_model.pkl')
            if os.path.exists(rf_model_path):
                rf_model = joblib.load(rf_model_path)

                # If we have both models and a labeled dataset, create ROC and PR curves
                if os.path.exists(labeled_path):
                    labeled_data = pd.read_csv(labeled_path)
                    feature_columns = [col for col in labeled_data.columns if
                                       col not in ['timestamp', 'label', 'month']]

                    if feature_columns and 'label' in labeled_data.columns:
                        # Prepare data
                        X = labeled_data[feature_columns].values
                        y = labeled_data['label'].values

                        # Get predictions
                        svm_probs = svm_model.predict_proba(X)[:, 1]
                        rf_probs = rf_model.predict_proba(X)[:, 1]

                        # ROC Curve
                        plt.figure(figsize=(10, 8))

                        # SVM ROC
                        fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
                        roc_auc_svm = roc_auc_score(y, svm_probs)
                        plt.plot(fpr_svm, tpr_svm, lw=2, label=f'SVM (AUC = {roc_auc_svm:.3f})')

                        # RF ROC
                        fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)
                        roc_auc_rf = roc_auc_score(y, rf_probs)
                        plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

                        # Random baseline
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)

                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC) Curve')
                        plt.legend(loc='lower right')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(output_dir, 'roc_curve_comparison.png'), dpi=300)
                        plt.close()

                        # PR Curve
                        plt.figure(figsize=(10, 8))

                        # SVM PR
                        precision_svm, recall_svm, _ = precision_recall_curve(y, svm_probs)
                        pr_auc_svm = average_precision_score(y, svm_probs)
                        plt.plot(recall_svm, precision_svm, lw=2, label=f'SVM (AUC = {pr_auc_svm:.3f})')

                        # RF PR
                        precision_rf, recall_rf, _ = precision_recall_curve(y, rf_probs)
                        pr_auc_rf = average_precision_score(y, rf_probs)
                        plt.plot(recall_rf, precision_rf, lw=2, label=f'Random Forest (AUC = {pr_auc_rf:.3f})')

                        # Baseline
                        plt.plot([0, 1], [sum(y) / len(y), sum(y) / len(y)], 'k--', lw=2,
                                 label=f'Baseline ({sum(y) / len(y):.3f})')

                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve')
                        plt.legend(loc='best')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(output_dir, 'pr_curve_comparison.png'), dpi=300)
                        plt.close()
    except Exception as e:
        print(f"Warning: Could not load models or generate ROC/PR curves: {str(e)}")

    print(f"Visualizations complete! Check the {output_dir} directory for output files.")


# Run the visualization
if __name__ == "__main__":
    visualize_results()