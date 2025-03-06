# Baseline Models and Skill Scores for Solar Flare Prediction
# This script implements baseline ML models and calculates skill scores

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, \
    average_precision_score
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')


class SolarFlarePredictor:
    """Class for solar flare prediction using machine learning models"""

    def __init__(self, features_dir='data/processed/features', goes_dir='data/processed/goes_xray',
                 output_dir='baisc_model/models'):
        """Initialize the predictor"""
        self.features_dir = features_dir
        self.goes_dir = goes_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load feature data
        self.features_df = pd.read_csv(os.path.join(features_dir, 'magnetogram_features.csv'))

        # Load GOES flare data
        self.flares_df = pd.read_csv(os.path.join(goes_dir, 'goes_flare_events.csv'))

        self._verify_flare_coverage()

        # Convert timestamps to datetime
        self.features_df['timestamp'] = pd.to_datetime(self.features_df['timestamp'])
        self.flares_df['start_time'] = pd.to_datetime(self.flares_df['start_time'])
        self.flares_df['peak_time'] = pd.to_datetime(self.flares_df['peak_time'])
        self.flares_df['end_time'] = pd.to_datetime(self.flares_df['end_time'])



        print(
            f"Initialized predictor with {len(self.features_df)} feature records and {len(self.flares_df)} flare events")

    def _verify_flare_coverage(self):
        """Check if known major flares exist in the dataset"""
        # October 2014 X1.1 flare
        x_flare_time = pd.Timestamp("2014-10-25 17:08:00")

        # Check if flare exists in database
        exists_in_db = self.flares_df[
            (self.flares_df['peak_time'] == x_flare_time) &
            (self.flares_df['class'].str.startswith('X'))
            ].any().any()

        # Check if any magnetogram captures it
        in_features = self.features_df[
            (self.features_df['timestamp'] >= x_flare_time - pd.Timedelta(hours=24)) &
            (self.features_df['timestamp'] <= x_flare_time)
            ].any().any()

        print(f"X1.1 flare in database: {'YES' if exists_in_db else 'NO'}")
        print(f"Magnetogram within 24h of flare: {'YES' if in_features else 'NO'}")

    def create_labeled_dataset(self, forecast_window=24, major_flare_classes=['X', 'M']):
        """
        Create a labeled dataset for training and testing

        Parameters:
        forecast_window (int): Forecast window in hours
        major_flare_classes (list): List of flare classes considered as positive events

        Returns:
        tuple: X (features), y (labels)
        """
        print(f"Creating labeled dataset with {forecast_window}h forecast window")
        print(f"Major flare classes: {major_flare_classes}")

        # Initialize labels as zeros (no flare)
        self.features_df['label'] = 0

        # For each magnetogram timestamp, check if a major flare occurs within the forecast window
        for i, row in self.features_df.iterrows():
            timestamp = row['timestamp']
            end_time = timestamp + timedelta(hours=forecast_window)

            # Check if any major flare starts within the forecast window
            major_flares = self.flares_df[
                (self.flares_df['start_time'] >= timestamp) &
                (self.flares_df['start_time'] <= end_time) &
                (self.flares_df['class'].str[0].isin(major_flare_classes))
                ]

            # If there's at least one major flare, set label to 1
            if len(major_flares) > 0:
                self.features_df.at[i, 'label'] = 1

        # Extract features and labels
        feature_columns = [
            'r_value', 'magnetic_shear', 'current_helicity',
            'ising_energy', 'lorentz_force', 'total_unsigned_flux',
            'max_field_strength'
        ]

        X = self.features_df[feature_columns].values
        y = self.features_df['label'].values

        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count / len(y) * 100:.2f}%)")

        return X, y

    def train_test_split(self, X, y, test_size=0.25, random_state=42):
        """Split the data into training and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def train_svm_model(self, X_train, y_train, cv=5):
        """Train a Support Vector Machine model"""
        print("Training SVM model...")

        # Define the pipeline with preprocessing and model
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])

        # Define parameter grid for grid search
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.1, 0.01],
            'svm__kernel': ['rbf', 'linear']
        }

        # Set up grid search with cross-validation
        grid_search = GridSearchCV(
            svm_pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train the model
        grid_search.fit(X_train, y_train)

        # Get the best model
        self.svm_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Save the model
        model_path = os.path.join(self.output_dir, 'svm_model.pkl')
        joblib.dump(self.svm_model, model_path)
        print(f"Saved SVM model to {model_path}")

        return self.svm_model

    def train_rf_model(self, X_train, y_train, cv=5):
        """Train a Random Forest model"""
        print("Training Random Forest model...")

        # Define the pipeline with preprocessing and model
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # Define parameter grid for grid search
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        # Set up grid search with cross-validation
        grid_search = GridSearchCV(
            rf_pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train the model
        grid_search.fit(X_train, y_train)

        # Get the best model
        self.rf_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Save the model
        model_path = os.path.join(self.output_dir, 'rf_model.pkl')
        joblib.dump(self.rf_model, model_path)
        print(f"Saved Random Forest model to {model_path}")

        return self.rf_model

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a model and calculate skill scores

        Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display

        Returns:
        dict: Dictionary of skill scores
        """
        print(f"\nEvaluating {model_name} model...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate skill scores

        # True Skill Statistic (TSS) = POD - POFD
        # Also known as the Hanssen-Kuiper Skill Score
        pod = tp / (tp + fn)  # Probability of Detection
        pofd = fp / (fp + tn)  # Probability of False Detection
        tss = pod - pofd

        # Heidke Skill Score (HSS)
        # HSS = 2 * (TP*TN - FP*FN) / ((TP+FN) * (FN+TN) + (TP+FP) * (FP+TN))
        hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_prob)

        # Precision-Recall AUC
        pr_auc = average_precision_score(y_test, y_prob)

        # Calculate F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = pod  # Same as POD
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Create a dictionary of scores
        scores = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tss': tss,
            'hss': hss,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

        # Print the scores
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (POD): {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Skill Statistic (TSS): {tss:.4f}")
        print(f"Heidke Skill Score (HSS): {hss:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(10, 5))

        # ROC curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

        # Precision-Recall curve
        plt.subplot(1, 2, 2)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_curves.png'))
        plt.close()

        return scores

    def run_complete_evaluation(self, forecast_window=24, cv=5):
        """Run complete training and evaluation pipeline"""
        # Create labeled dataset
        X, y = self.create_labeled_dataset(forecast_window=forecast_window)

        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        # Train models
        svm_model = self.train_svm_model(X_train, y_train, cv=cv)
        rf_model = self.train_rf_model(X_train, y_train, cv=cv)

        # Evaluate models
        svm_scores = self.evaluate_model(svm_model, X_test, y_test, 'SVM')
        rf_scores = self.evaluate_model(rf_model, X_test, y_test, 'RandomForest')

        # Compare models
        print("\nModel Comparison:")
        score_df = pd.DataFrame([svm_scores, rf_scores])
        score_df.set_index('model', inplace=True)
        print(score_df)

        # Save scores
        score_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'))

        return score_df

    def feature_importance_analysis(self):
        """Analyze feature importance from the Random Forest model"""
        if not hasattr(self, 'rf_model'):
            print("Random Forest model not trained yet.")
            return

        # Extract feature names
        feature_columns = [
            'r_value', 'magnetic_shear', 'current_helicity',
            'ising_energy', 'lorentz_force', 'total_unsigned_flux',
            'max_field_strength'
        ]

        # Get feature importances
        importances = self.rf_model.named_steps['rf'].feature_importances_

        # Sort feature importances
        indices = np.argsort(importances)[::-1]

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance from Random Forest')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()

        # Print feature importances
        print("\nFeature Importance:")
        for i in indices:
            print(f"{feature_columns[i]}: {importances[i]:.4f}")

        return importances




# Main function to run the complete pipeline
def main():
    # Initialize and run the predictor
    predictor = SolarFlarePredictor()

    # Run complete evaluation
    score_df = predictor.run_complete_evaluation(forecast_window=24, cv=3)  # Using 3-fold CV for faster execution

    # Analyze feature importance
    predictor.feature_importance_analysis()

    return predictor


# If running as a script
if __name__ == "__main__":
    main()