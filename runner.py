# Complete Solar Flare Prediction Pipeline
# This script runs the complete pipeline from data preprocessing to model evaluation

import os
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Import the modules we created
# Note: In a real implementation, these would be properly packaged
# For this example, assume they're in the same directory
from preprocessing.SDO_GOES_preprocessing import main as preprocess_data
from magnetogram_feature_extraction import MagnetogramFeatureExtractor
from basic import SolarFlarePredictor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Solar Flare Prediction Pipeline')

    parser.add_argument('--sample', action='store_true',
                        help='Run with a small sample dataset (for testing)')

    parser.add_argument('--start-date', type=str, default='2010-05-01',
                        help='Start date for data collection (YYYY-MM-DD)')

    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for data collection (YYYY-MM-DD)')

    parser.add_argument('--forecast-window', type=int, default=24,
                        help='Forecast window in hours (default: 24)')

    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')

    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download and preprocessing (use existing data)')

    return parser.parse_args()


def run_pipeline(args):
    """Run the complete pipeline"""
    # Create results directory
    results_dir = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Data Preprocessing
    if not args.skip_download:
        print("\n======= Step 1: Data Preprocessing =======")
        start_time = time.time()
        preprocess_results = preprocess_data(
            start_date=args.start_date,
            end_date=args.end_date,
            sample=args.sample
        )
        preprocessing_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    else:
        print("\n======= Skipping data download and preprocessing =======")

    # Step 2: Feature Extraction
    print("\n======= Step 2: Physics-Informed Feature Extraction =======")
    start_time = time.time()
    feature_extractor = MagnetogramFeatureExtractor(
        input_dir='1_1_to_1_31_2018/data/processed/magnetograms',
        output_dir='1_1_to_1_31_2018/data/processed/features'
    )
    features_df = feature_extractor.extract_all_features()
    feature_extraction_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")

    # Step 3: Model Training and Evaluation
    print("\n======= Step 3: Model Training and Evaluation =======")
    start_time = time.time()
    predictor = SolarFlarePredictor(
        features_dir='1_1_to_1_31_2018/data/processed/features',
        goes_dir='1_1_to_1_31_2018/data/processed/goes_xray',
        output_dir=results_dir
    )

    score_df = predictor.run_complete_evaluation(
        forecast_window=args.forecast_window,
        cv=args.cv_folds
    )

    # Analyze feature importance
    predictor.feature_importance_analysis()

    model_training_time = time.time() - start_time
    print(f"Model training and evaluation completed in {model_training_time:.2f} seconds")

    # Summary of results
    print("\n======= Pipeline Results Summary =======")
    print(f"Results saved to: {results_dir}")
    print("\nModel Performance Comparison:")
    print(score_df)

    # Plot model comparison
    plt.figure(figsize=(12, 6))
    score_df[['tss', 'hss', 'roc_auc', 'f1_score']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))

    # Save execution times
    execution_times = {
        'preprocessing_time': preprocessing_time if not args.skip_download else None,
        'feature_extraction_time': feature_extraction_time,
        'model_training_time': model_training_time,
        'total_time': (
                          preprocessing_time if not args.skip_download else 0) + feature_extraction_time + model_training_time
    }

    # Save execution times to file
    with open(os.path.join(results_dir, 'execution_times.txt'), 'w') as f:
        for key, value in execution_times.items():
            if value is not None:
                f.write(f"{key}: {value:.2f} seconds\n")
            else:
                f.write(f"{key}: skipped\n")

    return predictor, score_df


if __name__ == "__main__":
    args = parse_arguments()
    predictor, scores = run_pipeline(args)

    print("\nPipeline execution completed successfully!")