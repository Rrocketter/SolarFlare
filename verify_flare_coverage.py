# solarflare_predictor.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib
import warnings
from urllib.request import urlretrieve

warnings.filterwarnings('ignore')


class SolarFlarePredictor:
    """Updated solar flare predictor with validated data sources"""

    def __init__(self, features_dir='data/processed/features',
                 goes_year_month=201410,
                 output_dir='basic_model/models'):
        self.features_dir = features_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess data
        self._load_features()
        self._load_goes_flares(goes_year_month)
        self._verify_data_quality()

    def _load_features(self):
        """Load and preprocess magnetogram features"""
        self.features_df = pd.read_csv(
            os.path.join(self.features_dir, 'magnetogram_features.csv'),
            parse_dates=['timestamp']
        )
        # Ensure UTC timezone
        self.features_df['timestamp'] = pd.to_datetime(self.features_df['timestamp']).dt.tz_localize('UTC')

    def _load_goes_flares(self, year_month):
        """Load GOES flare data directly from NOAA"""
        base_url = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"
        filename = f"goes-xrs-report_{year_month}.txt"
        goes_url = f"{base_url}{filename}"

        # Download and parse flare data
        self.flares_df = pd.read_fwf(
            goes_url,
            skiprows=116,
            widths=[6, 8, 13, 13, 13, 8, 8, 8, 8, 8, 8],
            names=['year', 'month', 'day', 'start_time', 'end_time', 'peak_time',
                   'x_class', 'long_arcsec', 'lat_arcsec', 'region', 'quality']
        )

        # Convert time columns to datetime
        self.flares_df['peak_time'] = pd.to_datetime(
            self.flares_df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1) + ' ' +
            self.flares_df['peak_time'].str.strip(),
            format='%Y-%m-%d %H%M%S',
            errors='coerce'
        ).dt.tz_localize('UTC')

        # Filter valid entries
        self.flares_df = self.flares_df.dropna(subset=['peak_time'])
        self.flares_df = self.flares_df[self.flares_df['quality'] == 0]

        # Create flare class labels
        self.flares_df['class'] = self.flares_df['x_class'].apply(
            lambda x: 'X' if x >= 1e-6 else 'M' if x >= 1e-7 else 'C' if x >= 1e-8 else 'B'
        )

    def _verify_data_quality(self):
        """Validate critical data points"""
        # Check for known X1.1 flare
        x_flare_time = pd.Timestamp("2014-10-25 17:08:00").tz_localize('UTC')
        flare_exists = not self.flares_df[
            (self.flares_df['peak_time'] == x_flare_time) &
            (self.flares_df['class'] == 'X')
            ].empty

        mag_exists = not self.features_df[
            (self.features_df['timestamp'] >= x_flare_time - pd.Timedelta(hours=24)) &
            (self.features_df['timestamp'] <= x_flare_time)
            ].empty

        print(f"Data Quality Check:")
        print(f"X1.1 flare in database: {'✅' if flare_exists else '❌'}")
        print(f"Magnetograms in window: {'✅' if mag_exists else '❌'}")
        print(f"Total flares loaded: {len(self.flares_df)}")
        print(f"Total magnetograms: {len(self.features_df)}\n")

    def create_labeled_dataset(self, forecast_window=24, major_flare_classes=['X', 'M']):
        """Create labeled dataset with validation"""
        self.features_df['label'] = 0

        # Convert to numpy datetime64 for performance
        mag_times = self.features_df['timestamp'].values.astype('datetime64[s]')
        flare_times = self.flares_df['peak_time'].values.astype('datetime64[s]')
        flare_classes = self.flares_df['class'].values

        for i, mag_time in enumerate(mag_times):
            window_end = mag_time + np.timedelta64(forecast_window, 'h')
            mask = (flare_times >= mag_time) & (flare_times <= window_end)
            relevant_flares = flare_classes[mask]

            if any(cls in major_flare_classes for cls in relevant_flares):
                self.features_df.at[i, 'label'] = 1

        # Validate class balance
        y = self.features_df['label'].values
        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            raise ValueError(f"Only one class present (counts: {class_counts})")

        print(f"Class balance: {class_counts[0]} negative, {class_counts[1]} positive samples")
        return self.features_df.drop(columns=['label']).values, y

    # Keep other methods (train_test_split, train_svm_model, etc.) unchanged from previous version


if __name__ == "__main__":
    # Example usage
    predictor = SolarFlarePredictor(goes_year_month=201410)

    try:
        X, y = predictor.create_labeled_dataset(forecast_window=24)
        print("Data preparation successful. Proceeding to training...")

        # Example training
        X_train, X_test, y_train, y_test = predictor.train_test_split(X, y)
        svm_model = predictor.train_svm_model(X_train, y_train)

    except ValueError as e:
        print(f"Critical error: {e}")
        print("Check your data pipeline and flare database")