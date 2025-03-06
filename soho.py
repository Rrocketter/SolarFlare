import os
import numpy as np
import pandas as pd
import sunpy.map
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import median_filter
import traceback
import logging
import hashlib


def setup_logging(output_dir):
    """Set up logging to capture detailed error information."""
    log_file = os.path.join(output_dir, 'soho_processing_log.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    return logging.getLogger(__name__)


def generate_unique_filename(data, timestamp, instrument, detector):
    """
    Generate a unique filename based on data content and timestamp.

    Args:
        data (np.ndarray): Input data array
        timestamp (datetime): Timestamp of the observation
        instrument (str): Instrument name
        detector (str): Detector name

    Returns:
        str: Unique filename
    """
    # Create a hash of the data array to ensure uniqueness
    data_hash = hashlib.md5(data.tobytes()).hexdigest()[:8]

    # Combine timestamp, instrument, detector, and data hash
    unique_filename = f'{instrument}_{detector}_{timestamp.strftime("%Y%m%d_%H%M%S")}_{data_hash}.npy'

    return unique_filename


def preprocess_soho_data(input_dir, output_dir):
    """
    Enhanced preprocessing for SOHO LASCO C3 L0 RawData with comprehensive error tracking.
    """
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting preprocessing for SOHO LASCO C3 L0 RawData...")

    os.makedirs(output_dir, exist_ok=True)
    fits_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.fts', '.fits'))]

    logger.info(f"Total FITS files found: {len(fits_files)}")

    # Track different types of errors
    error_types = {
        'file_open_errors': 0,
        'data_invalid': 0,
        'timestamp_errors': 0,
        'unexpected_errors': 0
    }

    processed_data = []

    for file in fits_files:
        try:
            # Load FITS file
            try:
                with fits.open(file) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
            except Exception as file_open_error:
                error_types['file_open_errors'] += 1
                logger.error(f"Cannot open file {file}: {file_open_error}")
                continue

            # Validate data
            if data is None or not isinstance(data, np.ndarray) or data.size == 0:
                error_types['data_invalid'] += 1
                logger.warning(f"Invalid data in file {file}")
                continue

            data = data.astype(np.float32)

            # Extract metadata
            filename = os.path.basename(file)
            instrument = header.get('INSTRUME', 'LASCO')
            detector = header.get('DETECTOR', 'C3')
            date_obs = header.get('DATE-OBS', None)
            exposure = header.get('EXPTIME', 1.0)  # Avoid division by zero

            # Robust time parsing
            try:
                if date_obs:
                    # Replace '/' with '-' for standard ISO format
                    date_obs = date_obs.replace('/', '-')
                    timestamp = Time(date_obs, format='isot', scale='utc').datetime
                else:
                    raise ValueError("No DATE-OBS found")
            except Exception as time_parse_error:
                error_types['timestamp_errors'] += 1
                logger.warning(f"Timestamp error in {file}: {time_parse_error}")
                continue

            # Data Cleaning
            data[data <= 0] = np.nan  # Mask invalid pixels
            data = np.nan_to_num(data)  # Replace NaNs with 0

            # Cosmic Ray Removal (Median Filter)
            data_filtered = median_filter(data, size=3)

            # Normalize by Exposure Time
            data_normalized = data_filtered / exposure

            # Generate unique filename
            unique_filename = generate_unique_filename(data_normalized, timestamp, instrument, detector)
            output_filepath = os.path.join(output_dir, unique_filename)

            # Save processed data
            np.save(output_filepath, data_normalized)

            processed_data.append({
                'original_filename': filename,
                'processed_filename': unique_filename,
                'instrument': instrument,
                'detector': detector,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'exposure': exposure
            })

            logger.info(f"Processed {filename}")

        except Exception as unexpected_error:
            error_types['unexpected_errors'] += 1
            logger.error(f"Unexpected error processing {file}: {unexpected_error}")
            logger.error(traceback.format_exc())

    # Save metadata
    metadata_df = pd.DataFrame(processed_data)
    metadata_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)

    # Log summary of processing
    logger.info("Processing complete.")
    logger.info(f"Total files processed: {len(processed_data)}")
    logger.info(f"Error breakdown: {error_types}")

    return metadata_df


def main():
    input_dir = "solar_data/soho"
    output_dir = "data/processed/soho"
    preprocess_soho_data(input_dir, output_dir)


if __name__ == "__main__":
    main()