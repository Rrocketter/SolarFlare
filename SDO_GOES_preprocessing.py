import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sunpy.map
import sunpy.io
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import xarray as xr
import warnings
from scipy import signal
from scipy.ndimage import gaussian_filter
import pywt
import skimage.measure
from scipy.interpolate import griddata
import glob


warnings.filterwarnings('ignore')

# Update paths to match your actual data location
DATA_ROOT = 'solar_data'  # Change this to your data root
PROCESSED_ROOT = 'data/processed'  # Keep processed data in original location


# Create directories for data storage
def create_directories():
    directories = [
        f'{PROCESSED_ROOT}/magnetograms',
        f'{PROCESSED_ROOT}/aia_images',
        f'{PROCESSED_ROOT}/goes_xray',
        f'{PROCESSED_ROOT}/soho_data',    # Added SOHO directory
        f'{PROCESSED_ROOT}/features',
        f'{PROCESSED_ROOT}/combined'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Created directory structure for data storage")


def inspect_file_structure(directory):
    """Print information about files in a directory"""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return

    files = os.listdir(directory)
    print(f"Directory {directory} contains {len(files)} files")
    if len(files) > 0:
        print("Sample files:")
        for f in files[:5]:
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path) / 1024  # Size in KB
                print(f"  - {f} ({size:.1f} KB)")
            else:
                print(f"  - {f} (directory)")


def preprocess_hmi_data(input_dir=f'{DATA_ROOT}/sdo_hmi', output_dir=f'{PROCESSED_ROOT}/magnetograms'):
    """Preprocess HMI vector magnetogram data"""
    print("Preprocessing HMI vector magnetogram data")

    # Get list of files based on actual naming pattern
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        # Create empty metadata file to avoid downstream errors
        pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude', 'carrington_rotation', 'center']
                     ).to_csv(os.path.join(output_dir, 'magnetogram_metadata.csv'), index=False)
        return pd.DataFrame()

    # Find all FITS files that match the magnetogram pattern
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.endswith('.fits') and 'magnetogram' in f]

    print(f"Found {len(files)} HMI magnetogram files")

    processed_data = []

    for i, file in enumerate(files):
        if i % 10 == 0:
            print(f"Processing file {i + 1}/{len(files)}")

        try:
            # Load magnetogram
            hmi_map = sunpy.map.Map(file)

            # Extract the data array (magnetogram)
            magnetogram_data = hmi_map.data

            # Create timestamp from metadata
            if 'date-obs' in hmi_map.meta:
                timestamp = Time(hmi_map.meta['date-obs']).datetime
            else:
                # Extract from filename as fallback
                # Format: hmi.m_45s.2018.01.01_00_01_30_TAI.magnetogram.fits
                date_str = os.path.basename(file).split('.')[2].split('_TAI')[0]
                timestamp = datetime.strptime(date_str, '%Y.%m.%d_%H_%M_%S')

            # Save processed data
            output_file = os.path.join(output_dir, f'magnetogram_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy')
            np.save(output_file, magnetogram_data)

            # Store metadata
            processed_data.append({
                'filename': output_file,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'carrington_longitude': hmi_map.carrington_longitude.value if hasattr(hmi_map,
                                                                                      'carrington_longitude') else None,
                'carrington_rotation': hmi_map.carrington_rotation if hasattr(hmi_map, 'carrington_rotation') else None,
                'center': [hmi_map.reference_coordinate.Tx.value, hmi_map.reference_coordinate.Ty.value] if hasattr(
                    hmi_map, 'reference_coordinate') else [0, 0]
            })

            print(f"Successfully processed {os.path.basename(file)}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Save metadata
    metadata_df = pd.DataFrame(processed_data)
    if not metadata_df.empty:
        metadata_df.to_csv(os.path.join(output_dir, 'magnetogram_metadata.csv'), index=False)
        print(f"Processed {len(processed_data)} HMI magnetogram files")
    else:
        print("Warning: No magnetogram files were successfully processed")
        # Create an empty metadata file with headers to avoid downstream errors
        pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude', 'carrington_rotation', 'center']
                     ).to_csv(os.path.join(output_dir, 'magnetogram_metadata.csv'), index=False)

    return metadata_df


# def preprocess_aia_data(input_dir=f'{DATA_ROOT}/sdo_aia', output_dir=f'{PROCESSED_ROOT}/aia_images'):
#     """Preprocess AIA multi-wavelength images"""
#     print("Preprocessing AIA multi-wavelength images")
#
#     # Check if directory exists
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         pd.DataFrame(columns=['filename', 'wavelength', 'timestamp']
#                      ).to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)
#         return pd.DataFrame()
#
#     # Get available wavelength directories
#     wavelength_dirs = [d for d in os.listdir(input_dir)
#                        if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()]
#
#     if not wavelength_dirs:
#         # Try alternate approach with wavelength folders like "171"
#         wavelength_dirs = [d for d in os.listdir(input_dir)
#                            if os.path.isdir(os.path.join(input_dir, d))]
#
#     print(f"Found wavelength directories: {wavelength_dirs}")
#
#     processed_data = []
#
#     for wave_dir in wavelength_dirs:
#         try:
#             # Extract wavelength from directory name
#             wavelength = int(wave_dir)
#         except ValueError:
#             print(f"Skipping non-numeric directory: {wave_dir}")
#             continue
#
#         wave_path = os.path.join(input_dir, wave_dir)
#         files = [os.path.join(wave_path, f) for f in os.listdir(wave_path)
#                  if f.endswith('.fits') and not f.startswith('.')]
#
#         print(f"Found {len(files)} AIA {wave_dir}Å images")
#
#         for i, file in enumerate(files):
#             if i % 10 == 0:
#                 print(f"Processing {wave_dir}Å image {i + 1}/{len(files)}")
#
#             try:
#                 # Load AIA image
#                 aia_map = sunpy.map.Map(file)
#
#                 # Extract timestamp from metadata or filename
#                 if 'date-obs' in aia_map.meta:
#                     timestamp = Time(aia_map.meta['date-obs']).datetime
#                 else:
#                     # Example filename: aia.lev1.171A_2018_01_01T00_00_09.35Z.image_lev1.fits
#                     # Extract date part
#                     date_part = os.path.basename(file).split('_')[1:4]
#                     time_part = os.path.basename(file).split('T')[1].split('.')[0]
#                     date_str = f"{date_part[0]}_{date_part[1]}_{date_part[2]}T{time_part}"
#                     timestamp = datetime.strptime(date_str, '%Y_%m_%dT%H_%M_%S')
#
#                 # Save processed data
#                 output_file = os.path.join(output_dir, f'aia_{wavelength}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy')
#                 np.save(output_file, aia_map.data)
#
#                 # Store metadata
#                 processed_data.append({
#                     'filename': output_file,
#                     'wavelength': wavelength,
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
#                 })
#
#                 print(f"Successfully processed {os.path.basename(file)}")
#
#             except Exception as e:
#                 print(f"Error processing file {file}: {e}")
#
#     # Save metadata
#     metadata_df = pd.DataFrame(processed_data)
#     if not metadata_df.empty:
#         metadata_df.to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)
#         print(f"Processed {len(processed_data)} AIA images")
#     else:
#         print("Warning: No AIA images were successfully processed")
#         pd.DataFrame(columns=['filename', 'wavelength', 'timestamp']
#                      ).to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)
#
#     return metadata_df


def preprocess_aia_data(input_dir=f'{DATA_ROOT}/sdo_aia', output_dir=f'{PROCESSED_ROOT}/aia_images'):
    """Preprocess AIA multi-wavelength images"""
    print("Preprocessing AIA multi-wavelength images")

    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        pd.DataFrame(columns=['filename', 'wavelength', 'timestamp']
                     ).to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)
        return pd.DataFrame()

    # Get available wavelength directories
    wavelength_dirs = [d for d in os.listdir(input_dir)
                       if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()]

    if not wavelength_dirs:
        # Try alternate approach with wavelength folders like "171"
        wavelength_dirs = [d for d in os.listdir(input_dir)
                           if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Found wavelength directories: {wavelength_dirs}")

    processed_data = []

    for wave_dir in wavelength_dirs:
        try:
            # Extract wavelength from directory name
            wavelength = int(wave_dir)
        except ValueError:
            print(f"Skipping non-numeric directory: {wave_dir}")
            continue

        wave_path = os.path.join(input_dir, wave_dir)
        files = [os.path.join(wave_path, f) for f in os.listdir(wave_path)
                 if f.endswith('.fits') and not f.startswith('.')]

        print(f"Found {len(files)} AIA {wave_dir}Å images")

        for i, file in enumerate(files):
            if i % 10 == 0:
                print(f"Processing {wave_dir}Å image {i + 1}/{len(files)}")

            try:
                # Add verification step to check file integrity
                try:
                    with open(file, 'rb') as f:
                        # Check if file is readable and has content
                        data = f.read(1024)  # Just read a small chunk to verify
                        if not data:
                            print(f"Empty or unreadable file: {file}, skipping")
                            continue
                except IOError as e:
                    print(f"IO Error with file {file}: {e}, skipping")
                    continue

                # Try to create the map directly
                try:
                    aia_map = sunpy.map.Map(file)

                    # Check if data is valid
                    if aia_map.data.size == 0 or np.all(np.isnan(aia_map.data)):
                        print(f"Invalid data in {file}, skipping")
                        continue

                except Exception as e:
                    print(f"Error reading FITS file {file}: {e}, skipping")
                    continue

                # Extract timestamp from metadata or filename
                if 'date-obs' in aia_map.meta:
                    timestamp = Time(aia_map.meta['date-obs']).datetime
                else:
                    # Example filename: aia.lev1.171A_2018_01_01T00_00_09.35Z.image_lev1.fits
                    # Extract date part
                    date_part = os.path.basename(file).split('_')[1:4]
                    time_part = os.path.basename(file).split('T')[1].split('.')[0]
                    date_str = f"{date_part[0]}_{date_part[1]}_{date_part[2]}T{time_part}"
                    timestamp = datetime.strptime(date_str, '%Y_%m_%dT%H_%M_%S')

                # Add safeguard for output file
                try:
                    # Save processed data using compressed format
                    output_file = os.path.join(output_dir, f'aia_{wavelength}_{timestamp.strftime("%Y%m%d_%H%M%S")}')

                    # Convert data to float32 if it's not already to reduce size
                    data_to_save = aia_map.data.astype(np.float32) if aia_map.data.dtype != np.float32 else aia_map.data

                    # Compress data to save space and improve I/O performance
                    np.savez_compressed(output_file, data=data_to_save)

                    # Store metadata
                    processed_data.append({
                        'filename': output_file + '.npz',  # Note .npz extension
                        'wavelength': wavelength,
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })

                    print(f"Successfully processed {os.path.basename(file)}")

                except Exception as e:
                    print(f"Error saving processed file for {file}: {e}")

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Save metadata
    metadata_df = pd.DataFrame(processed_data)
    if not metadata_df.empty:
        metadata_df.to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)
        print(f"Processed {len(processed_data)} AIA images")
    else:
        print("Warning: No AIA images were successfully processed")
        pd.DataFrame(columns=['filename', 'wavelength', 'timestamp']
                     ).to_csv(os.path.join(output_dir, 'aia_metadata.csv'), index=False)

    return metadata_df

# def preprocess_goes_data(input_dir=f'{DATA_ROOT}/goes', output_dir=f'{PROCESSED_ROOT}/goes_xray'):
#     """Preprocess GOES X-ray flux data and identify flare events"""
#     print("Preprocessing GOES X-ray flux data")
#
#     # Check if directory exists
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         pd.DataFrame(columns=['timestamp', 'xray_long', 'xray_short']
#                      ).to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class']
#                      ).to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#         return pd.DataFrame(), pd.DataFrame()
#
#     # Get list of files - adjust the pattern based on your actual filenames
#     files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
#              if f.endswith('.nc') and 'sci_gxrs' in f]
#
#     if not files:
#         # Try different file pattern
#         files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
#                  if f.endswith('.nc')]
#
#     print(f"Found {len(files)} GOES data files")
#
#     # Initialize lists to store data
#     timestamps = []
#     xray_long = []  # 1-8Å (GOES A channel)
#     xray_short = []  # 0.5-4Å (GOES B channel)
#
#     for file in files:
#         try:
#             # Open NetCDF file
#             ds = xr.open_dataset(file)
#             print(f"Processing {os.path.basename(file)}")
#
#             # Print available variables to help debug
#             print(f"Available variables: {list(ds.variables)}")
#
#             # Extract time data - adjust variable names based on your specific file format
#             time_var = None
#             for var_name in ['time', 'time_tag', 'date']:
#                 if var_name in ds:
#                     time_var = var_name
#                     break
#
#             if time_var is None:
#                 print(f"Warning: No time variable found in {file}")
#                 continue
#
#             time_data = ds[time_var].values
#
#             # For A channel (1-8Å, long)
#             a_flux = None
#             for var_name in ['A_FLUX', 'a_flux', 'xrsa_flux', 'xray_long', 'irrad_long', 'flux_long']:
#                 if var_name in ds:
#                     a_flux = ds[var_name].values
#                     print(f"Found A channel (long) as {var_name}")
#                     break
#
#             # For B channel (0.5-4Å, short)
#             b_flux = None
#             for var_name in ['B_FLUX', 'b_flux', 'xrsb_flux', 'xray_short', 'irrad_short', 'flux_short']:
#                 if var_name in ds:
#                     b_flux = ds[var_name].values
#                     print(f"Found B channel (short) as {var_name}")
#                     break
#
#             if a_flux is None or b_flux is None:
#                 print(f"Warning: Could not find flux data in {file}")
#                 continue
#
#             # Ensure they're all the same length
#             min_len = min(len(time_data), len(a_flux), len(b_flux))
#
#             # Append to lists
#             timestamps.extend(time_data[:min_len])
#             xray_long.extend(a_flux[:min_len])
#             xray_short.extend(b_flux[:min_len])
#
#             print(f"Successfully processed {min_len} records from {os.path.basename(file)}")
#
#         except Exception as e:
#             print(f"Error processing GOES file {file}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # Create DataFrame
#     if not timestamps:
#         print("Warning: No GOES data was successfully processed")
#         goes_df = pd.DataFrame(columns=['timestamp', 'xray_long', 'xray_short'])
#         flares_df = pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class'])
#
#         # Save empty DataFrames to avoid downstream errors
#         goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#         return goes_df, flares_df
#
#     goes_df = pd.DataFrame({
#         'timestamp': timestamps,
#         'xray_long': xray_long,
#         'xray_short': xray_short
#     })
#
#     # Convert timestamps to datetime if needed
#     if not isinstance(goes_df['timestamp'].iloc[0], datetime):
#         goes_df['timestamp'] = pd.to_datetime(goes_df['timestamp'])
#
#     # Sort by timestamp
#     goes_df = goes_df.sort_values('timestamp')
#
#     # Identify flare events
#     # Define flare classification thresholds (W/m^2)
#     class_thresholds = {
#         'X': 1e-4,
#         'M': 1e-5,
#         'C': 1e-6,
#         'B': 1e-7
#     }
#
#     # Identify flares
#     flares = []
#     i = 0
#     while i < len(goes_df) - 1:
#         # If flux exceeds B-class threshold and is rising
#         if (goes_df.iloc[i]['xray_long'] > class_thresholds['B'] and
#                 goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#
#             # Find peak
#             start_idx = i
#             while (i < len(goes_df) - 1 and
#                    goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#                 i += 1
#             peak_idx = i
#
#             # Find end (when flux drops below half the peak)
#             while (i < len(goes_df) - 1 and
#                    goes_df.iloc[i]['xray_long'] > goes_df.iloc[peak_idx]['xray_long'] * 0.5):
#                 i += 1
#             end_idx = i
#
#             # Determine flare class
#             peak_flux = goes_df.iloc[peak_idx]['xray_long']
#             flare_class = None
#             for cls, threshold in class_thresholds.items():
#                 if peak_flux >= threshold:
#                     flare_class = cls
#                     magnitude = peak_flux / threshold
#                     break
#
#             if flare_class:
#                 flares.append({
#                     'start_time': goes_df.iloc[start_idx]['timestamp'],
#                     'peak_time': goes_df.iloc[peak_idx]['timestamp'],
#                     'end_time': goes_df.iloc[end_idx]['timestamp'],
#                     'peak_flux': peak_flux,
#                     'class': f"{flare_class}{magnitude:.1f}"
#                 })
#         else:
#             i += 1
#
#     # Create flares DataFrame
#     flares_df = pd.DataFrame(flares)
#
#     # Save processed data
#     goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#
#     if not flares_df.empty:
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#         print(f"Processed GOES X-ray flux data and identified {len(flares)} flare events")
#     else:
#         pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class']
#                      ).to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#         print("Processed GOES X-ray flux data but no flare events were identified")
#
#     return goes_df, flares_df

# def preprocess_goes_data(input_dir=f'{DATA_ROOT}/goes', output_dir=f'{PROCESSED_ROOT}/goes_xray'):
#     print("Preprocessing GOES X-ray flux data")
#
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         # Create empty DataFrames
#         goes_df = pd.DataFrame(columns=['timestamp', 'xray_long', 'xray_short'])
#         flares_df = pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class', 'magnitude'])
#
#         goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#         return goes_df, flares_df
#
#     # List all possible NetCDF files
#     files = [
#         os.path.join(input_dir, f) for f in os.listdir(input_dir)
#         if f.endswith('.nc') and ('sci_gxrs' in f.lower() or 'goes' in f.lower())
#     ]
#
#     if not files:
#         print(f"No NetCDF files found in {input_dir}")
#         # Create empty DataFrames
#         goes_df = pd.DataFrame(columns=['timestamp', 'xray_long', 'xray_short'])
#         flares_df = pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class', 'magnitude'])
#
#         goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#         return goes_df, flares_df
#
#     # Verbose logging of found files
#     print("Found GOES data files:")
#     for f in files:
#         print(f"  - {os.path.basename(f)}")
#
#     all_timestamps = []
#     all_xray_long = []
#     all_xray_short = []
#
#     for file in files:
#         try:
#             # Open the NetCDF dataset
#             with xr.open_dataset(file) as ds:
#                 print(f"\nProcessing file: {os.path.basename(file)}")
#                 print("Available variables:")
#                 for var_name in ds.variables:
#                     print(f"  - {var_name}")
#
#                 # Attempt to find time variables
#                 time_candidates = ['time', 'time_tag', 'date', 'datetime']
#                 time_var = None
#                 for candidate in time_candidates:
#                     if candidate in ds.variables:
#                         time_var = candidate
#                         break
#
#                 if time_var is None:
#                     print(f"Warning: No time variable found in {file}")
#                     continue
#
#                 # Attempt to find flux variables
#                 flux_candidates = {
#                     'long': ['A_FLUX', 'a_flux', 'xrsa_flux', 'xray_long', 'flux_a'],
#                     'short': ['B_FLUX', 'b_flux', 'xrsb_flux', 'xray_short', 'flux_b']
#                 }
#
#                 # Convert time to pandas datetime
#                 times = pd.to_datetime(ds[time_var].values)
#
#                 # Find flux variables
#                 long_flux_var = None
#                 short_flux_var = None
#
#                 for candidate in flux_candidates['long']:
#                     if candidate in ds.variables:
#                         long_flux_var = candidate
#                         break
#
#                 for candidate in flux_candidates['short']:
#                     if candidate in ds.variables:
#                         short_flux_var = candidate
#                         break
#
#                 if long_flux_var is None or short_flux_var is None:
#                     print(f"Warning: Could not find flux variables in {file}")
#                     continue
#
#                 # Extract data
#                 long_flux = ds[long_flux_var].values
#                 short_flux = ds[short_flux_var].values
#
#                 # Ensure consistent length
#                 min_len = min(len(times), len(long_flux), len(short_flux))
#                 times = times[:min_len]
#                 long_flux = long_flux[:min_len]
#                 short_flux = short_flux[:min_len]
#
#                 # Extend lists
#                 all_timestamps.extend(times)
#                 all_xray_long.extend(long_flux)
#                 all_xray_short.extend(short_flux)
#
#                 print(f"Successfully processed {min_len} records from {os.path.basename(file)}")
#
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # Create DataFrame
#     if all_timestamps:
#         goes_df = pd.DataFrame({
#             'timestamp': all_timestamps,
#             'xray_long': all_xray_long,
#             'xray_short': all_xray_short
#         })
#
#         # Sort and remove duplicates
#         goes_df = goes_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
#
#         # Identify flares
#         class_thresholds = {
#             'X': 1e-4,
#             'M': 1e-5,
#             'C': 1e-6,
#             'B': 1e-7
#         }
#
#         flares = []
#         goes_df['xray_long'] = pd.to_numeric(goes_df['xray_long'], errors='coerce')
#         goes_df = goes_df.dropna(subset=['xray_long'])
#
#         i = 0
#         while i < len(goes_df) - 1:
#             if (goes_df.iloc[i]['xray_long'] > class_thresholds['B'] and
#                     goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#
#                 start_idx = i
#                 while (i < len(goes_df) - 1 and
#                        goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#                     i += 1
#                 peak_idx = i
#
#                 while (i < len(goes_df) - 1 and
#                        goes_df.iloc[i]['xray_long'] > goes_df.iloc[peak_idx]['xray_long'] * 0.5):
#                     i += 1
#                 end_idx = i
#
#                 peak_flux = goes_df.iloc[peak_idx]['xray_long']
#                 flare_class = None
#                 for cls, threshold in class_thresholds.items():
#                     if peak_flux >= threshold:
#                         flare_class = cls
#                         magnitude = peak_flux / threshold
#                         break
#
#                 if flare_class:
#                     flares.append({
#                         'start_time': goes_df.iloc[start_idx]['timestamp'],
#                         'peak_time': goes_df.iloc[peak_idx]['timestamp'],
#                         'end_time': goes_df.iloc[end_idx]['timestamp'],
#                         'peak_flux': peak_flux,
#                         'class': f"{flare_class}{magnitude:.1f}",
#                         'magnitude': magnitude
#                     })
#             else:
#                 i += 1
#
#         flares_df = pd.DataFrame(flares)
#
#         # Save to CSV
#         goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#         print(f"Processed {len(goes_df)} GOES records")
#         print(f"Identified {len(flares_df)} flare events")
#
#         return goes_df, flares_df
#     else:
#         print("No GOES data could be processed")
#         # Create empty DataFrames
#         goes_df = pd.DataFrame(columns=['timestamp', 'xray_long', 'xray_short'])
#         flares_df = pd.DataFrame(columns=['start_time', 'peak_time', 'end_time', 'peak_flux', 'class', 'magnitude'])
#
#         goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#         flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#         return goes_df, flares_df

# def preprocess_goes_data(input_dir=f'{DATA_ROOT}/goes', output_dir=f'{PROCESSED_ROOT}/goes_xray'):
#     print("Preprocessing GOES X-ray flux data")
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         return pd.DataFrame(), pd.DataFrame()
#
#     # Use glob to find all .nc files
#     files = glob.glob(os.path.join(input_dir, '*.nc'))
#
#     if not files:
#         print(f"No NetCDF files found in {input_dir}")
#         return pd.DataFrame(), pd.DataFrame()
#
#     print(f"Found {len(files)} GOES data files")
#
#     # Lists to store processed data
#     all_timestamps = []
#     all_xray_long = []
#     all_xray_short = []
#
#     for file in files:
#         try:
#             # Open the dataset using netCDF4 directly
#             import netCDF4
#             nc = netCDF4.Dataset(file, 'r')
#
#             print(f"\nProcessing file: {os.path.basename(file)}")
#             print("Available variables:")
#             for var_name in nc.variables:
#                 print(f"  - {var_name}")
#
#             # Try to find time variable
#             time_candidates = ['time', 'time_tag', 'date', 'datetime']
#             time_var = None
#             for candidate in time_candidates:
#                 if candidate in nc.variables:
#                     time_var = candidate
#                     break
#
#             if time_var is None:
#                 print(f"Warning: No time variable found in {file}")
#                 continue
#
#             # Convert time to datetime
#             times_raw = nc.variables[time_var][:]
#
#             # Determine the base time for conversion
#             if hasattr(nc.variables[time_var], 'units'):
#                 time_units = nc.variables[time_var].units
#                 if 'since' in time_units:
#                     base_time = pd.to_datetime(time_units.split('since')[1].strip())
#                     times = [base_time + pd.Timedelta(seconds=float(t)) for t in times_raw]
#                 else:
#                     times = pd.to_datetime(times_raw)
#             else:
#                 times = pd.to_datetime(times_raw)
#
#             # Find flux variables
#             flux_candidates = {
#                 'long': ['A_FLUX', 'a_flux', 'xrsa_flux', 'xray_long', 'flux_a'],
#                 'short': ['B_FLUX', 'b_flux', 'xrsb_flux', 'xray_short', 'flux_b']
#             }
#
#             # Try to find long and short wavelength flux variables
#             long_flux_var = next((var for var in flux_candidates['long'] if var in nc.variables), None)
#             short_flux_var = next((var for var in flux_candidates['short'] if var in nc.variables), None)
#
#             if long_flux_var is None or short_flux_var is None:
#                 print(f"Warning: Could not find flux variables in {file}")
#                 continue
#
#             # Extract flux data
#             long_flux = nc.variables[long_flux_var][:]
#             short_flux = nc.variables[short_flux_var][:]
#
#             # Ensure consistent length
#             min_len = min(len(times), len(long_flux), len(short_flux))
#             times = times[:min_len]
#             long_flux = long_flux[:min_len]
#             short_flux = short_flux[:min_len]
#
#             # Extend lists
#             all_timestamps.extend(times)
#             all_xray_long.extend(long_flux)
#             all_xray_short.extend(short_flux)
#
#             print(f"Successfully processed {min_len} records from {os.path.basename(file)}")
#
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # Create DataFrame
#     goes_df = pd.DataFrame({
#         'timestamp': all_timestamps,
#         'xray_long': all_xray_long,
#         'xray_short': all_xray_short
#     })
#
#     # Basic data cleaning and flare identification
#     goes_df = goes_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
#     goes_df['xray_long'] = pd.to_numeric(goes_df['xray_long'], errors='coerce')
#     goes_df = goes_df.dropna(subset=['xray_long'])
#
#     # Flare identification logic remains the same as in previous version
#     flares = []
#     class_thresholds = {
#         'X': 1e-4,
#         'M': 1e-5,
#         'C': 1e-6,
#         'B': 1e-7
#     }
#
#     i = 0
#     while i < len(goes_df) - 1:
#         if (goes_df.iloc[i]['xray_long'] > class_thresholds['B'] and
#                 goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#
#             start_idx = i
#             while (i < len(goes_df) - 1 and
#                    goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
#                 i += 1
#             peak_idx = i
#
#             while (i < len(goes_df) - 1 and
#                    goes_df.iloc[i]['xray_long'] > goes_df.iloc[peak_idx]['xray_long'] * 0.5):
#                 i += 1
#             end_idx = i
#
#             peak_flux = goes_df.iloc[peak_idx]['xray_long']
#             flare_class = None
#             for cls, threshold in class_thresholds.items():
#                 if peak_flux >= threshold:
#                     flare_class = cls
#                     magnitude = peak_flux / threshold
#                     break
#
#             if flare_class:
#                 flares.append({
#                     'start_time': goes_df.iloc[start_idx]['timestamp'],
#                     'peak_time': goes_df.iloc[peak_idx]['timestamp'],
#                     'end_time': goes_df.iloc[end_idx]['timestamp'],
#                     'peak_flux': peak_flux,
#                     'class': f"{flare_class}{magnitude:.1f}",
#                     'magnitude': magnitude
#                 })
#         else:
#             i += 1
#
#     flares_df = pd.DataFrame(flares)
#
#     # Save to CSV
#     goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
#     flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)
#
#     print(f"Processed {len(goes_df)} GOES records")
#     print(f"Identified {len(flares_df)} flare events")
#
#     return goes_df, flares_df

def preprocess_goes_data(input_dir=f'{DATA_ROOT}/goes', output_dir=f'{PROCESSED_ROOT}/goes_xray'):
    print("Preprocessing GOES X-ray flux data")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return pd.DataFrame(), pd.DataFrame()

    # Use glob to find all .nc files
    files = glob.glob(os.path.join(input_dir, '*.nc'))

    if not files:
        print(f"No NetCDF files found in {input_dir}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(files)} GOES data files")

    # Lists to store processed data
    all_timestamps = []
    all_xray_long = []
    all_xray_short = []

    for file in files:
        try:
            # Open the dataset using netCDF4 directly
            import netCDF4
            nc = netCDF4.Dataset(file, 'r')

            print(f"\nProcessing file: {os.path.basename(file)}")
            print("Available variables:")
            for var_name in nc.variables:
                print(f"  - {var_name}")

            # Try to find time variable
            time_candidates = ['time', 'time_tag', 'date', 'datetime']
            time_var = None
            for candidate in time_candidates:
                if candidate in nc.variables:
                    time_var = candidate
                    break

            if time_var is None:
                print(f"Warning: No time variable found in {file}")
                continue

            # Convert time to datetime
            times_raw = nc.variables[time_var][:]

            # Determine the base time for conversion
            if hasattr(nc.variables[time_var], 'units'):
                time_units = nc.variables[time_var].units
                if 'since' in time_units:
                    base_time_str = time_units.split('since')[1].strip()
                    try:
                        base_time = pd.to_datetime(base_time_str)
                    except:
                        # If parsing fails, use a default base time
                        base_time = pd.Timestamp('1970-01-01')

                    # Convert to naive timestamps to avoid timezone issues
                    times = [
                        (base_time + pd.Timedelta(seconds=float(t))).replace(tzinfo=None)
                        for t in times_raw
                    ]
                else:
                    times = [pd.to_datetime(t).replace(tzinfo=None) for t in times_raw]
            else:
                times = [pd.to_datetime(t).replace(tzinfo=None) for t in times_raw]

            # Find flux variables
            flux_candidates = {
                'long': ['A_FLUX', 'a_flux', 'xrsa_flux', 'xray_long', 'flux_a'],
                'short': ['B_FLUX', 'b_flux', 'xrsb_flux', 'xray_short', 'flux_b']
            }

            # Try to find long and short wavelength flux variables
            long_flux_var = next((var for var in flux_candidates['long'] if var in nc.variables), None)
            short_flux_var = next((var for var in flux_candidates['short'] if var in nc.variables), None)

            if long_flux_var is None or short_flux_var is None:
                print(f"Warning: Could not find flux variables in {file}")
                continue

            # Extract flux data
            long_flux = nc.variables[long_flux_var][:]
            short_flux = nc.variables[short_flux_var][:]

            # Ensure consistent length
            min_len = min(len(times), len(long_flux), len(short_flux))
            times = times[:min_len]
            long_flux = long_flux[:min_len]
            short_flux = short_flux[:min_len]

            # Extend lists
            all_timestamps.extend(times)
            all_xray_long.extend(long_flux)
            all_xray_short.extend(short_flux)

            print(f"Successfully processed {min_len} records from {os.path.basename(file)}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()

    # Create DataFrame with explicit timezone-naive timestamps
    goes_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'xray_long': all_xray_long,
        'xray_short': all_xray_short
    })

    # Basic data cleaning and flare identification
    goes_df = goes_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
    goes_df['xray_long'] = pd.to_numeric(goes_df['xray_long'], errors='coerce')
    goes_df = goes_df.dropna(subset=['xray_long'])

    # Flare identification logic remains the same
    flares = []
    class_thresholds = {
        'X': 1e-4,
        'M': 1e-5,
        'C': 1e-6,
        'B': 1e-7
    }

    i = 0
    while i < len(goes_df) - 1:
        if (goes_df.iloc[i]['xray_long'] > class_thresholds['B'] and
                goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):

            start_idx = i
            while (i < len(goes_df) - 1 and
                   goes_df.iloc[i + 1]['xray_long'] > goes_df.iloc[i]['xray_long']):
                i += 1
            peak_idx = i

            while (i < len(goes_df) - 1 and
                   goes_df.iloc[i]['xray_long'] > goes_df.iloc[peak_idx]['xray_long'] * 0.5):
                i += 1
            end_idx = i

            peak_flux = goes_df.iloc[peak_idx]['xray_long']
            flare_class = None
            for cls, threshold in class_thresholds.items():
                if peak_flux >= threshold:
                    flare_class = cls
                    magnitude = peak_flux / threshold
                    break

            if flare_class:
                flares.append({
                    'start_time': goes_df.iloc[start_idx]['timestamp'],
                    'peak_time': goes_df.iloc[peak_idx]['timestamp'],
                    'end_time': goes_df.iloc[end_idx]['timestamp'],
                    'peak_flux': peak_flux,
                    'class': f"{flare_class}{magnitude:.1f}",
                    'magnitude': magnitude
                })
        else:
            i += 1

    flares_df = pd.DataFrame(flares)

    # Save to CSV
    goes_df.to_csv(os.path.join(output_dir, 'goes_xray_flux.csv'), index=False)
    flares_df.to_csv(os.path.join(output_dir, 'goes_flare_events.csv'), index=False)

    print(f"Processed {len(goes_df)} GOES records")
    print(f"Identified {len(flares_df)} flare events")

    return goes_df, flares_df

# def preprocess_soho_data(input_dir=f'{DATA_ROOT}/soho', output_dir=f'{PROCESSED_ROOT}/soho_data'):
#     print("Preprocessing SOHO instrument data")
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength']
#                      ).to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return pd.DataFrame()
#
#     # Look for LASCO, EIT, and MDI data
#     instrument_dirs = {
#         'lasco': os.path.join(input_dir, 'lasco'),
#         'eit': os.path.join(input_dir, 'eit'),
#         'mdi': os.path.join(input_dir, 'mdi')
#     }
#
#     processed_data = []
#
#     # Process LASCO coronagraph data
#     if os.path.exists(instrument_dirs['lasco']):
#         print("Processing LASCO coronagraph data")
#         lasco_files = []
#
#         # Check for C2 and C3 coronagraph subdirectories
#         for coronagraph in ['c2', 'c3']:
#             coro_dir = os.path.join(instrument_dirs['lasco'], coronagraph)
#             if os.path.exists(coro_dir):
#                 lasco_files.extend([
#                     (os.path.join(coro_dir, f), coronagraph)
#                     for f in os.listdir(coro_dir) if f.endswith('.fits')
#                 ])
#
#         print(f"Found {len(lasco_files)} LASCO files")
#
#         for i, (file, coronagraph) in enumerate(lasco_files):
#             if i % 10 == 0:
#                 print(f"Processing LASCO {coronagraph} file {i + 1}/{len(lasco_files)}")
#
#             try:
#                 lasco_map = sunpy.map.Map(file)
#                 lasco_data = lasco_map.data
#
#                 if 'date-obs' in lasco_map.meta:
#                     timestamp = Time(lasco_map.meta['date-obs']).datetime
#                 else:
#                     # Try to extract timestamp from filename
#                     filename = os.path.basename(file)
#                     date_str = filename.split('_')[1]
#                     timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S')
#
#                 output_file = os.path.join(output_dir, f'lasco_{coronagraph}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy')
#                 np.save(output_file, lasco_data)
#
#                 processed_data.append({
#                     'filename': output_file,
#                     'instrument': f'lasco_{coronagraph}',
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                     'wavelength': None
#                 })
#
#             except Exception as e:
#                 print(f"Error processing LASCO file {file}: {e}")
#
#     # Process EIT data
#     if os.path.exists(instrument_dirs['eit']):
#         print("Processing EIT data")
#         eit_files = [os.path.join(instrument_dirs['eit'], f)
#                      for f in os.listdir(instrument_dirs['eit']) if f.endswith('.fits')]
#
#         print(f"Found {len(eit_files)} EIT files")
#
#         for i, file in enumerate(eit_files):
#             if i % 10 == 0:
#                 print(f"Processing EIT file {i + 1}/{len(eit_files)}")
#
#             try:
#                 eit_map = sunpy.map.Map(file)
#                 eit_data = eit_map.data
#
#                 if 'date-obs' in eit_map.meta:
#                     timestamp = Time(eit_map.meta['date-obs']).datetime
#                 else:
#                     # Try to extract timestamp from filename
#                     filename = os.path.basename(file)
#                     date_str = filename.split('_')[1]
#                     timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S')
#
#                 # Try to get the wavelength from metadata
#                 wavelength = None
#                 if 'wavelnth' in eit_map.meta:
#                     wavelength = eit_map.meta['wavelnth']
#                 elif 'wavelength' in eit_map.meta:
#                     wavelength = eit_map.meta['wavelength']
#                 else:
#                     # Try to extract from filename (EIT typically observes at 171, 195, 284, and 304 Å)
#                     for wl in ['171', '195', '284', '304']:
#                         if wl in os.path.basename(file):
#                             wavelength = int(wl)
#                             break
#
#                 output_file = os.path.join(output_dir, f'eit_{wavelength}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy')
#                 np.save(output_file, eit_data)
#
#                 processed_data.append({
#                     'filename': output_file,
#                     'instrument': 'eit',
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                     'wavelength': wavelength
#                 })
#
#             except Exception as e:
#                 print(f"Error processing EIT file {file}: {e}")
#
#     # Process MDI data
#     if os.path.exists(instrument_dirs['mdi']):
#         print("Processing MDI data")
#         mdi_files = [os.path.join(instrument_dirs['mdi'], f)
#                      for f in os.listdir(instrument_dirs['mdi']) if f.endswith('.fits')]
#
#         print(f"Found {len(mdi_files)} MDI files")
#
#         for i, file in enumerate(mdi_files):
#             if i % 10 == 0:
#                 print(f"Processing MDI file {i + 1}/{len(mdi_files)}")
#
#             try:
#                 mdi_map = sunpy.map.Map(file)
#                 mdi_data = mdi_map.data
#
#                 if 'date-obs' in mdi_map.meta:
#                     timestamp = Time(mdi_map.meta['date-obs']).datetime
#                 else:
#                     # Try to extract timestamp from filename
#                     filename = os.path.basename(file)
#                     date_str = filename.split('_')[1]
#                     timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S')
#
#                 # Determine if it's a magnetogram or continuum image
#                 mdi_type = 'magnetogram'
#                 if 'content' in mdi_map.meta and 'continuum' in mdi_map.meta['content'].lower():
#                     mdi_type = 'continuum'
#                 elif 'cont' in os.path.basename(file).lower():
#                     mdi_type = 'continuum'
#
#                 output_file = os.path.join(output_dir, f'mdi_{mdi_type}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy')
#                 np.save(output_file, mdi_data)
#
#                 processed_data.append({
#                     'filename': output_file,
#                     'instrument': f'mdi_{mdi_type}',
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                     'wavelength': None
#                 })
#
#             except Exception as e:
#                 print(f"Error processing MDI file {file}: {e}")
#
#     metadata_df = pd.DataFrame(processed_data)
#     if not metadata_df.empty:
#         metadata_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         print(f"Processed {len(processed_data)} SOHO files")
#     else:
#         print("Warning: No SOHO files were successfully processed")
#         pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength']
#                      ).to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#
#     return metadata_df

# def preprocess_soho_data(input_dir=f'{DATA_ROOT}/soho', output_dir=f'{PROCESSED_ROOT}/soho_data'):
#     print("Preprocessing SOHO instrument data")
#
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         # Create empty DataFrame
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df
#
#     # Verbose logging of directory structure
#     print("Checking SOHO data directories:")
#     for item in os.listdir(input_dir):
#         full_path = os.path.join(input_dir, item)
#         if os.path.isdir(full_path):
#             print(f"  - Directory: {item}")
#             print(f"    Contains {len(os.listdir(full_path))} files")
#
#     # Predefined instrument directories
#     instrument_dirs = {
#         'lasco': os.path.join(input_dir, 'lasco'),
#         'eit': os.path.join(input_dir, 'eit'),
#         'mdi': os.path.join(input_dir, 'mdi')
#     }
#
#     processed_data = []
#
#     # Helper function to safely process files
#     def process_instrument_files(directory, instrument_name):
#         if not os.path.exists(directory):
#             print(f"Directory not found: {directory}")
#             return []
#
#         files = [
#             os.path.join(directory, f) for f in os.listdir(directory)
#             if f.endswith('.fits')
#         ]
#
#         print(f"Found {len(files)} {instrument_name.upper()} files")
#
#         instrument_processed = []
#         for file in files:
#             try:
#                 # Verbose file processing
#                 print(f"Processing {instrument_name.upper()} file: {os.path.basename(file)}")
#
#                 # Try to read the FITS file
#                 try:
#                     soho_map = sunpy.map.Map(file)
#                 except Exception as read_err:
#                     print(f"  Error reading FITS file: {read_err}")
#                     continue
#
#                 # Extract timestamp
#                 if 'date-obs' in soho_map.meta:
#                     timestamp = Time(soho_map.meta['date-obs']).datetime
#                 else:
#                     # Fallback timestamp extraction from filename
#                     try:
#                         filename = os.path.basename(file)
#                         date_str = filename.split('_')[1]
#                         timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S')
#                     except Exception as ts_err:
#                         print(f"  Error extracting timestamp: {ts_err}")
#                         continue
#
#                 # Extract wavelength (if applicable)
#                 wavelength = None
#                 if instrument_name == 'eit':
#                     wavelength_candidates = [
#                         soho_map.meta.get('wavelnth'),
#                         soho_map.meta.get('wavelength')
#                     ]
#                     wavelength = next((w for w in wavelength_candidates if w is not None), None)
#
#                 # Save the processed file
#                 output_filename = os.path.join(
#                     output_dir,
#                     f'{instrument_name}_{wavelength or "unknown"}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy'
#                 )
#
#                 # Save the data
#                 np.save(output_filename, soho_map.data)
#
#                 # Collect metadata
#                 instrument_processed.append({
#                     'filename': output_filename,
#                     'instrument': instrument_name,
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                     'wavelength': wavelength
#                 })
#
#                 print(f"  Successfully processed file")
#
#             except Exception as e:
#                 print(f"Error processing {instrument_name.upper()} file {file}: {e}")
#                 import traceback
#                 traceback.print_exc()
#
#         return instrument_processed
#
#     # Process each instrument
#     for instrument, directory in instrument_dirs.items():
#         processed_instrument_data = process_instrument_files(directory, instrument)
#         processed_data.extend(processed_instrument_data)
#
#     # Create DataFrame
#     if processed_data:
#         soho_df = pd.DataFrame(processed_data)
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         print(f"Processed {len(processed_data)} SOHO files")
#         return soho_df
#     else:
#         print("No SOHO files were successfully processed")
#         # Create empty DataFrame
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df

# def preprocess_soho_data(input_dir=f'{DATA_ROOT}/soho', output_dir=f'{PROCESSED_ROOT}/soho_data'):
#     print("Preprocessing SOHO instrument data")
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df
#
#     # List all subdirectories in the SOHO data directory
#     instrument_dirs = [
#         os.path.join(input_dir, d) for d in os.listdir(input_dir)
#         if os.path.isdir(os.path.join(input_dir, d))
#     ]
#
#     if not instrument_dirs:
#         print("No instrument directories found in SOHO data directory")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df
#
#     processed_data = []
#
#     for instrument_dir in instrument_dirs:
#         instrument_name = os.path.basename(instrument_dir)
#
#         # Find all FITS files in the directory
#         fits_files = glob.glob(os.path.join(instrument_dir, '*.fits'))
#
#         if not fits_files:
#             print(f"No FITS files found in {instrument_dir}")
#             continue
#
#         print(f"Processing {len(fits_files)} {instrument_name.upper()} files")
#
#         for file in fits_files:
#             try:
#                 # Use sunpy to read the FITS file
#                 soho_map = sunpy.map.Map(file)
#
#                 # Extract timestamp
#                 if 'date-obs' in soho_map.meta:
#                     timestamp = Time(soho_map.meta['date-obs']).datetime
#                 else:
#                     # Fallback timestamp extraction from filename
#                     try:
#                         filename = os.path.basename(file)
#                         # Assumes filename contains date information
#                         date_str = ''.join(filter(str.isdigit, filename))[:14]
#                         timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S')
#                     except Exception as ts_err:
#                         print(f"  Error extracting timestamp: {ts_err}")
#                         continue
#
#                 # Extract wavelength (if applicable)
#                 wavelength = None
#                 if instrument_name == 'eit':
#                     wavelength_candidates = [
#                         soho_map.meta.get('wavelnth'),
#                         soho_map.meta.get('wavelength')
#                     ]
#                     wavelength = next((w for w in wavelength_candidates if w is not None), None)
#
#                 # Save the map data
#                 output_filename = os.path.join(
#                     output_dir,
#                     f'{instrument_name}_{wavelength or "unknown"}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy'
#                 )
#                 np.save(output_filename, soho_map.data)
#
#                 # Collect metadata
#                 processed_data.append({
#                     'filename': output_filename,
#                     'instrument': instrument_name,
#                     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                     'wavelength': wavelength
#                 })
#
#                 print(f"  Successfully processed {os.path.basename(file)}")
#
#             except Exception as e:
#                 print(f"Error processing SOHO file {file}: {e}")
#                 import traceback
#                 traceback.print_exc()
#
#     # Create and save metadata DataFrame
#     if processed_data:
#         soho_df = pd.DataFrame(processed_data)
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         print(f"Processed {len(processed_data)} SOHO files")
#     else:
#         print("No SOHO files were successfully processed")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#
#     return soho_df


# Create simple feature extractor to generate basic magnetogram features
# def extract_magnetogram_features(magnetogram_metadata, output_dir=f'{PROCESSED_ROOT}/features'):
#     """Extract basic features from magnetograms"""
#     print(f"Extracting features from {len(magnetogram_metadata)} magnetograms")
#
#     if magnetogram_metadata.empty:
#         print("No magnetograms available for feature extraction")
#         # Create empty feature file to avoid downstream errors
#         pd.DataFrame(columns=['timestamp', 'total_flux', 'max_field_strength', 'has_flare']
#                      ).to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         return pd.DataFrame()
#
#     features = []
#
#     for _, row in magnetogram_metadata.iterrows():
#         try:
#             # Load magnetogram data
#             magnetogram = np.load(row['filename'])
#
#             # Extract timestamp
#             timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
#
#             # Calculate basic features
#             total_flux = np.sum(np.abs(magnetogram))
#             max_field_strength = np.max(np.abs(magnetogram))
#
#             features.append({
#                 'timestamp': timestamp,
#                 'total_flux': total_flux,
#                 'max_field_strength': max_field_strength,
#                 # Add more features as needed
#             })
#
#         except Exception as e:
#             print(f"Error extracting features from {row['filename']}: {e}")
#
#     # Create features DataFrame
#     features_df = pd.DataFrame(features)
#
#     if not features_df.empty:
#         # Save features
#         features_df.to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         print(f"Extracted features from {len(features)} magnetograms")
#     else:
#         # Create empty file with columns
#         pd.DataFrame(columns=['timestamp', 'total_flux', 'max_field_strength', 'has_flare']
#                      ).to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         print("No features were extracted")
#
#     return features_df

# def extract_magnetogram_features(magnetogram_metadata, output_dir=f'{PROCESSED_ROOT}/features', flares_df=None):
#     print(f"Extracting features from {len(magnetogram_metadata)} magnetograms")
#
#     if magnetogram_metadata.empty:
#         print("No magnetograms available for feature extraction")
#         pd.DataFrame(columns=[
#             'timestamp', 'total_flux', 'max_field_strength', 'neutral_line_length',
#             'r_value', 'wlsg', 'helicity_injection', 'free_energy_proxy', 'has_flare',
#             'next_flare_magnitude', 'time_to_next_flare'
#         ]).to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         return pd.DataFrame()
#
#     # Convert flares dataframe timestamps for later reference
#     if flares_df is not None and not flares_df.empty:
#         flares_df['start_time'] = pd.to_datetime(flares_df['start_time'])
#         flares_df['peak_time'] = pd.to_datetime(flares_df['peak_time'])
#         flares_df['end_time'] = pd.to_datetime(flares_df['end_time'])
#
#         # Extract magnitude values from flare class
#         flares_df['magnitude'] = flares_df['class'].apply(
#             lambda x: float(x[1:]) * (10 ** (ord(x[0]) - ord('A')))
#         )
#
#     features = []
#
#     # Get magnetogram files ordered by timestamp for time series analysis
#     magnetogram_metadata['timestamp'] = pd.to_datetime(magnetogram_metadata['timestamp'])
#     magnetogram_metadata = magnetogram_metadata.sort_values('timestamp')
#
#     # For temporal evolution features
#     time_window = 12  # hours
#     prev_magnetograms = {}  # Store previous magnetograms for temporal analysis
#
#     for i, row in magnetogram_metadata.iterrows():
#         try:
#             magnetogram = np.load(row['filename'])
#             timestamp = row['timestamp']
#
#             # Basic features
#             total_flux = np.sum(np.abs(magnetogram))
#             max_field_strength = np.max(np.abs(magnetogram))
#
#             # Calculate neutral line
#             # Smooth the magnetogram to reduce noise
#             smoothed_mag = gaussian_filter(magnetogram, sigma=2)
#
#             # Find the polarity inversion line (where field changes sign)
#             bz_pos = smoothed_mag > 0
#             bz_neg = smoothed_mag < 0
#
#             # Create binary masks for positive and negative polarities
#             pos_mask = bz_pos.astype(int)
#             neg_mask = bz_neg.astype(int)
#
#             # Dilate masks to find regions where they would overlap
#             from scipy.ndimage import binary_dilation
#             struct = np.ones((3, 3))
#             dilated_pos = binary_dilation(pos_mask, structure=struct)
#             dilated_neg = binary_dilation(neg_mask, structure=struct)
#
#             # The neutral line is where dilated masks overlap
#             neutral_line = dilated_pos & dilated_neg
#
#             # Calculate the length of the neutral line
#             neutral_line_length = np.sum(neutral_line)
#
#             # Calculate the gradient across the image
#             gradient_x = np.gradient(smoothed_mag, axis=1)
#             gradient_y = np.gradient(smoothed_mag, axis=0)
#
#             # Compute gradient magnitude
#             gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#
#             # R-value: gradient-weighted neutral line length
#             r_value = np.sum(gradient_magnitude * neutral_line)
#
#             # Weighted Integral over Neutral Line of the Squared Gradient (WLSG)
#             strongest_gradients = np.percentile(gradient_magnitude[neutral_line], 90) if np.any(neutral_line) else 0
#             wlsg = neutral_line_length * strongest_gradients
#
#             # Calculate proxy for magnetic helicity injection rate (simplified)
#             # In a real implementation, we'd use vector magnetogram data
#             # This is a simple proxy based on the curl of the field
#             curl_proxy = np.gradient(gradient_y, axis=1) - np.gradient(gradient_x, axis=0)
#             helicity_injection = np.sum(np.abs(curl_proxy))
#
#             # Proxy for free magnetic energy
#             # Using squared gradients as proxy for excess energy
#             free_energy_proxy = np.sum(gradient_magnitude ** 2)
#
#             # Calculate temporal evolution parameters if previous magnetograms exist
#             temporal_features = {}
#
#             if len(prev_magnetograms) > 0:
#                 # Find magnetograms within the time window
#                 window_start = timestamp - pd.Timedelta(hours=time_window)
#
#                 # Get the most recent previous magnetogram
#                 prev_times = [t for t in prev_magnetograms.keys() if window_start <= t < timestamp]
#
#                 if prev_times:
#                     most_recent = max(prev_times)
#                     time_diff = (timestamp - most_recent).total_seconds() / 3600  # hours
#                     prev_mag = prev_magnetograms[most_recent]
#
#                     # Calculate flux emergence rate
#                     flux_change = total_flux - np.sum(np.abs(prev_mag))
#                     flux_emergence_rate = flux_change / time_diff if time_diff > 0 else 0
#
#                     # Calculate abs change in free energy proxy
#                     prev_gradient = np.gradient(prev_mag)
#                     prev_gradient_magnitude = np.sqrt(prev_gradient[0] ** 2 + prev_gradient[1] ** 2)
#                     prev_energy = np.sum(prev_gradient_magnitude ** 2)
#                     energy_change_rate = (free_energy_proxy - prev_energy) / time_diff if time_diff > 0 else 0
#
#                     temporal_features = {
#                         'flux_emergence_rate': flux_emergence_rate,
#                         'energy_change_rate': energy_change_rate
#                     }
#
#             # Store current magnetogram for future comparisons
#             # Keep only the last 24 hours of magnetograms to limit memory usage
#             prev_magnetograms[timestamp] = magnetogram
#             outdated_times = [t for t in prev_magnetograms.keys() if t < (timestamp - pd.Timedelta(hours=24))]
#             for t in outdated_times:
#                 del prev_magnetograms[t]
#
#             # Determine if a flare occurs in the next 24 hours
#             has_flare = False
#             next_flare_magnitude = 0
#             time_to_next_flare = float('inf')
#
#             if flares_df is not None and not flares_df.empty:
#                 # Find flares that start within 24 hours after this magnetogram
#                 next_day = timestamp + pd.Timedelta(days=1)
#                 upcoming_flares = flares_df[(flares_df['start_time'] > timestamp) &
#                                             (flares_df['start_time'] <= next_day)]
#
#                 if not upcoming_flares.empty:
#                     has_flare = True
#                     next_flare = upcoming_flares.iloc[0]
#                     next_flare_magnitude = next_flare['magnitude']
#                     time_to_next_flare = (next_flare['start_time'] - timestamp).total_seconds() / 3600  # hours
#
#             # Combine all features
#             feature_dict = {
#                 'timestamp': timestamp,
#                 'total_flux': total_flux,
#                 'max_field_strength': max_field_strength,
#                 'neutral_line_length': neutral_line_length,
#                 'r_value': r_value,
#                 'wlsg': wlsg,
#                 'helicity_injection': helicity_injection,
#                 'free_energy_proxy': free_energy_proxy,
#                 'has_flare': has_flare,
#                 'next_flare_magnitude': next_flare_magnitude,
#                 'time_to_next_flare': time_to_next_flare
#             }
#
#             # Add temporal features if available
#             feature_dict.update(temporal_features)
#
#             features.append(feature_dict)
#
#         except Exception as e:
#             print(f"Error extracting features from {row['filename']}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     features_df = pd.DataFrame(features)
#
#     if not features_df.empty:
#         features_df.to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         print(f"Extracted features from {len(features)} magnetograms")
#     else:
#         pd.DataFrame(columns=[
#             'timestamp', 'total_flux', 'max_field_strength', 'neutral_line_length',
#             'r_value', 'wlsg', 'helicity_injection', 'free_energy_proxy', 'has_flare',
#             'next_flare_magnitude', 'time_to_next_flare'
#         ]).to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
#         print("No features were extracted")
#
#     return

# def preprocess_soho_data(input_dir=f'{DATA_ROOT}/soho', output_dir=f'{PROCESSED_ROOT}/soho_data'):
#     print("Preprocessing SOHO instrument data")
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(input_dir):
#         print(f"Error: Input directory {input_dir} does not exist")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df
#
#     # Expand search to include nested directories and handle more file extensions
#     fits_files = []
#     for root, _, files in os.walk(input_dir):
#         fits_files.extend([
#             os.path.join(root, f) for f in files
#             if f.lower().endswith(('.fits', '.fts', '.fit'))
#         ])
#
#     if not fits_files:
#         print("No FITS files found in SOHO data directory")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         return soho_df
#
#     processed_data = []
#
#     for file in fits_files:
#         try:
#             # More robust error handling and file reading
#             try:
#                 soho_map = sunpy.map.Map(file)
#             except Exception as map_error:
#                 print(f"Could not create map for {file}: {map_error}")
#                 continue
#
#             # Extract instrument name from file path or metadata
#             instrument_name = os.path.basename(os.path.dirname(file)).lower()
#             if not instrument_name or instrument_name == 'soho':
#                 # Try to extract from filename or metadata
#                 instrument_name = soho_map.meta.get('instrume', 'unknown').lower()
#
#             # Timestamp extraction with multiple fallback methods
#             try:
#                 if 'date-obs' in soho_map.meta:
#                     timestamp = Time(soho_map.meta['date-obs']).datetime
#                 elif 'time-obs' in soho_map.meta:
#                     timestamp = Time(f"{soho_map.meta.get('date-obs', '')}{soho_map.meta['time-obs']}").datetime
#                 else:
#                     # Fallback to filename parsing
#                     filename = os.path.basename(file)
#                     date_str = ''.join(filter(str.isdigit, filename))[:14]
#                     timestamp = datetime.strptime(date_str, '%Y%m%d%H%M%S') if date_str else datetime.now()
#             except Exception as time_error:
#                 print(f"Could not parse timestamp for {file}: {time_error}")
#                 timestamp = datetime.now()
#
#             # Wavelength extraction with multiple fallback methods
#             wavelength = None
#             wavelength_candidates = [
#                 soho_map.meta.get('wavelnth'),
#                 soho_map.meta.get('wavelength'),
#                 soho_map.meta.get('wavelen'),
#                 soho_map.meta.get('wave_len')
#             ]
#             wavelength = next((w for w in wavelength_candidates if w is not None), None)
#
#             # Validate and process image data
#             if soho_map.data is None or np.all(np.isnan(soho_map.data)) or np.all(soho_map.data == 0):
#                 print(f"Skipping {file}: Invalid image data")
#                 continue
#
#             # Ensure data is numeric and convert to float32
#             data = soho_map.data.astype(np.float32)
#
#             # Generate output filename
#             output_filename = os.path.join(
#                 output_dir,
#                 f'{instrument_name}_{wavelength or "unknown"}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy'
#             )
#
#             # Save processed data
#             np.save(output_filename, data)
#
#             processed_data.append({
#                 'filename': output_filename,
#                 'instrument': instrument_name,
#                 'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                 'wavelength': wavelength
#             })
#
#             print(f"Successfully processed {os.path.basename(file)}")
#
#         except Exception as e:
#             print(f"Unexpected error processing SOHO file {file}: {e}")
#             import traceback
#             traceback.print_exc()
#
#     # Create and save metadata
#     if processed_data:
#         soho_df = pd.DataFrame(processed_data)
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#         print(f"Processed {len(processed_data)} SOHO files")
#     else:
#         print("No SOHO files were successfully processed")
#         soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
#         soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
#
#     return soho_df

def preprocess_soho_data(input_dir=f'{DATA_ROOT}/soho', output_dir=f'{PROCESSED_ROOT}/soho_data'):
    print("Preprocessing SOHO instrument data")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
        soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
        return soho_df

    # Expand search to include nested directories and handle .fts files
    fits_files = []
    for root, _, files in os.walk(input_dir):
        fits_files.extend([
            os.path.join(root, f) for f in files
            if f.lower().endswith(('.fts', '.fits', '.fit'))
        ])

    if not fits_files:
        print("No FITS or FTS files found in SOHO data directory")
        soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
        soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
        return soho_df

    processed_data = []

    # LASCO specific wavelength mapping
    lasco_wavelengths = {
        'c2': 'white light (visible)',
        'c3': 'white light (visible)'
    }

    for file in fits_files:
        try:
            # More robust error handling and file reading
            try:
                soho_map = sunpy.map.Map(file)
            except Exception as map_error:
                print(f"Could not create map for {file}: {map_error}")
                continue

            # Extract instrument name with specific LASCO handling
            filename = os.path.basename(file)
            filepath = os.path.dirname(file).lower()

            # Determine instrument and detector
            instrument_name = 'lasco'
            detector = 'unknown'

            # Check for C2 or C3 in directory or filename
            for det in ['c2', 'c3']:
                if det in filepath or det in filename.lower():
                    detector = det
                    break

            # Wavelength extraction
            wavelength = lasco_wavelengths.get(detector, 'white light')

            # Timestamp extraction
            try:
                # Try extracting timestamp from FITS header
                if 'date-obs' in soho_map.meta:
                    timestamp = Time(soho_map.meta['date-obs']).datetime
                else:
                    # Use filename (assuming it's a numeric timestamp)
                    try:
                        # Remove file extension
                        timestamp_str = os.path.splitext(filename)[0]

                        # Assume the filename is a numeric representation of time
                        # Adjust the parsing based on the actual format of your filenames
                        timestamp = datetime(
                            year=2017,  # Assuming the data is from 2017 based on previous context
                            month=9,  # Assuming September based on previous context
                            day=int(timestamp_str[:2]),  # First two digits as day
                            hour=int(timestamp_str[2:4]) if len(timestamp_str) >= 4 else 0,
                            minute=int(timestamp_str[4:6]) if len(timestamp_str) >= 6 else 0,
                            second=int(timestamp_str[6:8]) if len(timestamp_str) >= 8 else 0
                        )
                    except Exception as parse_error:
                        print(f"Could not parse timestamp from filename {filename}: {parse_error}")
                        timestamp = datetime.now()

            except Exception as time_error:
                print(f"Could not parse timestamp for {file}: {time_error}")
                timestamp = datetime.now()

            # Validate and process image data
            if soho_map.data is None or np.all(np.isnan(soho_map.data)) or np.all(soho_map.data == 0):
                print(f"Skipping {file}: Invalid image data")
                continue

            # Ensure data is numeric and convert to float32
            data = soho_map.data.astype(np.float32)

            # Generate output filename
            output_filename = os.path.join(
                output_dir,
                f'{instrument_name}_{detector}_{timestamp.strftime("%Y%m%d_%H%M%S")}.npy'
            )

            # Save processed data
            np.save(output_filename, data)

            processed_data.append({
                'filename': output_filename,
                'instrument': instrument_name,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'wavelength': wavelength
            })

            print(f"Successfully processed {filename}")

        except Exception as e:
            print(f"Unexpected error processing SOHO file {file}: {e}")
            import traceback
            traceback.print_exc()

    # Create and save metadata
    if processed_data:
        soho_df = pd.DataFrame(processed_data)
        soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)
        print(f"Processed {len(processed_data)} SOHO files")
    else:
        print("No SOHO files were successfully processed")
        soho_df = pd.DataFrame(columns=['filename', 'instrument', 'timestamp', 'wavelength'])
        soho_df.to_csv(os.path.join(output_dir, 'soho_metadata.csv'), index=False)

    return soho_df


def extract_magnetogram_features(magnetogram_metadata, output_dir=f'{PROCESSED_ROOT}/features', flares_df=None):
    print(f"Extracting features from {len(magnetogram_metadata)} magnetograms")

    if magnetogram_metadata.empty:
        print("No magnetograms available for feature extraction")
        pd.DataFrame(columns=[
            'timestamp', 'total_flux', 'max_field_strength', 'neutral_line_length',
            'r_value', 'wlsg', 'helicity_injection', 'free_energy_proxy', 'has_flare',
            'next_flare_magnitude', 'time_to_next_flare'
        ]).to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
        return pd.DataFrame()

    # Ensure flares_df is properly processed
    if flares_df is not None and not flares_df.empty:
        flares_df['start_time'] = pd.to_datetime(flares_df['start_time'])
        flares_df['magnitude'] = flares_df['class'].apply(
            lambda x: float(x[1:]) * (10 ** max(0, ord(x[0]) - ord('A')))
        )

    features = []

    magnetogram_metadata['timestamp'] = pd.to_datetime(magnetogram_metadata['timestamp'])
    magnetogram_metadata = magnetogram_metadata.sort_values('timestamp')

    def safe_load_magnetogram(filename):
        try:
            magnetogram = np.load(filename)
            # Handle potential NaN or zero-filled arrays
            if np.all(np.isnan(magnetogram)) or np.all(magnetogram == 0):
                print(f"Warning: Magnetogram {filename} contains only NaN or zero values")
                return None
            return magnetogram
        except Exception as e:
            print(f"Error loading magnetogram {filename}: {e}")
            return None

    time_window = 12
    prev_magnetograms = {}

    for i, row in magnetogram_metadata.iterrows():
        try:
            # Robust magnetogram loading
            magnetogram = safe_load_magnetogram(row['filename'])
            if magnetogram is None:
                continue

            timestamp = row['timestamp']

            # More robust feature calculations with checks
            total_flux = np.nansum(np.abs(magnetogram))
            max_field_strength = np.nanmax(np.abs(magnetogram))

            # Skip if total flux is zero or NaN
            if total_flux == 0 or np.isnan(total_flux):
                print(f"Skipping magnetogram at {timestamp} due to zero total flux")
                continue

            # Enhanced smoothing and feature extraction
            smoothed_mag = gaussian_filter(magnetogram, sigma=2)
            smoothed_mag = np.nan_to_num(smoothed_mag, nan=0)

            # More robust neutral line detection
            bz_pos = smoothed_mag > 0
            bz_neg = smoothed_mag < 0

            pos_mask = bz_pos.astype(int)
            neg_mask = bz_neg.astype(int)

            from scipy.ndimage import binary_dilation
            struct = np.ones((3, 3))
            dilated_pos = binary_dilation(pos_mask, structure=struct)
            dilated_neg = binary_dilation(neg_mask, structure=struct)

            neutral_line = dilated_pos & dilated_neg
            neutral_line_length = np.sum(neutral_line)

            # Gradient calculations with NaN handling
            gradient_x = np.nan_to_num(np.gradient(smoothed_mag, axis=1), nan=0)
            gradient_y = np.nan_to_num(np.gradient(smoothed_mag, axis=0), nan=0)

            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

            # R-value calculation with additional checks
            r_value = np.sum(gradient_magnitude * neutral_line) if np.any(neutral_line) else 0

            # WLSG calculation with percentile check
            strongest_gradients = (np.percentile(gradient_magnitude[neutral_line], 90)
                                   if np.any(neutral_line) else 0)
            wlsg = neutral_line_length * strongest_gradients

            # Helicity and energy calculations
            curl_proxy = (np.nan_to_num(np.gradient(gradient_y, axis=1), nan=0) -
                          np.nan_to_num(np.gradient(gradient_x, axis=0), nan=0))
            helicity_injection = np.sum(np.abs(curl_proxy))
            free_energy_proxy = np.sum(gradient_magnitude ** 2)

            # Detailed logging for debugging
            print(f"Magnetogram at {timestamp}:")
            print(f"  Total Flux: {total_flux}")
            print(f"  Max Field Strength: {max_field_strength}")
            print(f"  Neutral Line Length: {neutral_line_length}")
            print(f"  R-Value: {r_value}")
            print(f"  WLSG: {wlsg}")
            print(f"  Helicity Injection: {helicity_injection}")
            print(f"  Free Energy Proxy: {free_energy_proxy}")

            # Flare prediction features
            has_flare = False
            next_flare_magnitude = 0
            time_to_next_flare = float('inf')

            if flares_df is not None and not flares_df.empty:
                next_day = timestamp + pd.Timedelta(days=1)
                upcoming_flares = flares_df[(flares_df['start_time'] > timestamp) &
                                            (flares_df['start_time'] <= next_day)]

                if not upcoming_flares.empty:
                    has_flare = True
                    next_flare = upcoming_flares.iloc[0]
                    next_flare_magnitude = next_flare['magnitude']
                    time_to_next_flare = (next_flare['start_time'] - timestamp).total_seconds() / 3600

            features.append({
                'timestamp': timestamp,
                'total_flux': total_flux,
                'max_field_strength': max_field_strength,
                'neutral_line_length': neutral_line_length,
                'r_value': r_value,
                'wlsg': wlsg,
                'helicity_injection': helicity_injection,
                'free_energy_proxy': free_energy_proxy,
                'has_flare': has_flare,
                'next_flare_magnitude': next_flare_magnitude,
                'time_to_next_flare': time_to_next_flare
            })

        except Exception as e:
            print(f"Error processing magnetogram {row['filename']}: {e}")
            import traceback
            traceback.print_exc()

    features_df = pd.DataFrame(features)

    if not features_df.empty:
        features_df.to_csv(os.path.join(output_dir, 'magnetogram_features.csv'), index=False)
        print(f"Extracted features from {len(features)} magnetograms")
    else:
        print("No features were extracted. Investigating potential issues.")

    return features_df


# New function for AIA multispectral image analysis with wavelet decomposition
def extract_aia_features(aia_metadata, output_dir=f'{PROCESSED_ROOT}/features'):
    print(f"Extracting features from {len(aia_metadata)} AIA images")

    if aia_metadata.empty:
        print("No AIA images available for feature extraction")
        pd.DataFrame(columns=[
            'timestamp', 'wavelength', 'mean_intensity', 'std_intensity',
            'max_intensity', 'wavelet_power', 'entropy'
        ]).to_csv(os.path.join(output_dir, 'aia_features.csv'), index=False)
        return pd.DataFrame()

    aia_features = []

    # Group by timestamp to analyze multi-wavelength data together
    aia_metadata['timestamp'] = pd.to_datetime(aia_metadata['timestamp'])
    grouped = aia_metadata.groupby('timestamp')

    for timestamp, group in grouped:
        print(f"Processing AIA images for timestamp {timestamp}")

        # Process each wavelength
        for _, row in group.iterrows():
            try:
                wavelength = row['wavelength']
                image_file = row['filename']

                # Load the compressed image data
                with np.load(image_file) as data:
                    image = data['data']

                # Basic statistics
                mean_intensity = np.mean(image)
                std_intensity = np.std(image)
                max_intensity = np.max(image)

                # Wavelet decomposition
                # Use 2D discrete wavelet transform
                coeffs = pywt.dwt2(image, 'haar')
                # Unpack the coefficients
                cA, (cH, cV, cD) = coeffs

                # Calculate wavelet power as sum of squared detail coefficients
                wavelet_power = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)

                # Calculate entropy (a measure of randomness/complexity)
                # Normalize the image
                img_norm = image / np.max(image) if np.max(image) > 0 else image
                # Use bins from 0 to 1 with step 0.1
                hist, _ = np.histogram(img_norm, bins=10, range=(0, 1), density=True)
                # Remove zeros to avoid log(0)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))

                # Store the features
                aia_features.append({
                    'timestamp': timestamp,
                    'wavelength': wavelength,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'max_intensity': max_intensity,
                    'wavelet_power': wavelet_power,
                    'entropy': entropy
                })

            except Exception as e:
                print(f"Error extracting features from AIA image {row['filename']}: {e}")
                import traceback
                traceback.print_exc()

    features_df = pd.DataFrame(aia_features)

    if not features_df.empty:
        features_df.to_csv(os.path.join(output_dir, 'aia_features.csv'), index=False)
        print(f"Extracted features from {len(aia_features)} AIA images")
    else:
        pd.DataFrame(columns=[
            'timestamp', 'wavelength', 'mean_intensity', 'std_intensity',
            'max_intensity', 'wavelet_power', 'entropy'
        ]).to_csv(os.path.join(output_dir, 'aia_features.csv'), index=False)
        print("No AIA features were extracted")

    return features_df

# def main(start_date='2018-01-01', end_date='2018-01-31', sample=False):
#     """Run the complete data preprocessing pipeline"""
#     try:
#         # Create directory structure
#         create_directories()
#
#         # Inspect input directories
#         print("\nInspecting input directories:")
#         inspect_file_structure(f'{DATA_ROOT}/sdo_hmi')
#         inspect_file_structure(f'{DATA_ROOT}/sdo_aia')
#         inspect_file_structure(f'{DATA_ROOT}/goes')
#
#         # Preprocess data
#         print("\nPreprocessing data:")
#         hmi_metadata = preprocess_hmi_data()
#         aia_metadata = preprocess_aia_data()
#         goes_df, flares_df = preprocess_goes_data()
#
#         # Extract features from magnetograms
#         features_df = extract_magnetogram_features(hmi_metadata)
#
#         # Inspect output directories after processing
#         print("\nInspecting output directories:")
#         inspect_file_structure(f'{PROCESSED_ROOT}/magnetograms')
#         inspect_file_structure(f'{PROCESSED_ROOT}/aia_images')
#         inspect_file_structure(f'{PROCESSED_ROOT}/goes_xray')
#         inspect_file_structure(f'{PROCESSED_ROOT}/features')
#
#         print("Data preprocessing completed successfully!")
#
#         return {
#             'hmi_metadata': hmi_metadata,
#             'aia_metadata': aia_metadata,
#             'goes_df': goes_df,
#             'flares_df': flares_df,
#             'features_df': features_df
#         }
#
#     except Exception as e:
#         print(f"Error in data preprocessing pipeline: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

def main(start_date='2017-09-01', end_date='2017-09-21', sample=False):
    try:
        create_directories()

        print("\nInspecting input directories:")
        # inspect_file_structure(f'{DATA_ROOT}/sdo_hmi')
        # inspect_file_structure(f'{DATA_ROOT}/sdo_aia')
        inspect_file_structure(f'{DATA_ROOT}/goes')
        inspect_file_structure(f'{DATA_ROOT}/soho')  # Added SOHO directory inspection

        print("\nPreprocessing data:")
        # hmi_metadata = preprocess_hmi_data()
        # aia_metadata = preprocess_aia_data()
        goes_df, flares_df = preprocess_goes_data()
        soho_metadata = preprocess_soho_data()  # Added SOHO data preprocessing

        print("\nExtracting features:")
        # magnetogram_features = extract_magnetogram_features(hmi_metadata, flares_df=flares_df)
        # aia_features = extract_aia_features(aia_metadata)  # Added AIA feature extraction

        print("\nInspecting output directories:")
        # inspect_file_structure(f'{PROCESSED_ROOT}/magnetograms')
        # inspect_file_structure(f'{PROCESSED_ROOT}/aia_images')
        inspect_file_structure(f'{PROCESSED_ROOT}/goes_xray')
        inspect_file_structure(f'{PROCESSED_ROOT}/soho_data')  # Added SOHO output inspection
        inspect_file_structure(f'{PROCESSED_ROOT}/features')

        print("Data preprocessing completed successfully!")

        return {
            # 'hmi_metadata': hmi_metadata,
            # 'aia_metadata': aia_metadata,
            'goes_df': goes_df,
            'flares_df': flares_df,
            'soho_metadata': soho_metadata,  # Added SOHO metadata to return
            # 'magnetogram_features': magnetogram_features,
            # 'aia_features': aia_features  # Added AIA features to return
        }

    except Exception as e:
        print(f"Error in data preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main(sample=False)