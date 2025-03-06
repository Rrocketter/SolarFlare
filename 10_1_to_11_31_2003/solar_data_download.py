# import os
# import datetime
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sunpy.net import Fido
#
# from sunpy.net import attrs as a
# from sunpy.net import jsoc
# import sunpy.map
# import sunpy.timeseries
# from sunpy.timeseries import TimeSeries
# import astropy.units as u
# from sunpy.net.dataretriever import XRSClient
# from astropy.time import Time
# from astropy.io import fits
# from pathlib import Path
# import warnings
#
# import json
# import requests
# from datetime import datetime, timedelta
#
# warnings.filterwarnings('ignore')
#
#
# # Create directory structure for organized data storage
# def create_directories():
#     base_dir = Path("./solar_data")
#     dirs = {
#         "sdo_hmi": base_dir / "sdo_hmi",
#         "sdo_aia": base_dir / "sdo_aia",
#         "goes": base_dir / "goes",
#         "noaa_ar": base_dir / "noaa_ar",
#         "soho": base_dir / "soho",
#         "processed": base_dir / "processed",
#         "visualizations": base_dir / "visualizations"
#     }
#
#     for dir_path in dirs.values():
#         dir_path.mkdir(parents=True, exist_ok=True)
#
#     return dirs
#
# # Call the function to create directories and assign the result to the dirs variable
#
#
#
# # Function to download SDO/HMI magnetogram data
# # def download_hmi_data(start_date, end_date, dirs, sample_cadence='6h', max_files=10):
# #     """
# #     Download SDO/HMI magnetogram data for a specified time range
# #
# #     Parameters:
# #     -----------
# #     start_date, end_date : str
# #         Date range in format 'YYYY-MM-DD'
# #     sample_cadence : str
# #         Time cadence for sampling data (to reduce volume)
# #     max_files : int
# #         Maximum number of files to download (to control storage)
# #     """
# #     print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")
# #
# #     # Define time range
# #     time_range = a.Time(start_date, end_date)
# #
# #     # Query for vector magnetograms (series: hmi.B_720s)
# #     result = Fido.search(
# #         time_range,
# #         a.Instrument.hmi,
# #         # a.Physobs.vector_magnetic_field,
# #         # a.Sample(sample_cadence)
# #         a.Physobs.los_magnetic_field,
# #         a.Sample(u.Quantity(sample_cadence))
# #     )
# #
# #     # Limit the number of files to download
# #     if len(result) > max_files:
# #         print(f"Found {len(result)} files, limiting to {max_files}")
# #         result = result[:max_files]
# #     else:
# #         print(f"Found {len(result)} files")
# #
# #     # Download the files
# #     downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}")
# #
# #     print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
# #     return downloaded_files
#
# # def download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10):
# #     """
# #     Download SDO/HMI magnetogram data for a specified time range
# #     """
# #     print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")
# #
# #     # Define time range
# #     time_range = a.Time(start_date, end_date)
# #
# #     # Query specifically for the LOS magnetogram product (hmi.M_720s)
# #     result = Fido.search(
# #         time_range,
# #         a.Instrument.hmi,
# #         a.Physobs.los_magnetic_field,
# #         a.Sample(u.Quantity(sample_cadence)),
# #         a.jsoc.Series('hmi.M_720s')  # Use a.jsoc.Series instead of a.Series
# #     )
# #
# #     # Limit the number of files to download
# #     if len(result) > max_files:
# #         print(f"Found {len(result)} files, limiting to {max_files}")
# #         result = result[:max_files]
# #     else:
# #         print(f"Found {len(result)} files")
# #
# #     # Download the files
# #     if len(result) > 0:
# #         downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}")
# #         print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
# #         return downloaded_files
# #     else:
# #         print("No HMI files found to download")
# #         return []
#
#
# def download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10):
#     """
#     Download SDO/HMI magnetogram data for a specified time range
#     """
#     print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     # Query for LOS magnetograms (series: hmi.M_720s)
#     result = Fido.search(
#         time_range,
#         a.Instrument.hmi,
#         a.Physobs.los_magnetic_field,
#         a.Sample(u.Quantity(sample_cadence))
#     )
#
#     # If the first query returns no results, try with the specific series
#     if len(result) == 0:
#         print("No results with default query. Trying with specific data series...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.hmi,
#             a.Physobs.los_magnetic_field,
#             a.Sample(u.Quantity(sample_cadence)),
#             a.Series('hmi.M_720s')
#         )
#
#     # Limit the number of files to download
#     if len(result) > max_files:
#         print(f"Found {len(result)} files, limiting to {max_files}")
#         result = result[:max_files]
#     else:
#         print(f"Found {len(result)} files")
#
#     # Download the files
#     if len(result) > 0:
#         downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}")
#         print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
#         return downloaded_files
#     else:
#         print("No HMI files found to download")
#         return []
#
# # Function to download AIA multi-wavelength images
# def download_aia_data(start_date, end_date, dirs, wavelengths=None, sample_cadence='1d', max_files=20):
#     """
#     Download SDO/AIA images for specified wavelengths and time range
#
#     Parameters:
#     -----------
#     start_date, end_date : str
#         Date range in format 'YYYY-MM-DD'
#     wavelengths : list
#         List of wavelengths to download (in Angstrom)
#     sample_cadence : str
#         Time cadence for sampling data (to reduce volume)
#     max_files : int
#         Maximum number of files to download per wavelength (to control storage)
#     """
#     if wavelengths is None:
#         wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600]
#
#     print(f"Downloading SDO/AIA data for wavelengths {wavelengths} from {start_date} to {end_date}...")
#
#     downloaded_files = []
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     for wl in wavelengths:
#         print(f"Searching for AIA {wl}Å data...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.aia,
#             a.Wavelength(wl * u.angstrom),
#             # a.Sample(sample_cadence)
#             a.Sample(u.Quantity(sample_cadence))
#         )
#
#         # Limit the number of files to download
#         if len(result) > max_files:
#             print(f"Found {len(result)} files for {wl}Å, limiting to {max_files}")
#             result = result[:max_files]
#         else:
#             print(f"Found {len(result)} files for {wl}Å")
#
#         # Download the files
#         wl_files = Fido.fetch(result, path=dirs["sdo_aia"] / f"{wl}" / "{file}")
#         downloaded_files.extend(wl_files)
#
#     print(f"Downloaded {len(downloaded_files)} AIA image files")
#     return downloaded_files
#
#
# # Function to download GOES X-ray flux data
# def download_goes_data(start_date, end_date, dirs):
#     """
#     Download GOES X-ray flux data for a specified time range
#
#     Parameters:
#     -----------
#     start_date, end_date : str
#         Date range in format 'YYYY-MM-DD'
#     """
#     print(f"Downloading GOES X-ray flux data from {start_date} to {end_date}...")
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     # Query for GOES X-ray data
#     result = Fido.search(
#         time_range,
#         a.Instrument.xrs
#     )
#
#     print(f"Found {len(result)} GOES data files")
#
#     # Download the files
#     downloaded_files = Fido.fetch(result, path=dirs["goes"] / "{file}")
#
#     print(f"Downloaded {len(downloaded_files)} GOES X-ray files")
#     return downloaded_files
#
#
# # Function to extract flare events from GOES data
# def extract_flare_events(goes_files, min_class='C1.0'):
#     """
#     Extract flare events from GOES X-ray data files
#
#     Parameters:
#     -----------
#     goes_files : list
#         List of downloaded GOES data files
#     min_class : str
#         Minimum flare class to extract (default: 'C1.0')
#
#     Returns:
#     --------
#     DataFrame containing flare events
#     """
#     flares = []
#
#     for file in goes_files:
#         try:
#             goes_ts = TimeSeries(file)
#             # Extract flux data
#             flux_data = goes_ts.to_dataframe()
#
#             # Identify peaks (very simple algorithm - in production you'd use a more sophisticated peak detection)
#             # Add columns for smoothed data
#             flux_data['smooth_long'] = flux_data['xrsa'].rolling(window=30, center=True).mean()
#             flux_data['smooth_short'] = flux_data['xrsb'].rolling(window=30, center=True).mean()
#
#             # Find local maxima in the smoothed data (very basic peak detection)
#             peaks = []
#             for i in range(2, len(flux_data) - 2):
#                 if (flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i - 1] and
#                         flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i + 1] and
#                         flux_data['smooth_long'].iloc[i] > 1e-6):  # Arbitrary threshold
#
#                     # Calculate flare class
#                     flux_value = flux_data['xrsb'].iloc[i]
#                     flare_class = flux_to_class(flux_value)
#
#                     # Skip if below minimum class
#                     if not is_above_min_class(flare_class, min_class):
#                         continue
#
#                     # Record the flare
#                     peaks.append({
#                         'time': flux_data.index[i],
#                         'flux': flux_value,
#                         'class': flare_class
#                     })
#
#             flares.extend(peaks)
#
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#             continue
#
#     # Convert to DataFrame
#     flares_df = pd.DataFrame(flares)
#     if not flares_df.empty:
#         flares_df.sort_values('time', inplace=True)
#
#     return flares_df
#
#
# # Helper function to convert flux to flare class
# def flux_to_class(flux_value):
#     """Convert a flux value to a flare classification"""
#     if flux_value < 1e-6:
#         return 'A' + str(round(flux_value * 1e7, 1))
#     elif flux_value < 1e-5:
#         return 'B' + str(round(flux_value * 1e6, 1))
#     elif flux_value < 1e-4:
#         return 'C' + str(round(flux_value * 1e5, 1))
#     elif flux_value < 1e-3:
#         return 'M' + str(round(flux_value * 1e4, 1))
#     else:
#         return 'X' + str(round(flux_value * 1e3, 1))
#
#
# # Helper function to check if a flare class is above a minimum class
# def is_above_min_class(flare_class, min_class):
#     """Check if a flare class is above a minimum class"""
#     class_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
#
#     flare_letter = flare_class[0]
#     flare_number = float(flare_class[1:])
#
#     min_letter = min_class[0]
#     min_number = float(min_class[1:])
#
#     if class_levels[flare_letter] > class_levels[min_letter]:
#         return True
#     elif class_levels[flare_letter] == class_levels[min_letter]:
#         return flare_number >= min_number
#     else:
#         return False
#
#
# # Function to preprocess HMI magnetogram data
# def preprocess_hmi_data(hmi_files, output_dir):
#     """
#     Preprocess HMI magnetogram data
#
#     Parameters:
#     -----------
#     hmi_files : list
#         List of downloaded HMI data files
#     output_dir : Path
#         Directory to save processed data
#     """
#     for i, file in enumerate(hmi_files):
#         try:
#             # Load the FITS file
#             hmi_map = sunpy.map.Map(file)
#
#             # Extract the data and metadata
#             data = hmi_map.data
#             header = hmi_map.meta
#
#             # Apply basic preprocessing
#             # 1. Remove NaNs
#             data = np.nan_to_num(data)
#
#             # 2. Normalize data to [-1, 1] range
#             data_max = np.max(np.abs(data))
#             if data_max > 0:
#                 data = data / data_max
#
#             # Save processed data
#             np.save(output_dir / f"hmi_processed_{i}.npy", data)
#
#             # Save metadata
#             with open(output_dir / f"hmi_metadata_{i}.txt", 'w') as f:
#                 for key, value in header.items():
#                     f.write(f"{key}: {value}\n")
#
#         except Exception as e:
#             print(f"Error processing HMI file {file}: {e}")
#             continue
#
#
# def download_noaa_ar_data(start_date, end_date, dirs):
#     """Download NOAA Active Region data from SWPC API"""
#     print(f"Downloading NOAA AR data from {start_date} to {end_date}")
#
#     base_url = "https://services.swpc.noaa.gov/json/solar-region/"
#     downloaded_files = []
#
#     current_dt = datetime.strptime(start_date, '%Y-%m-%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#
#     while current_dt <= end_dt:
#         try:
#             date_str = current_dt.strftime('%Y-%m-%d')
#             url = f"{base_url}{date_str}.json"
#
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#
#             file_path = dirs["noaa_ar"] / f"noaa_ar_{date_str.replace('-', '')}.json"
#
#             # Validate data structure
#             data = response.json()
#             if 'regions' not in data or not isinstance(data['regions'], list):
#                 raise ValueError("Invalid NOAA AR data format")
#
#             with open(file_path, 'w') as f:
#                 json.dump(data, f)
#
#             downloaded_files.append(file_path)
#             print(f"Downloaded NOAA AR data for {date_str}")
#
#         except requests.HTTPError:
#             print(f"No NOAA AR data for {date_str}")
#         except Exception as e:
#             print(f"Error downloading NOAA AR data {date_str}: {str(e)[:50]}")
#
#         current_dt += timedelta(days=1)
#
#     return downloaded_files
#
# def download_soho_data(start_date, end_date, dirs, instruments=None, sample_cadence='1h', max_files=20):
#     """Download SOHO data with quality filtering"""
#     if instruments is None:
#         instruments = ['c2', 'c3']  # LASCO coronagraphs by default
#
#     print(f"Downloading SOHO data from {start_date} to {end_date}")
#
#     downloaded_files = []
#     time_range = a.Time(start_date, end_date)
#
#     for instrument in instruments:
#         print(f"Searching SOHO/LASCO {instrument.upper()} data...")
#
#         # Build quality-controlled query
#         result = Fido.search(
#             time_range,
#             a.Instrument.lasco,
#             a.Detector(instrument),
#             a.Physobs.intensity,
#             a.jsoc.Notify(os.getenv('JSOC_EMAIL')),  # Required for JSOC access
#             a.Sample(u.Quantity(sample_cadence)),
#             a.jsoc.Quality('<= 2')  # Only good quality data
#         )
#
#         if not result:
#             print(f"No data found for LASCO {instrument.upper()}")
#             continue
#
#         # Prioritize recent calibrations
#         result = sorted(result, key=lambda x: x['CALVER'], reverse=True)[:max_files]
#
#         print(f"Downloading {len(result)} LASCO {instrument.upper()} files...")
#         files = Fido.fetch(result, path=dirs["soho"] / f"lasco_{instrument}" / "{file}")
#         downloaded_files.extend(files)
#
#     return downloaded_files
#
#
# # Function to visualize data samples
# # def create_visualizations(hmi_files, aia_files, goes_files, output_dir):
# #     """
# #     Create visualizations of data samples
# #
# #     Parameters:
# #     -----------
# #     hmi_files : list
# #         List of downloaded HMI data files
# #     aia_files : list
# #         List of downloaded AIA data files
# #     goes_files : list
# #         List of downloaded GOES data files
# #     output_dir : Path
# #         Directory to save visualizations
# #     """
# #     # Visualize HMI magnetogram samples
# #     # for i, file in enumerate(hmi_files[:5]):
# #     #     try:
# #     #         hmi_map = sunpy.map.Map(file)
# #     #
# #     #         plt.figure(figsize=(10, 8))
# #     #         hmi_map.plot()
# #     #         plt.colorbar(label='Magnetic Field Strength (Gauss)')
# #     #         plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
# #     #         plt.tight_layout()
# #     #         plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
# #     #         plt.close()
# #     #     except Exception as e:
# #     #         print(f"Error visualizing HMI file {file}: {e}")
# #     #         continue
# #
# #     # Visualize HMI magnetogram samples
# #     for i, file in enumerate(hmi_files[:15]):
# #         try:
# #             hmi_map = sunpy.map.Map(file)
# #
# #             plt.figure(figsize=(10, 8))
# #             # Use a diverging colormap and set reasonable data limits
# #             norm = plt.Normalize(-200, 200)  # Adjust range based on your data
# #             hmi_map.plot(cmap='RdBu_r', norm=norm)
# #             plt.colorbar(label='Magnetic Field Strength (Gauss)')
# #             plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
# #             plt.tight_layout()
# #             plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
# #             plt.close()
# #         except Exception as e:
# #             print(f"Error visualizing HMI file {file}: {e}")
# #             continue
# #
# #     # Visualize AIA samples (for a few wavelengths)
# #     aia_by_wavelength = {}
# #     for file in aia_files:
# #         try:
# #             aia_map = sunpy.map.Map(file)
# #             wl = int(aia_map.wavelength.value)
# #             if wl not in aia_by_wavelength:
# #                 aia_by_wavelength[wl] = []
# #             aia_by_wavelength[wl].append(file)
# #         except Exception as e:
# #             print(f"Error categorizing AIA file {file}: {e}")
# #             continue
# #
# #     # Create visualizations for each wavelength
# #     for wl, files in aia_by_wavelength.items():
# #         for i, file in enumerate(files[:5]):
# #             try:
# #                 aia_map = sunpy.map.Map(file)
# #
# #                 plt.figure(figsize=(10, 8))
# #                 aia_map.plot()
# #                 plt.colorbar(label='Intensity')
# #                 plt.title(f'SDO/AIA {wl}Å - {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
# #                 plt.tight_layout()
# #                 plt.savefig(output_dir / f"aia_{wl}A_sample_{i}.png", dpi=300)
# #                 plt.close()
# #             except Exception as e:
# #                 print(f"Error visualizing AIA file {file}: {e}")
# #                 continue
# #
# #     # Visualize GOES X-ray flux
# #     for i, file in enumerate(goes_files[:5]):
# #         try:
# #             goes_ts = TimeSeries(file)
# #
# #             plt.figure(figsize=(12, 6))
# #             goes_ts.plot()
# #             plt.title(f'GOES X-ray Flux - {goes_ts.source}')
# #             plt.tight_layout()
# #             plt.savefig(output_dir / f"goes_sample_{i}.png", dpi=300)
# #             plt.close()
# #         except Exception as e:
# #             print(f"Error visualizing GOES file {file}: {e}")
# #             continue
#
#
# def create_visualizations(hmi_files, aia_files, goes_files, soho_files, noaa_ar_files, output_dir):
#     """
#     Create visualizations of data samples
#
#     Parameters:
#     -----------
#     hmi_files : list
#         List of downloaded HMI data files
#     aia_files : list
#         List of downloaded AIA data files
#     goes_files : list
#         List of downloaded GOES data files
#     soho_files : list
#         List of downloaded SOHO data files
#     noaa_ar_files : list
#         List of downloaded NOAA AR data files
#     output_dir : Path
#         Directory to save visualizations
#     """
#     # Existing visualization code for HMI, AIA, and GOES...
#
#     # Visualize HMI magnetogram samples
#     for i, file in enumerate(hmi_files[:15]):
#         try:
#             hmi_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 8))
#             # Use a diverging colormap and set reasonable data limits
#             norm = plt.Normalize(-200, 200)  # Adjust range based on your data
#             hmi_map.plot(cmap='RdBu_r', norm=norm)
#             plt.colorbar(label='Magnetic Field Strength (Gauss)')
#             plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing HMI file {file}: {e}")
#             continue
#
#     # Visualize AIA samples (for a few wavelengths)
#     aia_by_wavelength = {}
#     for file in aia_files:
#         try:
#             aia_map = sunpy.map.Map(file)
#             wl = int(aia_map.wavelength.value)
#             if wl not in aia_by_wavelength:
#                 aia_by_wavelength[wl] = []
#             aia_by_wavelength[wl].append(file)
#         except Exception as e:
#             print(f"Error categorizing AIA file {file}: {e}")
#             continue
#
#     # Create visualizations for each wavelength
#     for wl, files in aia_by_wavelength.items():
#         for i, file in enumerate(files[:5]):
#             try:
#                 aia_map = sunpy.map.Map(file)
#
#                 plt.figure(figsize=(10, 8))
#                 aia_map.plot()
#                 plt.colorbar(label='Intensity')
#                 plt.title(f'SDO/AIA {wl}Å - {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"aia_{wl}A_sample_{i}.png", dpi=300)
#                 plt.close()
#             except Exception as e:
#                 print(f"Error visualizing AIA file {file}: {e}")
#                 continue
#
#     # Visualize GOES X-ray flux
#     for i, file in enumerate(goes_files[:5]):
#         try:
#             goes_ts = TimeSeries(file)
#
#             plt.figure(figsize=(12, 6))
#             goes_ts.plot()
#             plt.title(f'GOES X-ray Flux - {goes_ts.source}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"goes_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing GOES file {file}: {e}")
#             continue
#
#     # Add visualization for SOHO LASCO coronagraph data
#     for i, file in enumerate(soho_files[:5]):
#         try:
#             # Try to detect instrument from filename or path
#             if 'c2' in str(file).lower():
#                 instrument = 'C2'
#             elif 'c3' in str(file).lower():
#                 instrument = 'C3'
#             else:
#                 instrument = 'LASCO'
#
#             soho_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 10))
#             soho_map.plot()
#             plt.title(f'SOHO/LASCO {instrument} - {soho_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"soho_lasco_{instrument.lower()}_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing SOHO file {file}: {e}")
#             continue
#
#     # Add visualization for NOAA Active Region data
#     # This is more challenging since NOAA AR data typically comes as text reports
#     # Here's a simple approach to visualize basic information
#     for i, file in enumerate(noaa_ar_files[:5]):
#         try:
#             # Check if it's a text file
#             if str(file).endswith('.txt'):
#                 with open(file, 'r') as f:
#                     content = f.readlines()
#
#                 # Create a simple visualization of the content
#                 plt.figure(figsize=(12, 8))
#                 plt.text(0.1, 0.5, ''.join(content[:30]), fontsize=10)  # Show first 30 lines
#                 plt.axis('off')
#                 plt.title(f'NOAA Active Region Data - {Path(file).name}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"noaa_ar_sample_{i}.png", dpi=300)
#                 plt.close()
#             # If it's actual NOAA data in a different format, we would need format-specific code
#         except Exception as e:
#             print(f"Error visualizing NOAA AR file {file}: {e}")
#             continue
#
#
#
# # Main function to run the entire workflow
# def main():
#     # Define date range (starting with a smaller range)
#     start_date = '2018-01-01'
#     end_date = '2018-01-05'  # Just 5 days to start with
#
#     dirs = create_directories()
#
#     # Download HMI magnetogram data
#     hmi_files = download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10)
#
#     # Download AIA data for a few wavelengths to save space
#     aia_files = download_aia_data(start_date, end_date, dirs, wavelengths=[171, 193, 304], sample_cadence='1d', max_files=5)
#
#     # Download GOES X-ray flux data
#     goes_files = download_goes_data(start_date, end_date, dirs)
#
#     # Extract flare events
#     flares_df = extract_flare_events(goes_files)
#     if not flares_df.empty:
#         print(f"Extracted {len(flares_df)} flare events")
#         flares_df.to_csv(dirs["processed"] / "flare_events.csv", index=False)
#     else:
#         print("No flare events found in the data")
#
#     # Preprocess HMI data
#     preprocess_hmi_data(hmi_files, dirs["processed"])
#
#     # Create visualizations
#     create_visualizations(hmi_files, aia_files, goes_files, dirs["visualizations"])
#
#     print("\nData download and preprocessing complete!")
#     print(f"Processed data saved to: {dirs['processed']}")
#     print(f"Visualizations saved to: {dirs['visualizations']}")
#
#     # Print storage usage
#     total_size = 0
#     for path, dirs, files in os.walk('./solar_data'):
#         for f in files:
#             fp = os.path.join(path, f)
#             total_size += os.path.getsize(fp)
#
#     print(f"\nTotal storage used: {total_size / (1024 ** 3):.2f} GB")
#
#
# if __name__ == "__main__":
#     main()


# import os
# import numpy as np
# import pandas as pd
# import json
# import requests
# import matplotlib.pyplot as plt
# import sunpy.map
# import sunpy.timeseries
# from sunpy.timeseries import TimeSeries
# from sunpy.net import Fido
# from sunpy.net import attrs as a
# import astropy.units as u
# from pathlib import Path
# from datetime import datetime, timedelta
#
#
# def create_directories(base_dir="./solar_data"):
#     base_dir = Path(base_dir)
#     dirs = {
#         "sdo_hmi": base_dir / "sdo_hmi",
#         "sdo_aia": base_dir / "sdo_aia",
#         "goes": base_dir / "goes",
#         "noaa_ar": base_dir / "noaa_ar",
#         "soho": base_dir / "soho",
#         "processed": base_dir / "processed",
#         "visualizations": base_dir / "visualizations"
#     }
#
#     for dir_path in dirs.values():
#         dir_path.mkdir(parents=True, exist_ok=True)
#
#     return dirs
#
#
# def download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10):
#     print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     # Query for LOS magnetograms (series: hmi.M_720s)
#     result = Fido.search(
#         time_range,
#         a.Instrument.hmi,
#         a.Physobs.los_magnetic_field,
#         a.Sample(u.Quantity(sample_cadence))
#     )
#
#     if len(result) == 0:
#         print("No results with default query. Trying with specific data series...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.hmi,
#             a.Physobs.los_magnetic_field,
#             a.Sample(u.Quantity(sample_cadence)),
#             a.Series('hmi.M_720s')
#         )
#
#     if len(result) > max_files:
#         print(f"Found {len(result)} files, limiting to {max_files}")
#         result = result[:max_files]
#     else:
#         print(f"Found {len(result)} files")
#
#     if len(result) > 0:
#         downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}")
#         print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
#         return downloaded_files
#     else:
#         print("No HMI files found to download")
#         return []
#
#
# def download_aia_data(start_date, end_date, dirs, wavelengths=None, sample_cadence='1d', max_files=20):
#     if wavelengths is None:
#         wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600]
#
#     # Ensure wavelengths are integers
#     wavelengths = [int(wl) if isinstance(wl, str) else wl for wl in wavelengths]
#
#     print(f"Downloading SDO/AIA data for wavelengths {wavelengths} from {start_date} to {end_date}...")
#
#     downloaded_files = []
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     for wl in wavelengths:
#         print(f"Searching for AIA {wl}Å data...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.aia,
#             a.Wavelength(wl * u.angstrom),
#             a.Sample(u.Quantity(sample_cadence))
#         )
#
#         if len(result) > max_files:
#             print(f"Found {len(result)} files for {wl}Å, limiting to {max_files}")
#             result = result[:max_files]
#         else:
#             print(f"Found {len(result)} files for {wl}Å")
#
#         if len(result) > 0:
#             wl_files = Fido.fetch(result, path=dirs["sdo_aia"] / f"{wl}" / "{file}")
#             downloaded_files.extend(wl_files)
#         else:
#             print(f"No files found for AIA {wl}Å")
#
#     print(f"Downloaded {len(downloaded_files)} AIA image files")
#     return downloaded_files
#
#
# def download_goes_data(start_date, end_date, dirs):
#     print(f"Downloading GOES X-ray flux data from {start_date} to {end_date}...")
#
#     time_range = a.Time(start_date, end_date)
#
#     result = Fido.search(
#         time_range,
#         a.Instrument.xrs
#     )
#
#     print(f"Found {len(result)} GOES data files")
#
#     if len(result) > 0:
#         # Download the files
#         downloaded_files = Fido.fetch(result, path=dirs["goes"] / "{file}")
#         print(f"Downloaded {len(downloaded_files)} GOES X-ray files")
#         return downloaded_files
#     else:
#         print("No GOES files found to download")
#         return []
#
#
# def extract_flare_events(goes_files, min_class='C1.0'):
#     flares = []
#
#     for file in goes_files:
#         try:
#             goes_ts = TimeSeries(file)
#             flux_data = goes_ts.to_dataframe()
#
#             flux_data['smooth_long'] = flux_data['xrsa'].rolling(window=30, center=True).mean()
#             flux_data['smooth_short'] = flux_data['xrsb'].rolling(window=30, center=True).mean()
#
#             peaks = []
#             for i in range(2, len(flux_data) - 2):
#                 if (flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i - 1] and
#                         flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i + 1] and
#                         flux_data['smooth_long'].iloc[i] > 1e-6):
#
#                     flux_value = flux_data['xrsb'].iloc[i]
#                     flare_class = flux_to_class(flux_value)
#
#                     if not is_above_min_class(flare_class, min_class):
#                         continue
#
#                     peaks.append({
#                         'time': flux_data.index[i],
#                         'flux': flux_value,
#                         'class': flare_class
#                     })
#
#             flares.extend(peaks)
#
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#             continue
#
#     flares_df = pd.DataFrame(flares)
#     if not flares_df.empty:
#         flares_df.sort_values('time', inplace=True)
#
#     return flares_df
#
#
# def flux_to_class(flux_value):
#     if flux_value < 1e-6:
#         return 'A' + str(round(flux_value * 1e7, 1))
#     elif flux_value < 1e-5:
#         return 'B' + str(round(flux_value * 1e6, 1))
#     elif flux_value < 1e-4:
#         return 'C' + str(round(flux_value * 1e5, 1))
#     elif flux_value < 1e-3:
#         return 'M' + str(round(flux_value * 1e4, 1))
#     else:
#         return 'X' + str(round(flux_value * 1e3, 1))
#
#
# def is_above_min_class(flare_class, min_class):
#     class_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
#
#     flare_letter = flare_class[0]
#     flare_number = float(flare_class[1:])
#
#     min_letter = min_class[0]
#     min_number = float(min_class[1:])
#
#     if class_levels[flare_letter] > class_levels[min_letter]:
#         return True
#     elif class_levels[flare_letter] == class_levels[min_letter]:
#         return flare_number >= min_number
#     else:
#         return False
#
#
# def preprocess_hmi_data(hmi_files, output_dir):
#     for i, file in enumerate(hmi_files):
#         try:
#             hmi_map = sunpy.map.Map(file)
#
#             data = hmi_map.data
#             header = hmi_map.meta
#
#             data = np.nan_to_num(data)
#
#             # 2. Normalize data to [-1, 1] range
#             data_max = np.max(np.abs(data))
#             if data_max > 0:
#                 data = data / data_max
#
#             # Save processed data
#             np.save(output_dir / f"hmi_processed_{i}.npy", data)
#
#             # Save metadata
#             with open(output_dir / f"hmi_metadata_{i}.txt", 'w') as f:
#                 for key, value in header.items():
#                     f.write(f"{key}: {value}\n")
#
#         except Exception as e:
#             print(f"Error processing HMI file {file}: {e}")
#             continue
#
#
# def download_noaa_ar_data(start_date, end_date, dirs):
#     print(f"Downloading NOAA AR data from {start_date} to {end_date}")
#
#     base_url = "https://services.swpc.noaa.gov/json/solar-region/"
#     downloaded_files = []
#
#     # Convert string dates to datetime objects
#     current_dt = datetime.strptime(start_date, '%Y-%m-%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#
#     while current_dt <= end_dt:
#         try:
#             date_str = current_dt.strftime('%Y-%m-%d')
#             url = f"{base_url}{date_str}.json"
#
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#
#             file_path = dirs["noaa_ar"] / f"noaa_ar_{date_str.replace('-', '')}.json"
#
#             # Validate data structure
#             data = response.json()
#             if isinstance(data, dict) and 'regions' in data and isinstance(data['regions'], list):
#                 with open(file_path, 'w') as f:
#                     json.dump(data, f)
#                 downloaded_files.append(file_path)
#                 print(f"Downloaded NOAA AR data for {date_str}")
#             else:
#                 print(f"Invalid NOAA AR data format for {date_str}")
#
#         except requests.HTTPError:
#             print(f"No NOAA AR data for {date_str}")
#         except Exception as e:
#             print(f"Error downloading NOAA AR data {date_str}: {str(e)[:50]}")
#
#         current_dt += timedelta(days=1)
#
#     return downloaded_files
#
#
# def download_soho_data(start_date, end_date, dirs, instruments=None, sample_cadence='1h', max_files=20):
#     if instruments is None:
#         instruments = ['c2', 'c3']  # LASCO coronagraphs by default
#
#     print(f"Downloading SOHO data from {start_date} to {end_date}")
#
#     downloaded_files = []
#     time_range = a.Time(start_date, end_date)
#
#     for instrument in instruments:
#         print(f"Searching SOHO/LASCO {instrument.upper()} data...")
#
#         try:
#             result = Fido.search(
#                 time_range,
#                 a.Instrument.lasco,
#                 a.Detector(instrument),
#                 a.Sample(u.Quantity(sample_cadence))
#             )
#
#             if not result or len(result) == 0:
#                 print(f"No data found for LASCO {instrument.upper()}")
#                 continue
#
#             if len(result) > max_files:
#                 print(f"Found {len(result)} files for LASCO {instrument.upper()}, limiting to {max_files}")
#                 result = result[:max_files]
#             else:
#                 print(f"Found {len(result)} files for LASCO {instrument.upper()}")
#
#             print(f"Downloading LASCO {instrument.upper()} files...")
#             files = Fido.fetch(result, path=dirs["soho"] / f"lasco_{instrument}" / "{file}")
#             downloaded_files.extend(files)
#         except Exception as e:
#             print(f"Error searching/fetching SOHO/LASCO {instrument.upper()} data: {e}")
#             continue
#
#     return downloaded_files
#
#
# def create_visualizations(hmi_files, aia_files, goes_files, soho_files, noaa_ar_files, output_dir):
#     # Create output directory if it doesn't exist
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # HMI visualizations
#     for i, file in enumerate(hmi_files[:5]):  # Limit to first 5 files
#         try:
#             hmi_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 8))
#             norm = plt.Normalize(-200, 200)  # Adjust range based on your data
#             hmi_map.plot(cmap='RdBu_r', norm=norm)
#             plt.colorbar(label='Magnetic Field Strength (Gauss)')
#             plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing HMI file {file}: {e}")
#             continue
#
#     # AIA visualizations
#     aia_by_wavelength = {}
#     for file in aia_files:
#         try:
#             aia_map = sunpy.map.Map(file)
#             wl = int(aia_map.wavelength.value)
#             if wl not in aia_by_wavelength:
#                 aia_by_wavelength[wl] = []
#             aia_by_wavelength[wl].append(file)
#         except Exception as e:
#             print(f"Error categorizing AIA file {file}: {e}")
#             continue
#
#     for wl, files in aia_by_wavelength.items():
#         for i, file in enumerate(files[:2]):  # Limit to first 2 files per wavelength
#             try:
#                 aia_map = sunpy.map.Map(file)
#
#                 plt.figure(figsize=(10, 8))
#                 aia_map.plot()
#                 plt.colorbar(label='Intensity')
#                 plt.title(f'SDO/AIA {wl}Å - {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"aia_{wl}A_sample_{i}.png", dpi=300)
#                 plt.close()
#             except Exception as e:
#                 print(f"Error visualizing AIA file {file}: {e}")
#                 continue
#
#     # GOES visualizations
#     for i, file in enumerate(goes_files[:2]):  # Limit to first 2 files
#         try:
#             goes_ts = TimeSeries(file)
#
#             plt.figure(figsize=(12, 6))
#             goes_ts.plot()
#             plt.title(f'GOES X-ray Flux - {goes_ts.source}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"goes_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing GOES file {file}: {e}")
#             continue
#
#     # SOHO visualizations
#     for i, file in enumerate(soho_files[:5]):  # Limit to first 5 files
#         try:
#             if 'c2' in str(file).lower():
#                 instrument = 'C2'
#             elif 'c3' in str(file).lower():
#                 instrument = 'C3'
#             else:
#                 instrument = 'LASCO'
#
#             soho_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 10))
#             soho_map.plot()
#             plt.title(f'SOHO/LASCO {instrument} - {soho_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"soho_lasco_{instrument.lower()}_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing SOHO file {file}: {e}")
#             continue
#
#     # NOAA AR visualizations
#     for i, file in enumerate(noaa_ar_files[:3]):  # Limit to first 3 files
#         try:
#             file_path = Path(file)
#             if file_path.suffix == '.json':
#                 with open(file_path, 'r') as f:
#                     ar_data = json.load(f)
#
#                 # Create a summary visualization of the active regions
#                 plt.figure(figsize=(12, 8))
#
#                 if 'regions' in ar_data and ar_data['regions']:
#                     date_str = file_path.stem.split('_')[-1]
#                     formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
#
#                     plt.text(0.05, 0.95, f"NOAA Active Regions: {formatted_date}", fontsize=14,
#                              transform=plt.gca().transAxes, verticalalignment='top')
#
#                     for i, region in enumerate(ar_data['regions'][:10]):  # Show first 10 regions
#                         y_pos = 0.85 - i * 0.08
#                         region_text = f"AR {region.get('number', 'N/A')}: "
#                         region_text += f"Loc: {region.get('location', 'N/A')}, "
#                         region_text += f"Class: {region.get('class', 'N/A')}, "
#                         region_text += f"Area: {region.get('area', 'N/A')}"
#
#                         plt.text(0.05, y_pos, region_text, fontsize=10,
#                                  transform=plt.gca().transAxes)
#                 else:
#                     plt.text(0.5, 0.5, "No active regions data available",
#                              fontsize=12, ha='center', transform=plt.gca().transAxes)
#
#                 plt.axis('off')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"noaa_ar_summary_{i}.png", dpi=300)
#                 plt.close()
#
#             elif file_path.suffix == '.txt':
#                 with open(file_path, 'r') as f:
#                     content = f.readlines()
#
#                 plt.figure(figsize=(12, 8))
#                 plt.text(0.1, 0.5, ''.join(content[:30]), fontsize=10)  # Show first 30 lines
#                 plt.axis('off')
#                 plt.title(f'NOAA Active Region Data - {file_path.name}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"noaa_ar_sample_{i}.png", dpi=300)
#                 plt.close()
#
#         except Exception as e:
#             print(f"Error visualizing NOAA AR file {file}: {e}")
#             continue
# import os
# import numpy as np
# import pandas as pd
# import json
# import requests
# import matplotlib.pyplot as plt
# import sunpy.map
# import sunpy.timeseries
# from sunpy.timeseries import TimeSeries
# from sunpy.net import Fido
# from sunpy.net import attrs as a
# import astropy.units as u
# from pathlib import Path
# from datetime import datetime, timedelta
#
#
# def create_directories(base_dir="./solar_data"):
#     base_dir = Path(base_dir)
#     dirs = {
#         "sdo_hmi": base_dir / "sdo_hmi",
#         "sdo_aia": base_dir / "sdo_aia",
#         "goes": base_dir / "goes",
#         "noaa_ar": base_dir / "noaa_ar",
#         "soho": base_dir / "soho",
#         "processed": base_dir / "processed",
#         "visualizations": base_dir / "visualizations"
#     }
#
#     for dir_path in dirs.values():
#         dir_path.mkdir(parents=True, exist_ok=True)
#
#     return dirs
#
#
# def download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10):
#     print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     # Query for LOS magnetograms (series: hmi.M_720s)
#     result = Fido.search(
#         time_range,
#         a.Instrument.hmi,
#         a.Physobs.los_magnetic_field,
#         a.Sample(u.Quantity(sample_cadence))
#     )
#
#     if len(result) == 0:
#         print("No results with default query. Trying with specific data series...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.hmi,
#             a.Physobs.los_magnetic_field,
#             a.Sample(u.Quantity(sample_cadence)),
#             a.Series('hmi.M_720s')
#         )
#
#     if len(result) > max_files:
#         print(f"Found {len(result)} files, limiting to {max_files}")
#         result = result[:max_files]
#     else:
#         print(f"Found {len(result)} files")
#
#     if len(result) > 0:
#         downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}")
#         print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
#         return downloaded_files
#     else:
#         print("No HMI files found to download")
#         return []
#
#
# def download_aia_data(start_date, end_date, dirs, wavelengths=None, sample_cadence='1d', max_files=20):
#     if wavelengths is None:
#         wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600]
#
#     # Ensure wavelengths are integers
#     wavelengths = [int(wl) if isinstance(wl, str) else wl for wl in wavelengths]
#
#     print(f"Downloading SDO/AIA data for wavelengths {wavelengths} from {start_date} to {end_date}...")
#
#     downloaded_files = []
#
#     # Define time range
#     time_range = a.Time(start_date, end_date)
#
#     for wl in wavelengths:
#         print(f"Searching for AIA {wl}Å data...")
#         result = Fido.search(
#             time_range,
#             a.Instrument.aia,
#             a.Wavelength(wl * u.angstrom),
#             a.Sample(u.Quantity(sample_cadence))
#         )
#
#         if len(result) > max_files:
#             print(f"Found {len(result)} files for {wl}Å, limiting to {max_files}")
#             result = result[:max_files]
#         else:
#             print(f"Found {len(result)} files for {wl}Å")
#
#         if len(result) > 0:
#             wl_files = Fido.fetch(result, path=dirs["sdo_aia"] / f"{wl}" / "{file}")
#             downloaded_files.extend(wl_files)
#         else:
#             print(f"No files found for AIA {wl}Å")
#
#     print(f"Downloaded {len(downloaded_files)} AIA image files")
#     return downloaded_files
#
#
# def download_goes_data(start_date, end_date, dirs):
#     print(f"Downloading GOES X-ray flux data from {start_date} to {end_date}...")
#
#     time_range = a.Time(start_date, end_date)
#
#     result = Fido.search(
#         time_range,
#         a.Instrument.xrs
#     )
#
#     print(f"Found {len(result)} GOES data files")
#
#     if len(result) > 0:
#         # Download the files
#         downloaded_files = Fido.fetch(result, path=dirs["goes"] / "{file}")
#         print(f"Downloaded {len(downloaded_files)} GOES X-ray files")
#         return downloaded_files
#     else:
#         print("No GOES files found to download")
#         return []
#
#
# def extract_flare_events(goes_files, min_class='C1.0'):
#     flares = []
#
#     for file in goes_files:
#         try:
#             goes_ts = TimeSeries(file)
#             flux_data = goes_ts.to_dataframe()
#
#             flux_data['smooth_long'] = flux_data['xrsa'].rolling(window=30, center=True).mean()
#             flux_data['smooth_short'] = flux_data['xrsb'].rolling(window=30, center=True).mean()
#
#             peaks = []
#             for i in range(2, len(flux_data) - 2):
#                 if (flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i - 1] and
#                         flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i + 1] and
#                         flux_data['smooth_long'].iloc[i] > 1e-6):
#
#                     flux_value = flux_data['xrsb'].iloc[i]
#                     flare_class = flux_to_class(flux_value)
#
#                     if not is_above_min_class(flare_class, min_class):
#                         continue
#
#                     peaks.append({
#                         'time': flux_data.index[i],
#                         'flux': flux_value,
#                         'class': flare_class
#                     })
#
#             flares.extend(peaks)
#
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#             continue
#
#     flares_df = pd.DataFrame(flares)
#     if not flares_df.empty:
#         flares_df.sort_values('time', inplace=True)
#
#     return flares_df
#
#
# def flux_to_class(flux_value):
#     if flux_value < 1e-6:
#         return 'A' + str(round(flux_value * 1e7, 1))
#     elif flux_value < 1e-5:
#         return 'B' + str(round(flux_value * 1e6, 1))
#     elif flux_value < 1e-4:
#         return 'C' + str(round(flux_value * 1e5, 1))
#     elif flux_value < 1e-3:
#         return 'M' + str(round(flux_value * 1e4, 1))
#     else:
#         return 'X' + str(round(flux_value * 1e3, 1))
#
#
# def is_above_min_class(flare_class, min_class):
#     class_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
#
#     flare_letter = flare_class[0]
#     flare_number = float(flare_class[1:])
#
#     min_letter = min_class[0]
#     min_number = float(min_class[1:])
#
#     if class_levels[flare_letter] > class_levels[min_letter]:
#         return True
#     elif class_levels[flare_letter] == class_levels[min_letter]:
#         return flare_number >= min_number
#     else:
#         return False
#
#
# def preprocess_hmi_data(hmi_files, output_dir):
#     for i, file in enumerate(hmi_files):
#         try:
#             hmi_map = sunpy.map.Map(file)
#
#             data = hmi_map.data
#             header = hmi_map.meta
#
#             data = np.nan_to_num(data)
#
#             # 2. Normalize data to [-1, 1] range
#             data_max = np.max(np.abs(data))
#             if data_max > 0:
#                 data = data / data_max
#
#             # Save processed data
#             np.save(output_dir / f"hmi_processed_{i}.npy", data)
#
#             # Save metadata
#             with open(output_dir / f"hmi_metadata_{i}.txt", 'w') as f:
#                 for key, value in header.items():
#                     f.write(f"{key}: {value}\n")
#
#         except Exception as e:
#             print(f"Error processing HMI file {file}: {e}")
#             continue
#
#
# def download_noaa_ar_data(start_date, end_date, dirs):
#     print(f"Downloading NOAA AR data from {start_date} to {end_date}")
#
#     base_url = "https://services.swpc.noaa.gov/json/solar-region/"
#     downloaded_files = []
#
#     # Convert string dates to datetime objects
#     current_dt = datetime.strptime(start_date, '%Y-%m-%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#
#     while current_dt <= end_dt:
#         try:
#             date_str = current_dt.strftime('%Y-%m-%d')
#             url = f"{base_url}{date_str}.json"
#
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#
#             file_path = dirs["noaa_ar"] / f"noaa_ar_{date_str.replace('-', '')}.json"
#
#             # Validate data structure
#             data = response.json()
#             if isinstance(data, dict) and 'regions' in data and isinstance(data['regions'], list):
#                 with open(file_path, 'w') as f:
#                     json.dump(data, f)
#                 downloaded_files.append(file_path)
#                 print(f"Downloaded NOAA AR data for {date_str}")
#             else:
#                 print(f"Invalid NOAA AR data format for {date_str}")
#
#         except requests.HTTPError:
#             print(f"No NOAA AR data for {date_str}")
#         except Exception as e:
#             print(f"Error downloading NOAA AR data {date_str}: {str(e)[:50]}")
#
#         current_dt += timedelta(days=1)
#
#     return downloaded_files
#
#
# def download_soho_data(start_date, end_date, dirs, instruments=None, sample_cadence='1h', max_files=20):
#     if instruments is None:
#         instruments = ['c2', 'c3']  # LASCO coronagraphs by default
#
#     print(f"Downloading SOHO data from {start_date} to {end_date}")
#
#     downloaded_files = []
#     time_range = a.Time(start_date, end_date)
#
#     for instrument in instruments:
#         print(f"Searching SOHO/LASCO {instrument.upper()} data...")
#
#         try:
#             result = Fido.search(
#                 time_range,
#                 a.Instrument.lasco,
#                 a.Detector(instrument),
#                 a.Sample(u.Quantity(sample_cadence))
#             )
#
#             if not result or len(result) == 0:
#                 print(f"No data found for LASCO {instrument.upper()}")
#                 continue
#
#             if len(result) > max_files:
#                 print(f"Found {len(result)} files for LASCO {instrument.upper()}, limiting to {max_files}")
#                 result = result[:max_files]
#             else:
#                 print(f"Found {len(result)} files for LASCO {instrument.upper()}")
#
#             print(f"Downloading LASCO {instrument.upper()} files...")
#             files = Fido.fetch(result, path=dirs["soho"] / f"lasco_{instrument}" / "{file}")
#             downloaded_files.extend(files)
#         except Exception as e:
#             print(f"Error searching/fetching SOHO/LASCO {instrument.upper()} data: {e}")
#             continue
#
#     return downloaded_files
#
#
# def create_visualizations(hmi_files, aia_files, goes_files, soho_files, noaa_ar_files, output_dir):
#     # Create output directory if it doesn't exist
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # HMI visualizations
#     for i, file in enumerate(hmi_files[:5]):  # Limit to first 5 files
#         try:
#             hmi_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 8))
#             norm = plt.Normalize(-200, 200)  # Adjust range based on your data
#             hmi_map.plot(cmap='RdBu_r', norm=norm)
#             plt.colorbar(label='Magnetic Field Strength (Gauss)')
#             plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing HMI file {file}: {e}")
#             continue
#
#     # AIA visualizations
#     aia_by_wavelength = {}
#     for file in aia_files:
#         try:
#             aia_map = sunpy.map.Map(file)
#             wl = int(aia_map.wavelength.value)
#             if wl not in aia_by_wavelength:
#                 aia_by_wavelength[wl] = []
#             aia_by_wavelength[wl].append(file)
#         except Exception as e:
#             print(f"Error categorizing AIA file {file}: {e}")
#             continue
#
#     for wl, files in aia_by_wavelength.items():
#         for i, file in enumerate(files[:2]):  # Limit to first 2 files per wavelength
#             try:
#                 aia_map = sunpy.map.Map(file)
#
#                 plt.figure(figsize=(10, 8))
#                 aia_map.plot()
#                 plt.colorbar(label='Intensity')
#                 plt.title(f'SDO/AIA {wl}Å - {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"aia_{wl}A_sample_{i}.png", dpi=300)
#                 plt.close()
#             except Exception as e:
#                 print(f"Error visualizing AIA file {file}: {e}")
#                 continue
#
#     # GOES visualizations
#     for i, file in enumerate(goes_files[:2]):  # Limit to first 2 files
#         try:
#             goes_ts = TimeSeries(file)
#
#             plt.figure(figsize=(12, 6))
#             goes_ts.plot()
#             plt.title(f'GOES X-ray Flux - {goes_ts.source}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"goes_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing GOES file {file}: {e}")
#             continue
#
#     # SOHO visualizations
#     for i, file in enumerate(soho_files[:5]):  # Limit to first 5 files
#         try:
#             if 'c2' in str(file).lower():
#                 instrument = 'C2'
#             elif 'c3' in str(file).lower():
#                 instrument = 'C3'
#             else:
#                 instrument = 'LASCO'
#
#             soho_map = sunpy.map.Map(file)
#
#             plt.figure(figsize=(10, 10))
#             soho_map.plot()
#             plt.title(f'SOHO/LASCO {instrument} - {soho_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
#             plt.tight_layout()
#             plt.savefig(output_dir / f"soho_lasco_{instrument.lower()}_sample_{i}.png", dpi=300)
#             plt.close()
#         except Exception as e:
#             print(f"Error visualizing SOHO file {file}: {e}")
#             continue
#
#     # NOAA AR visualizations
#     for i, file in enumerate(noaa_ar_files[:3]):  # Limit to first 3 files
#         try:
#             file_path = Path(file)
#             if file_path.suffix == '.json':
#                 with open(file_path, 'r') as f:
#                     ar_data = json.load(f)
#
#                 # Create a summary visualization of the active regions
#                 plt.figure(figsize=(12, 8))
#
#                 if 'regions' in ar_data and ar_data['regions']:
#                     date_str = file_path.stem.split('_')[-1]
#                     formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
#
#                     plt.text(0.05, 0.95, f"NOAA Active Regions: {formatted_date}", fontsize=14,
#                              transform=plt.gca().transAxes, verticalalignment='top')
#
#                     for i, region in enumerate(ar_data['regions'][:10]):  # Show first 10 regions
#                         y_pos = 0.85 - i * 0.08
#                         region_text = f"AR {region.get('number', 'N/A')}: "
#                         region_text += f"Loc: {region.get('location', 'N/A')}, "
#                         region_text += f"Class: {region.get('class', 'N/A')}, "
#                         region_text += f"Area: {region.get('area', 'N/A')}"
#
#                         plt.text(0.05, y_pos, region_text, fontsize=10,
#                                  transform=plt.gca().transAxes)
#                 else:
#                     plt.text(0.5, 0.5, "No active regions data available",
#                              fontsize=12, ha='center', transform=plt.gca().transAxes)
#
#                 plt.axis('off')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"noaa_ar_summary_{i}.png", dpi=300)
#                 plt.close()
#
#             elif file_path.suffix == '.txt':
#                 with open(file_path, 'r') as f:
#                     content = f.readlines()
#
#                 plt.figure(figsize=(12, 8))
#                 plt.text(0.1, 0.5, ''.join(content[:30]), fontsize=10)  # Show first 30 lines
#                 plt.axis('off')
#                 plt.title(f'NOAA Active Region Data - {file_path.name}')
#                 plt.tight_layout()
#                 plt.savefig(output_dir / f"noaa_ar_sample_{i}.png", dpi=300)
#                 plt.close()
#
#         except Exception as e:
#             print(f"Error visualizing NOAA AR file {file}: {e}")

import os
import numpy as np
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import sunpy.map
import sunpy.timeseries
from sunpy.timeseries import TimeSeries
from sunpy.net import Fido
from sunpy.net import attrs as a
import astropy.units as u
from pathlib import Path
from datetime import datetime, timedelta
import gc
import time
import contextlib
import warnings


def increase_file_limit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
        print(f"Increased file limit from {soft} to {min(4096, hard)}")
    except (ImportError, ValueError):
        print("Could not increase file limit - not supported on this platform")


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


def create_directories(base_dir="./solar_data"):
    base_dir = Path(base_dir)
    dirs = {
        "sdo_hmi": base_dir / "sdo_hmi",
        "sdo_aia": base_dir / "sdo_aia",
        "goes": base_dir / "goes",
        "soho": base_dir / "soho",
        "processed": base_dir / "processed",
        "visualizations": base_dir / "visualizations"
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def download_hmi_data(start_date, end_date, dirs, sample_cadence='12h', max_files=10):
    print(f"Downloading SDO/HMI magnetogram data from {start_date} to {end_date}...")

    time_range = a.Time(start_date, end_date)

    try:
        result = Fido.search(
            time_range,
            a.Instrument.hmi,
            a.Physobs.los_magnetic_field,
            a.Sample(u.Quantity(sample_cadence)))  # Added closing parenthesis here

        if len(result) == 0:
            print("No results with default query. Trying with specific data series...")
            result = Fido.search(
                time_range,
                a.Instrument.hmi,
                a.Physobs.los_magnetic_field,
                a.Sample(u.Quantity(sample_cadence)),
                a.Series('hmi.M_720s'))
        if len(result) > max_files:
            print(f"Found {len(result)} files, limiting to {max_files}")
            result = result[:max_files]
        else:
            print(f"Found {len(result)} files")
        if len(result) > 0:
            downloaded_files = Fido.fetch(result, path=dirs["sdo_hmi"] / "{file}", max_conn=3, progress=True)
            print(f"Downloaded {len(downloaded_files)} HMI magnetogram files")
            return downloaded_files
        else:
            print("No HMI files found to download")
            return []
    except Exception as e:
        print(f"Error during HMI data download: {e}")
        return []


def download_aia_data(start_date, end_date, dirs, wavelengths=None, sample_cadence='1d', max_files=20):
    if wavelengths is None:
        wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600]
    wavelengths = [int(wl) if isinstance(wl, str) else wl for wl in wavelengths]
    print(f"Downloading SDO/AIA data for wavelengths {wavelengths} from {start_date} to {end_date}...")
    downloaded_files = []
    time_range = a.Time(start_date, end_date)
    for wl in wavelengths:
        try:
            print(f"Searching for AIA {wl}Å data...")
            result = Fido.search(
                time_range,
                a.Instrument.aia,
                a.Wavelength(wl * u.angstrom),
                a.Sample(u.Quantity(sample_cadence)))
            if len(result) > max_files:
                print(f"Found {len(result)} files for {wl}Å, limiting to {max_files}")
                result = result[:max_files]
            else:
                print(f"Found {len(result)} files for {wl}Å")
            if len(result) > 0:
                wl_files = Fido.fetch(result, path=dirs["sdo_aia"] / f"{wl}" / "{file}", max_conn=3, progress=True)
                downloaded_files.extend(wl_files)
                time.sleep(1)
            else:
                print(f"No files found for AIA {wl}Å")
        except Exception as e:
            print(f"Error downloading AIA {wl}Å data: {e}")
            continue
    print(f"Downloaded {len(downloaded_files)} AIA image files")
    return downloaded_files


def download_goes_data(start_date, end_date, dirs):
    print(f"Downloading GOES X-ray flux data from {start_date} to {end_date}...")
    time_range = a.Time(start_date, end_date)
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            result = Fido.search(
                time_range,
                a.Instrument.xrs)
            print(f"Found {len(result)} GOES data files")
            if len(result) > 0:
                # Increased timeout
                downloaded_files = Fido.fetch(result, path=dirs["goes"] / "{file}", max_conn=2, progress=True,
                                              timeout=120)
                print(f"Downloaded {len(downloaded_files)} GOES X-ray files")
                return downloaded_files
            else:
                print("No GOES files found to download")
                return []
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(
                    f"Timeout during GOES data download. Retry {attempt + 1}/{max_retries} in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries reached for GOES data download. Moving on.")
                return []
        except Exception as e:
            print(f"Error during GOES data download: {e}")
            return []


def extract_flare_events(goes_files, min_class='C1.0'):
    flares = []
    for file in goes_files:
        try:
            goes_ts = None
            try:
                goes_ts = TimeSeries(file)
                flux_data = goes_ts.to_dataframe()
                flux_data['smooth_long'] = flux_data['xrsa'].rolling(window=30, center=True).mean()
                flux_data['smooth_short'] = flux_data['xrsb'].rolling(window=30, center=True).mean()
                peaks = []
                for i in range(2, len(flux_data) - 2):
                    if (flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i - 1] and
                            flux_data['smooth_long'].iloc[i] > flux_data['smooth_long'].iloc[i + 1] and
                            flux_data['smooth_long'].iloc[i] > 1e-6):
                        flux_value = flux_data['xrsb'].iloc[i]
                        flare_class = flux_to_class(flux_value)
                        if not is_above_min_class(flare_class, min_class):
                            continue
                        peaks.append({
                            'time': flux_data.index[i],
                            'flux': flux_value,
                            'class': flare_class})
                flares.extend(peaks)
            finally:
                if goes_ts is not None:
                    del goes_ts
                    gc.collect()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    if flares:
        flares_df = pd.DataFrame(flares)
        if not flares_df.empty:
            flares_df.sort_values('time', inplace=True)
    else:
        flares_df = pd.DataFrame(columns=['time', 'flux', 'class'])
    return flares_df


def flux_to_class(flux_value):
    if flux_value < 1e-6:
        return 'A' + str(round(flux_value * 1e7, 1))
    elif flux_value < 1e-5:
        return 'B' + str(round(flux_value * 1e6, 1))
    elif flux_value < 1e-4:
        return 'C' + str(round(flux_value * 1e5, 1))
    elif flux_value < 1e-3:
        return 'M' + str(round(flux_value * 1e4, 1))
    else:
        return 'X' + str(round(flux_value * 1e3, 1))


def is_above_min_class(flare_class, min_class):
    class_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
    flare_letter = flare_class[0]
    flare_number = float(flare_class[1:])
    min_letter = min_class[0]
    min_number = float(min_class[1:])
    if class_levels[flare_letter] > class_levels[min_letter]:
        return True
    elif class_levels[flare_letter] == class_levels[min_letter]:
        return flare_number >= min_number
    else:
        return False


def preprocess_hmi_data(hmi_files, output_dir):
    for i, file in enumerate(hmi_files):
        try:
            hmi_map = None
            try:
                hmi_map = sunpy.map.Map(file)
                data = hmi_map.data
                header = hmi_map.meta
                data = np.nan_to_num(data)
                data_max = np.max(np.abs(data))
                if data_max > 0:
                    data = data / data_max
                np.save(output_dir / f"hmi_processed_{i}.npy", data)
                with open(output_dir / f"hmi_metadata_{i}.txt", 'w') as f:
                    for key, value in header.items():
                        f.write(f"{key}: {value}\n")
            finally:
                if hmi_map is not None:
                    del hmi_map
                    gc.collect()
        except Exception as e:
            print(f"Error processing HMI file {file}: {e}")
            continue


# def download_noaa_ar_data(start_date, end_date, dirs):
#     print(f"Downloading NOAA AR data from {start_date} to {end_date}")
#     # Use the Space Weather Prediction Center API for active region data
#     base_url = "https://services.swpc.noaa.gov/json/solar-region/"
#     alt_base_url = "https://services.swpc.noaa.gov/text/solar-region-summary.txt"
#
#     downloaded_files = []
#     current_dt = datetime.strptime(start_date, '%Y-%m-%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#     session = requests.Session()
#
#     # Configure retry strategy
#     try:
#         from requests.adapters import HTTPAdapter
#         from urllib3.util.retry import Retry
#         retry_strategy = Retry(
#             total=3,
#             backoff_factor=1,
#             status_forcelist=[429, 500, 502, 503, 504],
#         )
#         adapter = HTTPAdapter(max_retries=retry_strategy)
#         session.mount("https://", adapter)
#         session.mount("http://", adapter)
#     except ImportError:
#         print("Requests retry adapter not available, using basic session")
#
#     # Get the latest SRS data as a fallback
#     try:
#         response = session.get(alt_base_url, timeout=30)
#         if response.status_code == 200:
#             srs_file_path = dirs["noaa_ar"] / "solar_region_summary.txt"
#             with open(srs_file_path, 'w') as f:
#                 f.write(response.text)
#             downloaded_files.append(srs_file_path)
#             print("Downloaded latest Solar Region Summary as fallback")
#     except Exception as e:
#         print(f"Failed to download Solar Region Summary: {e}")
#
#     # Try to download JSON data for each day
#     while current_dt <= end_dt:
#         try:
#             date_str = current_dt.strftime('%Y-%m-%d')
#             url = f"{base_url}{date_str}.json"
#             response = session.get(url, timeout=30)
#
#             if response.status_code == 200:
#                 try:
#                     data = response.json()
#                     file_path = dirs["noaa_ar"] / f"noaa_ar_{date_str.replace('-', '')}.json"
#
#                     if isinstance(data, dict) and 'regions' in data and isinstance(data['regions'], list):
#                         with open(file_path, 'w') as f:
#                             json.dump(data, f)
#                         downloaded_files.append(file_path)
#                         print(f"Downloaded NOAA AR data for {date_str}")
#                     else:
#                         print(f"Invalid NOAA AR data format for {date_str}")
#                 except (json.JSONDecodeError, TypeError) as e:
#                     print(f"Error parsing NOAA AR data for {date_str}: {e}")
#             else:
#                 print(f"No NOAA AR data for {date_str}: {response.status_code} status code")
#         except requests.ConnectionError as e:
#             print(f"Connection error downloading NOAA AR data for {date_str}: {e}")
#             time.sleep(5)
#         except Exception as e:
#             print(f"Error downloading NOAA AR data {date_str}: {str(e)[:100]}")
#
#         current_dt += timedelta(days=1)
#
#     session.close()
#     return downloaded_files

# def download_noaa_ar_data(start_date, end_date, dirs):
#     print(f"Downloading NOAA AR data from {start_date} to {end_date}")
#     downloaded_files = []
#
#     # Convert date strings to datetime objects for easier manipulation
#     start_dt = datetime.strptime(start_date, '%Y-%m-%d')
#     end_dt = datetime.strptime(end_date, '%Y-%m-%d')
#
#     # Method 1: NOAA SWPC FTP Archive - SRS Files
#     # SRS (Solar Region Summary) files contain active region data
#     # Format: ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/{month}/SRS/{year}{month}{day}SRS.txt
#
#     session = requests.Session()
#
#     # Add retry capability
#     try:
#         from requests.adapters import HTTPAdapter
#         from urllib3.util.retry import Retry
#         retry_strategy = Retry(
#             total=3,
#             backoff_factor=1,
#             status_forcelist=[429, 500, 502, 503, 504],
#         )
#         adapter = HTTPAdapter(max_retries=retry_strategy)
#         session.mount("https://", adapter)
#         session.mount("http://", adapter)
#         session.mount("ftp://", adapter)
#     except ImportError:
#         print("Requests retry adapter not available, using basic session")
#
#     # Loop through each day in the date range
#     current_dt = start_dt
#     while current_dt <= end_dt:
#         year = current_dt.strftime('%Y')
#         month = current_dt.strftime('%m')
#         day = current_dt.strftime('%d')
#         date_str = current_dt.strftime('%Y%m%d')
#
#         # Try multiple potential sources for the data
#         sources = [
#             # Method 1: NOAA SWPC FTP Archive (HTTP access)
#             f"https://www.swpc.noaa.gov/ftpdir/warehouse/{year}/{month}/SRS/{date_str}SRS.txt",
#             # Method 2: SWPC Legacy FTP server (HTTP access)
#             f"https://web.archive.org/web/https://legacy-www.swpc.noaa.gov/ftpdir/forecasts/SRS/{date_str}SRS.txt",
#             # Method 3: NGDC Archive
#             f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/sunspot-regions/usaf_mwl/usaf_solar-region-reports/{year}/{date_str}SRS.txt"
#         ]
#
#         success = False
#         for source_url in sources:
#             try:
#                 print(f"Trying to download NOAA AR data from: {source_url}")
#                 response = session.get(source_url, timeout=30)
#
#                 if response.status_code == 200:
#                     file_path = dirs["noaa_ar"] / f"noaa_ar_srs_{date_str}.txt"
#                     with open(file_path, 'w') as f:
#                         f.write(response.text)
#
#                     downloaded_files.append(file_path)
#                     print(f"Successfully downloaded NOAA AR data for {date_str}")
#                     success = True
#                     break
#                 else:
#                     print(f"Failed to download from {source_url}: {response.status_code}")
#             except Exception as e:
#                 print(f"Error trying {source_url}: {str(e)[:100]}")
#
#         if not success:
#             # Method 4: Try to get data from NASA HELIO database via SunPy/HEK
#             try:
#                 from sunpy.net import hek
#                 client = hek.HEKClient()
#                 start_time = current_dt.strftime('%Y-%m-%dT00:00:00')
#                 end_time = current_dt.strftime('%Y-%m-%dT23:59:59')
#                 print(f"Trying to query HEK for Active Regions on {date_str}")
#
#                 ar_query = client.search(
#                     hek.attrs.Time(start_time, end_time),
#                     hek.attrs.EventType('AR')
#                 )
#
#                 if len(ar_query) > 0:
#                     file_path = dirs["noaa_ar"] / f"noaa_ar_hek_{date_str}.json"
#                     with open(file_path, 'w') as f:
#                         json.dump(ar_query, f, default=str)
#
#                     downloaded_files.append(file_path)
#                     print(f"Retrieved {len(ar_query)} active regions from HEK for {date_str}")
#                     success = True
#                 else:
#                     print(f"No active region data found in HEK for {date_str}")
#             except Exception as e:
#                 print(f"Error querying HEK: {str(e)[:100]}")
#
#         # Move to next day
#         current_dt += timedelta(days=1)
#         time.sleep(2)  # Be gentle with the servers
#
#     # If we couldn't get data for specific days, try one more source for the entire range
#     if len(downloaded_files) == 0:
#         try:
#             # Method 5: NASA HEK Search for the entire date range
#             from sunpy.net import hek
#             client = hek.HEKClient()
#             start_time = start_dt.strftime('%Y-%m-%dT00:00:00')
#             end_time = end_dt.strftime('%Y-%m-%dT23:59:59')
#             print(f"Trying to query HEK for Active Regions for entire date range {start_date} to {end_date}")
#
#             ar_query = client.search(
#                 hek.attrs.Time(start_time, end_time),
#                 hek.attrs.EventType('AR')
#             )
#
#             if len(ar_query) > 0:
#                 file_path = dirs[
#                                 "noaa_ar"] / f"noaa_ar_hek_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.json"
#                 with open(file_path, 'w') as f:
#                     json.dump(ar_query, f, default=str)
#
#                 downloaded_files.append(file_path)
#                 print(f"Retrieved {len(ar_query)} active regions from HEK for date range")
#         except Exception as e:
#             print(f"Error querying HEK for date range: {str(e)[:100]}")
#
#     session.close()
#     return downloaded_files

def download_soho_data(start_date, end_date, dirs, instruments=None, sample_cadence='1h', max_files=20):
    if instruments is None:
        instruments = ['c2', 'c3']
    print(f"Downloading SOHO data from {start_date} to {end_date}")
    downloaded_files = []
    time_range = a.Time(start_date, end_date)
    for instrument in instruments:
        print(f"Searching SOHO/LASCO {instrument.upper()} data...")
        try:
            result = Fido.search(
                time_range,
                a.Instrument.lasco,
                a.Detector(instrument),
                a.Sample(u.Quantity(sample_cadence)))
            if not result or len(result) == 0:
                print(f"No data found for LASCO {instrument.upper()}")
                continue
            files_to_download = min(max_files, len(result))
            print(f"Found {len(result)} files for LASCO {instrument.upper()}, downloading {files_to_download}")
            batch_size = 5
            for i in range(0, files_to_download, batch_size):
                batch_end = min(i + batch_size, files_to_download)
                print(
                    f"Downloading batch {i // batch_size + 1} of {(files_to_download + batch_size - 1) // batch_size} for LASCO {instrument.upper()}")
                try:
                    batch_result = result[i:batch_end]
                    files = Fido.fetch(batch_result, path=dirs["soho"] / f"lasco_{instrument}" / "{file}", max_conn=3,
                                       progress=True)
                    downloaded_files.extend(files)
                    time.sleep(2)
                    gc.collect()
                except Exception as e:
                    print(f"Error fetching batch {i // batch_size + 1} for LASCO {instrument.upper()}: {e}")
                    continue
        except Exception as e:
            print(f"Error searching/fetching SOHO/LASCO {instrument.upper()} data: {e}")
            continue
        time.sleep(3)
    return downloaded_files


def extract_date_from_filename(filename):
    filename = str(filename)
    try:
        # For SOHO/LASCO FITS files
        if "lasco" in filename.lower() and (filename.endswith('.fts') or filename.endswith('.fits')):
            try:
                # Use sunpy to directly extract date from the file headers
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with sunpy.map.Map(filename, silence_errors=True) as soho_map:
                        return soho_map.date.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                # Fallback to extracting date from directory structure
                try:
                    parent_dir = os.path.basename(os.path.dirname(filename))
                    if parent_dir.isdigit() and len(parent_dir) == 6:
                        # Handle 2-digit years properly
                        year_prefix = parent_dir[:2]
                        year = int(year_prefix)

                        # Determine century based on year value
                        if year < 50:  # Assume 21st century for years 00-49
                            year += 2000
                        else:  # Assume 20th century for years 50-99
                            year += 1900

                        return f"{year}-{parent_dir[2:4]}-{parent_dir[4:6]}"
                except Exception:
                    pass

                # If all else fails, use safe default
                print(f"Could not extract date from {filename}. Using placeholder date.")
                return "2000-01-01T00:00:00"  # Safe placeholder

        # For HMI files
        if ".fits" in filename and "hmi" in filename:
            parts = filename.split('.')
            for part in parts:
                if '_' in part and len(part) > 10:
                    date_parts = part.split('_')
                    if len(date_parts) >= 4 and len(date_parts[0]) == 10:
                        date_str = date_parts[0]
                        time_str = f"{date_parts[1]}:{date_parts[2]}:{date_parts[3]}"
                        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str}"

        # For AIA files
        elif "aia" in filename and "_lev1.fits" in filename:
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if 'T' in part and i > 0 and parts[i - 1].endswith('A'):
                    year = parts[i - 2] if i >= 2 else None
                    month = parts[i - 1][:-1] if i >= 1 else None
                    day = part.split('T')[0]
                    time_parts = part.split('T')[1].split('.')
                    hour, minute = time_parts[0], time_parts[1]
                    if year and month and day:
                        return f"{year}-{month}-{day}T{hour}:{minute}:00"

        # General pattern for dates in filenames
        parts = Path(filename).stem.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            if 'T' in part and len(part) > 10:
                date_part, time_part = part.split('T')
                if len(date_part) == 8:
                    return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T{time_part}"
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")

    # If nothing worked, return None
    return None


def create_visualizations(hmi_files, aia_files, goes_files, soho_files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend('agg')

    # Suppress common SunPy warnings during visualization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="astropy.io.fits.verify")
        warnings.filterwarnings("ignore", module="sunpy.map.mapbase")

        # Process HMI files
        max_files_to_process = min(5, len(hmi_files) if hmi_files else 0)
        for i, file in enumerate(hmi_files[:max_files_to_process]):
            try:
                hmi_map = None
                try:
                    hmi_map = sunpy.map.Map(file)
                    with figure_context(figsize=(10, 8)) as fig:
                        norm = plt.Normalize(-200, 200)
                        hmi_map.plot(cmap='RdBu_r', norm=norm)
                        plt.colorbar(label='Magnetic Field Strength (Gauss)')
                        plt.title(f'SDO/HMI Magnetogram - {hmi_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
                        plt.tight_layout()
                        plt.savefig(output_dir / f"hmi_sample_{i}.png", dpi=300)
                finally:
                    if hmi_map is not None:
                        del hmi_map
                    gc.collect()
            except Exception as e:
                print(f"Error visualizing HMI file {file}: {e}")
                continue
            time.sleep(0.5)

        # Process AIA files
        aia_by_wavelength = {}
        for file in aia_files if aia_files else []:
            try:
                filename = str(file)
                for wl in ['94', '131', '171', '193', '211', '304', '335', '1600']:
                    if wl in filename:
                        if wl not in aia_by_wavelength:
                            aia_by_wavelength[wl] = []
                        aia_by_wavelength[wl].append(file)
                        break
            except Exception as e:
                print(f"Error categorizing AIA file {file}: {e}")
                continue

        for wl, files in aia_by_wavelength.items():
            max_files_per_wl = min(2, len(files))
            for i, file in enumerate(files[:max_files_per_wl]):
                try:
                    aia_map = None
                    try:
                        aia_map = sunpy.map.Map(file)
                        with figure_context(figsize=(10, 8)) as fig:
                            aia_map.plot()
                            plt.colorbar(label='Intensity')
                            plt.title(f'SDO/AIA {wl}Å - {aia_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
                            plt.tight_layout()
                            plt.savefig(output_dir / f"aia_{wl}A_sample_{i}.png", dpi=300)
                    finally:
                        if aia_map is not None:
                            del aia_map
                        gc.collect()
                except Exception as e:
                    print(f"Error visualizing AIA file {file}: {e}")
                    continue
                time.sleep(0.5)

        # Process GOES files
        max_goes_files = min(2, len(goes_files) if goes_files else 0)
        for i, file in enumerate(goes_files[:max_goes_files]):
            try:
                goes_ts = None
                try:
                    goes_ts = TimeSeries(file)
                    with figure_context(figsize=(12, 6)) as fig:
                        goes_ts.plot()
                        plt.title(f'GOES X-ray Flux - {goes_ts.source}')
                        plt.tight_layout()
                        plt.savefig(output_dir / f"goes_sample_{i}.png", dpi=300)
                finally:
                    if goes_ts is not None:
                        del goes_ts
                    gc.collect()
            except Exception as e:
                print(f"Error visualizing GOES file {file}: {e}")
                continue
            time.sleep(0.5)

        # Process SOHO files
        max_soho_files = min(5, len(soho_files) if soho_files else 0)
        for i, file in enumerate(soho_files[:max_soho_files]):
            try:
                if 'c2' in str(file).lower():
                    instrument = 'C2'
                elif 'c3' in str(file).lower():
                    instrument = 'C3'
                else:
                    instrument = 'LASCO'

                soho_map = None
                try:
                    soho_map = sunpy.map.Map(file)
                    with figure_context(figsize=(10, 10)) as fig:
                        soho_map.plot()
                        plt.title(f'SOHO/LASCO {instrument} - {soho_map.date.strftime("%Y-%m-%d %H:%M:%S")}')
                        plt.tight_layout()
                        plt.savefig(output_dir / f"soho_lasco_{instrument.lower()}_sample_{i}.png", dpi=300)
                finally:
                    if soho_map is not None:
                        del soho_map
                    gc.collect()
            except Exception as e:
                print(f"Error visualizing SOHO file {file}: {e}")
                continue
            time.sleep(0.5)



    gc.collect()


def visualize_storage(base_dir):
    print("Generating storage report...")
    try:
        plt.switch_backend('agg')
        base_dir = Path(base_dir)
        dir_sizes = {}
        for dir_name in os.listdir(base_dir):
            dir_path = base_dir / dir_name
            if dir_path.is_dir():
                size = 0
                for path, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = Path(path) / file
                        if file_path.exists():
                            size += file_path.stat().st_size
                dir_sizes[dir_name] = size / (1024 * 1024)
        with figure_context(figsize=(10, 6)) as fig:
            plt.bar(dir_sizes.keys(), dir_sizes.values())
            plt.ylabel('Size (MB)')
            plt.title('Storage Usage by Directory')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(base_dir / 'storage_report.png', dpi=300)
        print(f"Storage report saved to {base_dir / 'storage_report.png'}")
    except Exception as e:
        print(f"Error generating storage report: {e}")


def run_solar_data_pipeline(start_date, end_date, base_dir="./solar_data"):
    try:
        # Setup warning filters
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="astropy.io.fits.verify")
        warnings.filterwarnings("ignore", message=".*EarthLocation.*")

        # Initialize
        print(f"Starting solar data pipeline for period {start_date} to {end_date}")
        increase_file_limit()
        dirs = create_directories(base_dir)

        # Download data from different sources
        hmi_files = download_hmi_data(start_date, end_date, dirs, sample_cadence='12h')
        aia_files = download_aia_data(start_date, end_date, dirs, sample_cadence='1d')
        goes_files = download_goes_data(start_date, end_date, dirs)
        soho_files = download_soho_data(start_date, end_date, dirs)

        # Process data
        if hmi_files:
            print("Preprocessing HMI data...")
            preprocess_hmi_data(hmi_files, dirs["processed"])

        # Extract flare events
        if goes_files:
            print("Extracting flare events...")
            flares_df = extract_flare_events(goes_files, min_class='C1.0')
            flares_csv_path = dirs["processed"] / "flare_events.csv"
            flares_df.to_csv(flares_csv_path)
            print(f"Saved {len(flares_df)} flare events to {flares_csv_path}")

        # Create data catalog
        print("Creating data catalog...")
        catalog = []

        # Add HMI files to catalog
        for file in hmi_files:
            date = extract_date_from_filename(file)
            if date:
                catalog.append({
                    "type": "magnetogram",
                    "instrument": "SDO/HMI",
                    "date": date,
                    "path": str(file)
                })

        # Add AIA files to catalog
        for file in aia_files:
            date = extract_date_from_filename(file)
            if date:
                wavelength = None
                for wl in ['94', '131', '171', '193', '211', '304', '335', '1600']:
                    if wl in str(file):
                        wavelength = wl
                        break

                catalog.append({
                    "type": "euv_image",
                    "instrument": "SDO/AIA",
                    "wavelength": wavelength,
                    "date": date,
                    "path": str(file)
                })

        # Add GOES files to catalog
        for file in goes_files:
            catalog.append({
                "type": "xray_flux",
                "instrument": "GOES",
                "path": str(file)
            })

        # Add SOHO files to catalog
        for file in soho_files:
            date = extract_date_from_filename(file)
            if date:
                instrument = "C2" if "c2" in str(file).lower() else "C3" if "c3" in str(file).lower() else "LASCO"
                catalog.append({
                    "type": "coronagraph",
                    "instrument": f"SOHO/LASCO {instrument}",
                    "date": date,
                    "path": str(file)
                })



        # Save catalog
        catalog_path = Path(base_dir) / "data_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        print(f"Saved data catalog to {catalog_path}")

        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(hmi_files, aia_files, goes_files, soho_files, dirs["visualizations"])

        # Generate storage report
        visualize_storage(base_dir)

        print("Solar data pipeline completed successfully")
        return {
            "hmi_files": len(hmi_files) if hmi_files else 0,
            "aia_files": len(aia_files) if aia_files else 0,
            "goes_files": len(goes_files) if goes_files else 0,
            "soho_files": len(soho_files) if soho_files else 0,
            "catalog_path": str(catalog_path),
            "visualizations_dir": str(dirs["visualizations"])
        }

    except Exception as e:
        print(f"Error in solar data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up
        gc.collect()
        plt.close('all')