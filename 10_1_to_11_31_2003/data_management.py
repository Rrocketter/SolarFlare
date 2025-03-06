# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from datetime import datetime, timedelta
# import json
# import requests
# import re
#
#
# class SolarDataManager:
#     """
#     Utility class to manage solar data download, storage, and processing
#     with improved protection for important training data.
#     """
#
#     def __init__(self, base_dir="./solar_data", max_storage_gb=90, training_period=None):
#         self.base_dir = Path(base_dir)
#         self.max_storage_bytes = max_storage_gb * 1024 ** 3
#         self.data_tracking_file = self.base_dir / "data_inventory.csv"
#
#         # Set training period if provided (format: ('YYYY-MM-DD', 'YYYY-MM-DD'))
#         # Files within this period will be protected from deletion
#         self.training_period = training_period
#
#         # Create directory structure if needed
#         self._setup_directories()
#
#         # Initialize or load data tracking
#         self._init_data_tracking()
#
#         # Keep track of flare dates for importance scoring
#         self.flare_dates = []
#
#     def _setup_directories(self):
#         """Set up the directory structure"""
#         dirs = {
#             "sdo_hmi": self.base_dir / "sdo_hmi",
#             "sdo_aia": self.base_dir / "sdo_aia",
#             "goes": self.base_dir / "goes",
#             "noaa_ar": self.base_dir / "noaa_ar",
#             "soho": self.base_dir / "soho",
#             "processed": self.base_dir / "processed",
#             "visualizations": self.base_dir / "visualizations"
#         }
#
#         for dir_path in dirs.values():
#             dir_path.mkdir(parents=True, exist_ok=True)
#
#         self.dirs = dirs
#
#     def _init_data_tracking(self):
#         """Initialize or load data tracking information"""
#         if self.data_tracking_file.exists():
#             self.data_inventory = pd.read_csv(self.data_tracking_file)
#             # Convert date string to datetime for easier comparison
#             if 'date' in self.data_inventory.columns:
#                 self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])
#         else:
#             self.data_inventory = pd.DataFrame(columns=[
#                 'file_path', 'data_type', 'date', 'size_bytes', 'processed',
#                 'priority', 'importance_score', 'protected'
#             ])
#             self.data_inventory.to_csv(self.data_tracking_file, index=False)
#
#     def _extract_date_from_filename(self, filename):
#         """Extract date from filename using multiple patterns"""
#         patterns = [
#             # YYYYMMDD
#             (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
#             # YYYY-MM-DD variants
#             (r'(\d{4})[-_.](\d{2})[-_.](\d{2})', '%Y%m%d'),
#             # YYYYMMDDTHHMMSS
#             (r'(\d{4})(\d{2})(\d{2})T', '%Y%m%d')
#         ]
#
#         for pattern, fmt in patterns:
#             match = re.search(pattern, filename)
#             if match:
#                 try:
#                     y, m, d = match.groups()[:3]
#                     return datetime(int(y), int(m), int(d))
#                 except (ValueError, IndexError):
#                     continue
#         return None
#
#     def get_storage_usage(self):
#         """Get current storage usage"""
#         total_size = 0
#         for path, dirs, files in os.walk(self.base_dir):
#             for f in files:
#                 fp = os.path.join(path, f)
#                 total_size += os.path.getsize(fp)
#
#         return total_size
#
#     def check_storage_available(self, required_bytes):
#         """Check if there's enough storage available"""
#         current_usage = self.get_storage_usage()
#         available_bytes = self.max_storage_bytes - current_usage
#
#         return available_bytes >= required_bytes
#
#     def update_importance_scores(self):
#         """
#         Update importance scores based on:
#         1. Whether the data is in the training period
#         2. Whether the data corresponds to flare events
#         3. The importance of the data type itself
#         """
#         if self.data_inventory.empty:
#             return
#
#         # Initialize importance score column if it doesn't exist
#         if 'importance_score' not in self.data_inventory.columns:
#             self.data_inventory['importance_score'] = 0.0
#
#         # Initialize protected column if it doesn't exist
#         if 'protected' not in self.data_inventory.columns:
#             self.data_inventory['protected'] = False
#
#         # Convert dates to datetime for comparison if needed
#         if not pd.api.types.is_datetime64_any_dtype(self.data_inventory['date']):
#             self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])
#
#         # Base importance from priority (scale to 0-10)
#         self.data_inventory['importance_score'] = self.data_inventory['priority'].astype(float)
#
#         # Protection for training period data
#         if self.training_period:
#             train_start = pd.to_datetime(self.training_period[0])
#             train_end = pd.to_datetime(self.training_period[1])
#
#             in_training_period = (self.data_inventory['date'] >= train_start) & \
#                                  (self.data_inventory['date'] <= train_end)
#
#             # Mark files in training period as protected
#             self.data_inventory.loc[in_training_period, 'protected'] = True
#
#             # Increase importance score for training period data
#             self.data_inventory.loc[in_training_period, 'importance_score'] += 5.0
#
#         # Increase importance for flare events
#         for flare_date in self.flare_dates:
#             # Find files within 6 hours of flare
#             flare_dt = pd.to_datetime(flare_date)
#             time_diff = abs(self.data_inventory['date'] - flare_dt)
#             near_flare = time_diff <= pd.Timedelta(hours=6)
#
#             # Boost importance for data near flares
#             self.data_inventory.loc[near_flare, 'importance_score'] += 3.0
#
#             # Protect critical data types during flares
#             critical_during_flare = near_flare & self.data_inventory['data_type'].isin(['hmi', 'goes', 'aia'])
#             self.data_inventory.loc[critical_during_flare, 'protected'] = True
#
#         # Ensure processed files have slightly higher importance
#         self.data_inventory.loc[self.data_inventory['processed'], 'importance_score'] += 1.0
#
#         # Cap the importance score at 10
#         self.data_inventory['importance_score'] = self.data_inventory['importance_score'].clip(0, 10)
#
#         # Save the updated inventory
#         self.data_inventory.to_csv(self.data_tracking_file, index=False)
#
#     def make_space(self, required_bytes):
#         """
#         Make space for new data while preserving important files for training.
#         Files marked as protected will not be deleted.
#         """
#         if self.check_storage_available(required_bytes):
#             return True
#
#         # Update importance scores before deletion decisions
#         self.update_importance_scores()
#
#         # Sort by importance (low first) and protection status
#         to_remove = self.data_inventory.sort_values(
#             by=['protected', 'importance_score', 'priority'],
#             ascending=[True, True, True]
#         )
#
#         bytes_freed = 0
#         files_removed = []
#
#         for idx, row in to_remove.iterrows():
#             # Skip if it's a protected file
#             if row['protected']:
#                 continue
#
#             file_path = row['file_path']
#
#             # Check if file exists
#             if os.path.exists(file_path):
#                 file_size = os.path.getsize(file_path)
#                 os.remove(file_path)
#                 bytes_freed += file_size
#                 files_removed.append(idx)
#                 print(f"Removed {file_path} to free {file_size / 1024 ** 2:.2f} MB of space")
#             else:
#                 # File doesn't exist, just mark for removal from inventory
#                 files_removed.append(idx)
#
#             # Check if we've freed enough space
#             if bytes_freed >= required_bytes:
#                 break
#
#         # Update inventory
#         self.data_inventory = self.data_inventory.drop(files_removed)
#         self.data_inventory.to_csv(self.data_tracking_file, index=False)
#
#         # Check if we managed to free enough space
#         if bytes_freed < required_bytes:
#             print(
#                 f"WARNING: Could only free {bytes_freed / 1024 ** 2:.2f} MB of {required_bytes / 1024 ** 2:.2f} MB needed. Some files may be protected.")
#
#         return bytes_freed >= required_bytes
#
#     def register_files(self, files, data_type, priority=5, protected=False):
#         """
#         Register downloaded files in the inventory.
#         Parameters:
#             files (list): List of file paths
#             data_type (str): Type of data ('hmi', 'aia', etc.)
#             priority (int): Base priority level (1-10)
#             protected (bool): Whether the file should be protected from deletion
#         """
#         new_entries = []
#
#         for file_path in files:
#             # Try to extract date from filename for various data types
#             try:
#                 # Inside the file registration loop:
#                 file_path_str = str(file_path)
#                 date = self._extract_date_from_filename(os.path.basename(file_path_str))
#                 if date is None:
#                     print(f"Warning: Could not extract date from {file_path}")
#                     date = datetime.now()
#
#                 # Different parsing for different data types
#                 if data_type == 'hmi' or data_type == 'aia':
#                     # SDO data typically has a date format in the filename
#                     if '_' in file_path_str:
#                         parts = os.path.basename(file_path_str).split('_')
#                         for part in parts:
#                             if part.startswith('20') and len(part) >= 8:  # 20YY... format
#                                 date_str = part[:8]  # Extract YYYYMMDD
#                                 date = datetime.strptime(date_str, '%Y%m%d')
#                                 break
#                         else:
#                             raise ValueError("Date not found in filename parts after splitting by underscores")
#                     else:
#                         # Try a different approach for SDO format
#                         date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_path_str)
#                         if date_match:
#                             year, month, day = date_match.groups()
#                             date = datetime(int(year), int(month), int(day))
#                         else:
#                             raise ValueError("Date not found in filename")
#
#                 elif data_type == 'goes':
#                     # GOES files often contain the date in format g??_YYYYMMDD
#                     date_match = re.search(r'g\d+_(\d{8})', os.path.basename(file_path_str))
#                     if date_match:
#                         date_str = date_match.group(1)
#                         date = datetime.strptime(date_str, '%Y%m%d')
#                     else:
#                         raise ValueError("Date not found in GOES filename")
#
#                 elif data_type == 'noaa_ar':
#                     # Extract date from NOAA AR files (typically noaa_ar_YYYYMMDD.json)
#                     date_match = re.search(r'noaa_ar_(\d{8})\.json', os.path.basename(file_path_str))
#                     if date_match:
#                         date_str = date_match.group(1)
#                         date = datetime.strptime(date_str, '%Y%m%d')
#                     else:
#                         raise ValueError("Date not found in NOAA AR filename")
#
#                 elif data_type == 'soho':
#                     # SOHO/LASCO data
#                     if 'lasco' in file_path_str.lower():
#                         date_match = re.search(r'(\d{8})_(\d{6})', os.path.basename(file_path_str))
#                         if date_match:
#                             date_str, time_str = date_match.groups()
#                             date = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
#                         else:
#                             raise ValueError("Date not found in SOHO filename")
#                     else:
#                         raise ValueError("Unknown SOHO file format")
#                 else:
#                     # Default fallback
#                     date = datetime.now()
#
#             except Exception as e:
#                 # If date extraction fails, use current time as fallback
#                 print(f"Warning: Could not extract date from {file_path}: {e}")
#                 date = datetime.now()
#
#             # Get file size
#             if os.path.exists(file_path):
#                 size_bytes = os.path.getsize(file_path)
#             else:
#                 size_bytes = 0
#                 print(f"Warning: File not found at {file_path}")
#
#             # Create entry
#             entry = {
#                 'file_path': str(file_path),
#                 'data_type': data_type,
#                 'date': date,
#                 'size_bytes': size_bytes,
#                 'processed': False,
#                 'priority': priority,
#                 'importance_score': float(priority),  # Initialize with priority
#                 'protected': protected
#             }
#
#             new_entries.append(entry)
#
#             # Convert to DataFrame
#             new_df = pd.DataFrame(new_entries)
#
#             # If empty, just return
#             if len(new_entries) == 0:
#                 return
#
#             # Convert date column to datetime for consistency
#             new_df['date'] = pd.to_datetime(new_df['date'])
#
#             # Add to inventory
#             self.data_inventory = pd.concat([
#                 self.data_inventory,
#                 new_df
#             ], ignore_index=True)
#
#             # Update importance scores
#             self.update_importance_scores()
#
#             # Save updated inventory
#             self.data_inventory.to_csv(self.data_tracking_file, index=False)
#
#             print(f"Registered {len(new_entries)} files of type {data_type}")
#
#     def mark_as_processed(self, files):
#         """Mark files as processed"""
#         for file_path in files:
#             mask = self.data_inventory['file_path'] == str(file_path)
#             self.data_inventory.loc[mask, 'processed'] = True
#
#         # Update importance scores
#         self.update_importance_scores()
#
#         # Save updated inventory
#         self.data_inventory.to_csv(self.data_tracking_file, index=False)
#
#     def get_storage_report(self):
#         """Get detailed storage usage report"""
#         total_size = self.get_storage_usage()
#
#         # If inventory is empty, return basic report
#         if self.data_inventory.empty:
#             return {
#                 'total_gb': total_size / 1024 ** 3,
#                 'used_percent': total_size / self.max_storage_bytes * 100,
#                 'available_gb': (self.max_storage_bytes - total_size) / 1024 ** 3,
#                 'by_type_gb': pd.Series(),
#                 'by_type_percent': pd.Series()
#             }
#
#         # Group by data type
#         by_type = self.data_inventory.groupby('data_type')['size_bytes'].sum()
#
#         # Calculate percentages
#         total_accounted = by_type.sum()
#         if total_accounted > 0:
#             percentages = by_type / total_accounted * 100
#         else:
#             percentages = by_type * 0
#
#         # Include protected data stats
#         protected_size = self.data_inventory[self.data_inventory['protected']]['size_bytes'].sum()
#         protected_percent = (protected_size / total_accounted * 100) if total_accounted > 0 else 0
#
#         report = {
#             'total_gb': total_size / 1024 ** 3,
#             'used_percent': total_size / self.max_storage_bytes * 100,
#             'available_gb': (self.max_storage_bytes - total_size) / 1024 ** 3,
#             'by_type_gb': by_type / 1024 ** 3,
#             'by_type_percent': percentages,
#             'protected_gb': protected_size / 1024 ** 3,
#             'protected_percent': protected_percent
#         }
#
#         return report
#
#     def visualize_storage(self, save_path=None):
#         """Create storage usage visualization"""
#         report = self.get_storage_report()
#
#         # Create figure and subplots
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
#
#         # Plot 1: Overall usage
#         labels = ['Used', 'Available']
#         sizes = [report['used_percent'], 100 - report['used_percent']]
#         explode = (0.1, 0)
#         colors = ['#ff9999', '#66b3ff']
#
#         ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
#                 shadow=True, startangle=90)
#         ax1.axis('equal')
#         ax1.set_title('Storage Usage')
#
#         # Plot 2: Usage by data type
#         if not report['by_type_gb'].empty:
#             report['by_type_gb'].plot(kind='bar', ax=ax2, color='skyblue')
#             ax2.set_ylabel('GB')
#             ax2.set_title('Storage by Data Type')
#
#             # Add percentage labels
#             for i, v in enumerate(report['by_type_gb']):
#                 ax2.text(i, v + 0.1, f"{report['by_type_percent'].iloc[i]:.1f}%",
#                          ha='center', va='bottom')
#
#         # Plot 3: Protected vs Unprotected
#         if 'protected_gb' in report:
#             protected_gb = report['protected_gb']
#             total_gb = report['total_gb']
#             unprotected_gb = total_gb - protected_gb
#
#             labels = ['Protected', 'Unprotected']
#             sizes = [protected_gb, unprotected_gb]
#             colors = ['#aaffaa', '#ffaaaa']
#
#             ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
#                     shadow=True, startangle=90)
#             ax3.axis('equal')
#             ax3.set_title('Protected vs Unprotected Data')
#
#         plt.tight_layout()
#
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()
#
#     def get_date_coverage_report(self):
#         """Get report on date coverage of data"""
#         if self.data_inventory.empty:
#             return pd.DataFrame()
#
#         # Ensure date is datetime type
#         self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])
#
#         # Get date range
#         min_date = self.data_inventory['date'].min()
#         max_date = self.data_inventory['date'].max()
#
#         # Create date range
#         date_range = pd.date_range(min_date, max_date)
#
#         # Create a cross-tabulation of dates and data types
#         coverage = pd.crosstab(
#             self.data_inventory['date'].dt.date,
#             self.data_inventory['data_type']
#         )
#
#         # Reindex to include all dates
#         coverage = coverage.reindex(date_range.date, fill_value=0)
#
#         return coverage
#
#     def visualize_date_coverage(self, save_path=None):
#         """Create date coverage visualization"""
#         coverage = self.get_date_coverage_report()
#
#         if coverage.empty:
#             print("No data available for date coverage visualization")
#             return
#
#         plt.figure(figsize=(15, 8))
#
#         # Plot coverage as a heatmap
#         plt.imshow(coverage.T, aspect='auto', cmap='viridis')
#
#         # Configure axes
#         plt.yticks(range(len(coverage.columns)), coverage.columns)
#
#         # Only show some date ticks to avoid overcrowding
#         date_ticks = range(0, len(coverage), max(1, len(coverage) // 15))
#         date_labels = [coverage.index[i].strftime('%Y-%m-%d') for i in date_ticks]
#         plt.xticks(date_ticks, date_labels, rotation=45, ha='right')
#
#         # Mark flare events with red dots if available
#         if self.flare_dates:
#             for flare_date in self.flare_dates:
#                 try:
#                     flare_dt = pd.to_datetime(flare_date).date()
#                     if flare_dt in coverage.index:
#                         flare_idx = coverage.index.get_loc(flare_dt)
#                         plt.axvline(x=flare_idx, color='red', alpha=0.3, linestyle='--')
#                 except Exception as e:
#                     print(f"Error marking flare date {flare_date}: {e}")
#
#         plt.colorbar(label='Number of files')
#         plt.title('Dataset Coverage by Date and Type')
#
#         # Mark training period if set
#         if self.training_period:
#             try:
#                 train_start = pd.to_datetime(self.training_period[0]).date()
#                 train_end = pd.to_datetime(self.training_period[1]).date()
#
#                 if train_start in coverage.index and train_end in coverage.index:
#                     start_idx = coverage.index.get_loc(train_start)
#                     end_idx = coverage.index.get_loc(train_end)
#                     plt.axvspan(start_idx - 0.5, end_idx + 0.5, color='green', alpha=0.2, label='Training Period')
#                     plt.legend(loc='upper right')
#             except Exception as e:
#                 print(f"Error marking training period: {e}")
#
#         plt.tight_layout()
#
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()
#
#     def estimate_download_size(self, date_range, data_types, cadence='12h'):
#         """
#         Estimate download size for multiple data types
#
#         Parameters:
#         -----------
#         date_range : tuple
#             (start_date, end_date) in 'YYYY-MM-DD' format
#         data_types : list
#             List of data types to include ('hmi', 'aia', 'goes', 'noaa_ar', 'soho')
#         cadence : str
#             Data cadence (e.g., '12h', '1d')
#
#         Returns:
#         --------
#         float : Estimated size in GB
#         """
#         # Convert dates to datetime
#         start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
#         end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
#
#         # Calculate number of days
#         days = (end_date - start_date).days + 1
#
#         # Updated size estimates per file in MB
#         sizes = {
#             'hmi': 15,  # HMI magnetograms
#             'aia': 8,  # AIA images
#             'goes': 0.5,  # GOES X-ray data
#             'noaa_ar': 0.1,  # NOAA AR data (JSON format)
#             'soho': 5  # SOHO LASCO images
#         }
#
#         # Calculate cadence in hours
#         if cadence.endswith('h'):
#             hours_per_sample = float(cadence[:-1])
#         elif cadence.endswith('d'):
#             hours_per_sample = float(cadence[:-1]) * 24
#         else:
#             hours_per_sample = 12  # Default
#
#         # Calculate number of samples per day
#         samples_per_day = 24 / hours_per_sample
#
#         # Calculate total size
#         total_mb = 0
#         for data_type in data_types:
#             if data_type == 'aia':
#                 # AIA has multiple wavelengths
#                 wavelengths = 8  # Default number of wavelengths
#                 total_mb += days * samples_per_day * sizes[data_type] * wavelengths
#             elif data_type == 'noaa_ar':
#                 # NOAA AR is daily regardless of cadence
#                 total_mb += days * sizes[data_type]
#             elif data_type == 'soho':
#                 # SOHO typically has 2 instruments (C2, C3)
#                 instruments = 2
#                 total_mb += days * samples_per_day * sizes[data_type] * instruments
#             else:
#                 total_mb += days * samples_per_day * sizes[data_type]
#
#         # Convert to GB
#         total_gb = total_mb / 1024
#
#         return total_gb
#
#     def download_incremental_data(self, download_func, start_date, end_date, step_days=30, **kwargs):
#         """
#         Download data incrementally to avoid storage issues
#
#         Parameters:
#         -----------
#         download_func : function
#             Function to download data for a specific time range
#         start_date : str
#             Start date in 'YYYY-MM-DD' format
#         end_date : str
#             End date in 'YYYY-MM-DD' format
#         step_days : int
#             Number of days to download in each step
#         **kwargs
#             Additional arguments to pass to download_func
#
#         Returns:
#         --------
#         list : All downloaded files
#         """
#         # Convert dates to datetime
#         current_date = datetime.strptime(start_date, '%Y-%m-%d')
#         final_date = datetime.strptime(end_date, '%Y-%m-%d')
#
#         all_files = []
#
#         # Get data_type from kwargs
#         data_type = kwargs.get('data_type', 'unknown')
#
#         # Set protection status based on training period
#         protected = False
#         if self.training_period:
#             train_start = datetime.strptime(self.training_period[0], '%Y-%m-%d')
#             train_end = datetime.strptime(self.training_period[1], '%Y-%m-%d')
#
#             # If any part of the requested period overlaps with training period, protect the files
#             if not (current_date > train_end or final_date < train_start):
#                 protected = True
#                 print(f"Files will be protected as they overlap with training period")
#
#         # Download data in chunks
#         while current_date <= final_date:
#             # Calculate end date for this chunk
#             chunk_end = current_date + timedelta(days=step_days)
#             if chunk_end > final_date:
#                 chunk_end = final_date
#
#             print(
#                 f"Downloading {data_type} data from {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
#
#             # Estimate size
#             est_size_gb = self.estimate_download_size(
#                 (current_date.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')),
#                 [data_type],
#                 kwargs.get('sample_cadence', '12h')
#             )
#             print(f"Estimated size: {est_size_gb:.2f} GB")
#
#             # Check if we have enough space
#             if not self.check_storage_available(est_size_gb * 1024 ** 3):
#                 # Try to make space if files aren't protected
#                 if not protected and not self.make_space(est_size_gb * 1024 ** 3):
#                     print("WARNING: Could not free up enough space. Skipping this chunk.")
#                     current_date = chunk_end + timedelta(days=1)
#                     continue
#
#             # Download this chunk
#             try:
#                 # Prepare arguments for the download function
#                 download_args = kwargs.copy()
#                 download_args['start_date'] = current_date.strftime('%Y-%m-%d')
#                 download_args['end_date'] = chunk_end.strftime('%Y-%m-%d')
#
#                 # Remove data_type from args as it's used by register_files, not the download function
#                 if 'data_type' in download_args:
#                     del download_args['data_type']
#
#                 download_args.pop('priority', None)
#
#                 # Call download function with appropriate arguments
#                 files = download_func(**download_args)
#
#                 # Register files
#                 self.register_files(
#                     files,
#                     data_type,
#                     kwargs.get('priority', 5),
#                     protected=protected
#                 )
#
#                 all_files.extend(files)
#
#             except Exception as e:
#                 print(f"Error downloading chunk: {e}")
#
#             # Move to next chunk
#             current_date = chunk_end + timedelta(days=1)
#
#         return all_files
#
#     def import_flare_events(self, flare_file_path=None):
#         """
#         Import flare events from a CSV file or look in processed directory
#         This helps in determining which data periods are important
#         """
#         if flare_file_path is None:
#             flare_file_path = self.dirs['processed'] / 'flare_events.csv'
#
#         try:
#             if os.path.exists(flare_file_path):
#                 flares = pd.read_csv(flare_file_path)
#                 if 'time' in flares.columns:
#                     flares['time'] = pd.to_datetime(flares['time'])
#                     self.flare_dates = flares['time'].tolist()
#                     print(f"Imported {len(self.flare_dates)} flare events")
#
#                     # Update importance scores based on flare events
#                     self.update_importance_scores()
#                     return True
#             else:
#                 print(f"Flare file not found at {flare_file_path}")
#         except Exception as e:
#             print(f"Error importing flare events: {e}")
#
#         return False
#
#     def set_training_period(self, start_date, end_date):
#         """
#         Set the training period for the model.
#         Files within this period will be protected from deletion.
#
#         Parameters:
#         -----------
#         start_date : str
#             Start date in 'YYYY-MM-DD' format
#         end_date : str
#             End date in 'YYYY-MM-DD' format
#         """
#         self.training_period = (start_date, end_date)
#         print(f"Training period set to {start_date} to {end_date}")
#
#         # Update protection status for files in this period
#         self.update_importance_scores()
#
#         return True


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
import re


class SolarDataManager:

    def __init__(self, base_dir="./solar_data", max_storage_gb=90, training_period=None):
        self.base_dir = Path(base_dir)
        self.max_storage_bytes = max_storage_gb * 1024 ** 3
        self.data_tracking_file = self.base_dir / "data_inventory.csv"
        self.training_period = training_period
        self._setup_directories()
        self._init_data_tracking()
        self.flare_dates = []

    def _setup_directories(self):
        dirs = {
            "sdo_hmi": self.base_dir / "sdo_hmi",
            "sdo_aia": self.base_dir / "sdo_aia",
            "goes": self.base_dir / "goes",
            "soho": self.base_dir / "soho",
            "processed": self.base_dir / "processed",
            "visualizations": self.base_dir / "visualizations"
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        self.dirs = dirs

    def _init_data_tracking(self):
        if self.data_tracking_file.exists():
            self.data_inventory = pd.read_csv(self.data_tracking_file)
            if 'date' in self.data_inventory.columns:
                self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])
        else:
            self.data_inventory = pd.DataFrame(columns=[
                'file_path', 'data_type', 'date', 'size_bytes', 'processed',
                'priority', 'importance_score', 'protected'
            ])
            self.data_inventory.to_csv(self.data_tracking_file, index=False)

    def _extract_date_from_filename(self, filename):
        # Add more robust pattern matching for different filename formats
        patterns = [
            # HMI format: hmi.m_45s.2017.09.05_18_01_30_TAI.magnetogram.fits
            (r'(\d{4})\.(\d{2})\.(\d{2})_', '%Y.%m.%d'),

            # AIA format: aia.lev1.1600A_2017_09_05T12_00_14.13Z.image_lev1.fits
            (r'(\d{4})_(\d{2})_(\d{2})T', '%Y_%m_%d'),

            # General formats
            (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
            (r'(\d{4})[-_.](\d{2})[-_.](\d{2})', '%Y%m%d'),
            (r'(\d{4})(\d{2})(\d{2})T', '%Y%m%d')
        ]

        filename_str = str(filename)

        for pattern, fmt in patterns:
            match = re.search(pattern, filename_str)
            if match:
                try:
                    if fmt == '%Y%m%d':
                        y, m, d = match.groups()[:3]
                        return datetime(int(y), int(m), int(d))
                    elif fmt == '%Y.%m.%d':
                        y, m, d = match.groups()[:3]
                        return datetime(int(y), int(m), int(d))
                    elif fmt == '%Y_%m_%d':
                        y, m, d = match.groups()[:3]
                        return datetime(int(y), int(m), int(d))
                except (ValueError, IndexError):
                    continue

        # If all patterns fail, try to find any date-like strings
        date_match = re.search(r'20\d{2}[._-]?\d{2}[._-]?\d{2}', filename_str)
        if date_match:
            date_str = date_match.group(0).replace('.', '').replace('-', '').replace('_', '')
            try:
                return datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                pass

        return None

    def get_storage_usage(self):
        total_size = 0
        for path, dirs, files in os.walk(self.base_dir):
            for f in files:
                fp = os.path.join(path, f)
                total_size += os.path.getsize(fp)

        return total_size

    def check_storage_available(self, required_bytes):
        current_usage = self.get_storage_usage()
        available_bytes = self.max_storage_bytes - current_usage

        return available_bytes >= required_bytes

    def update_importance_scores(self):
        if self.data_inventory.empty:
            return

        if 'importance_score' not in self.data_inventory.columns:
            self.data_inventory['importance_score'] = 0.0

        if 'protected' not in self.data_inventory.columns:
            self.data_inventory['protected'] = False

        if not pd.api.types.is_datetime64_any_dtype(self.data_inventory['date']):
            self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])

        self.data_inventory['importance_score'] = self.data_inventory['priority'].astype(float)

        if self.training_period:
            train_start = pd.to_datetime(self.training_period[0])
            train_end = pd.to_datetime(self.training_period[1])

            in_training_period = (self.data_inventory['date'] >= train_start) & \
                                 (self.data_inventory['date'] <= train_end)

            self.data_inventory.loc[in_training_period, 'protected'] = True
            self.data_inventory.loc[in_training_period, 'importance_score'] += 5.0

        for flare_date in self.flare_dates:
            flare_dt = pd.to_datetime(flare_date)
            time_diff = abs(self.data_inventory['date'] - flare_dt)
            near_flare = time_diff <= pd.Timedelta(hours=6)

            self.data_inventory.loc[near_flare, 'importance_score'] += 3.0

            critical_during_flare = near_flare & self.data_inventory['data_type'].isin(['hmi', 'goes', 'aia'])
            self.data_inventory.loc[critical_during_flare, 'protected'] = True

        self.data_inventory.loc[self.data_inventory['processed'], 'importance_score'] += 1.0
        self.data_inventory['importance_score'] = self.data_inventory['importance_score'].clip(0, 10)
        self.data_inventory.to_csv(self.data_tracking_file, index=False)

    def make_space(self, required_bytes):
        if self.check_storage_available(required_bytes):
            return True

        self.update_importance_scores()

        to_remove = self.data_inventory.sort_values(
            by=['protected', 'importance_score', 'priority'],
            ascending=[True, True, True]
        )

        bytes_freed = 0
        files_removed = []

        for idx, row in to_remove.iterrows():
            if row['protected']:
                continue

            file_path = row['file_path']

            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                bytes_freed += file_size
                files_removed.append(idx)
                print(f"Removed {file_path} to free {file_size / 1024 ** 2:.2f} MB of space")
            else:
                files_removed.append(idx)

            if bytes_freed >= required_bytes:
                break

        self.data_inventory = self.data_inventory.drop(files_removed)
        self.data_inventory.to_csv(self.data_tracking_file, index=False)

        if bytes_freed < required_bytes:
            print(
                f"WARNING: Could only free {bytes_freed / 1024 ** 2:.2f} MB of {required_bytes / 1024 ** 2:.2f} MB needed. Some files may be protected.")

        return bytes_freed >= required_bytes

    def register_files(self, files, data_type, priority=5, protected=False):
        new_entries = []

        for file_path in files:
            try:
                file_path_str = str(file_path)
                date = self._extract_date_from_filename(os.path.basename(file_path_str))

                # If the default extraction method fails, try more specific parsing based on data type
                if date is None:
                    if data_type == 'hmi':
                        # For HMI files like: hmi.m_45s.2017.09.05_18_01_30_TAI.magnetogram.fits
                        date_match = re.search(r'(\d{4})\.(\d{2})\.(\d{2})_', file_path_str)
                        if date_match:
                            y, m, d = date_match.groups()
                            date = datetime(int(y), int(m), int(d))
                    elif data_type == 'aia':
                        # For AIA files like: aia.lev1.1600A_2017_09_05T12_00_14.13Z.image_lev1.fits
                        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})T', file_path_str)
                        if date_match:
                            y, m, d = date_match.groups()
                            date = datetime(int(y), int(m), int(d))
                    elif data_type == 'goes':
                        date_match = re.search(r'_d(\d{8})_', os.path.basename(file_path_str))
                        if date_match:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y%m%d')
                    elif data_type == 'soho':
                        # Try to get date from SOHO/LASCO filename patterns
                        if 'lasco' in file_path_str.lower():
                            date_match = re.search(r'(\d{8})_(\d{6})', os.path.basename(file_path_str))
                            if date_match:
                                date_str, time_str = date_match.groups()
                                date = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')

                # If still no date found, use current date as fallback
                if date is None:
                    print(f"Warning: Could not extract date from {file_path}. Using current date.")
                    date = datetime.now()

            except Exception as e:
                print(f"Warning: Could not extract date from {file_path}: {e}")
                date = datetime.now()

            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
            else:
                size_bytes = 0
                print(f"Warning: File not found at {file_path}")

            entry = {
                'file_path': str(file_path),
                'data_type': data_type,
                'date': date,
                'size_bytes': size_bytes,
                'processed': False,
                'priority': priority,
                'importance_score': float(priority),
                'protected': protected
            }

            new_entries.append(entry)

        if len(new_entries) == 0:
            return

        # Create a new DataFrame with the proper dtypes first
        new_df = pd.DataFrame(new_entries)
        new_df['date'] = pd.to_datetime(new_df['date'])

        # Create a copy of the current inventory with the same schema
        if self.data_inventory.empty:
            # If empty, just use the new dataframe
            self.data_inventory = new_df
        else:
            # Fix the FutureWarning by ensuring both DataFrames have same columns before concat
            # Get all columns from both dataframes
            all_columns = set(list(self.data_inventory.columns) + list(new_df.columns))

            # Add missing columns to each dataframe
            for col in all_columns:
                if col not in self.data_inventory.columns:
                    self.data_inventory[col] = None
                if col not in new_df.columns:
                    new_df[col] = None

            # Concatenate with columns aligned
            self.data_inventory = pd.concat([
                self.data_inventory,
                new_df
            ], ignore_index=True)

        self.update_importance_scores()
        self.data_inventory.to_csv(self.data_tracking_file, index=False)

        print(f"Registered {len(new_entries)} files of type {data_type}")

    def mark_as_processed(self, files):
        for file_path in files:
            mask = self.data_inventory['file_path'] == str(file_path)
            self.data_inventory.loc[mask, 'processed'] = True

        self.update_importance_scores()

        self.data_inventory.to_csv(self.data_tracking_file, index=False)

    def get_storage_report(self):
        total_size = self.get_storage_usage()

        if self.data_inventory.empty:
            return {
                'total_gb': total_size / 1024 ** 3,
                'used_percent': total_size / self.max_storage_bytes * 100,
                'available_gb': (self.max_storage_bytes - total_size) / 1024 ** 3,
                'by_type_gb': pd.Series(),
                'by_type_percent': pd.Series()
            }

        by_type = self.data_inventory.groupby('data_type')['size_bytes'].sum()

        total_accounted = by_type.sum()
        if total_accounted > 0:
            percentages = by_type / total_accounted * 100
        else:
            percentages = by_type * 0

        protected_size = self.data_inventory[self.data_inventory['protected']]['size_bytes'].sum()
        protected_percent = (protected_size / total_accounted * 100) if total_accounted > 0 else 0

        report = {
            'total_gb': total_size / 1024 ** 3,
            'used_percent': total_size / self.max_storage_bytes * 100,
            'available_gb': (self.max_storage_bytes - total_size) / 1024 ** 3,
            'by_type_gb': by_type / 1024 ** 3,
            'by_type_percent': percentages,
            'protected_gb': protected_size / 1024 ** 3,
            'protected_percent': protected_percent
        }

        return report

    def visualize_storage(self, save_path=None):
        report = self.get_storage_report()

        # Close any existing figures to prevent memory/file descriptor leaks
        plt.close('all')

        # Define backend explicitly to avoid MacOSX backend issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        labels = ['Used', 'Available']
        sizes = [report['used_percent'], 100 - report['used_percent']]
        explode = (0.1, 0)
        colors = ['#ff9999', '#66b3ff']

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title('Storage Usage')

        if not report['by_type_gb'].empty:
            report['by_type_gb'].plot(kind='bar', ax=ax2, color='skyblue')
            ax2.set_ylabel('GB')
            ax2.set_title('Storage by Data Type')

            for i, v in enumerate(report['by_type_gb']):
                ax2.text(i, v + 0.1, f"{report['by_type_percent'].iloc[i]:.1f}%",
                         ha='center', va='bottom')

        if 'protected_gb' in report:
            protected_gb = report['protected_gb']
            total_gb = report['total_gb']
            unprotected_gb = max(0, total_gb - protected_gb)  # Ensure non-negative

            labels = ['Protected', 'Unprotected']
            sizes = [protected_gb, unprotected_gb]
            colors = ['#aaffaa', '#ffaaaa']

            # Only plot if there's data to show
            if sum(sizes) > 0:
                ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax3.axis('equal')
                ax3.set_title('Protected vs Unprotected Data')

        plt.tight_layout()

        if save_path:
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Explicitly close figure
        else:
            plt.savefig(self.base_dir / "storage_report.png", dpi=300, bbox_inches='tight')
            plt.close(fig)  # Always close to prevent resource leaks

    def get_date_coverage_report(self):
        if self.data_inventory.empty:
            return pd.DataFrame()

        self.data_inventory['date'] = pd.to_datetime(self.data_inventory['date'])

        min_date = self.data_inventory['date'].min()
        max_date = self.data_inventory['date'].max()

        date_range = pd.date_range(min_date, max_date)

        coverage = pd.crosstab(
            self.data_inventory['date'].dt.date,
            self.data_inventory['data_type']
        )

        coverage = coverage.reindex(date_range.date, fill_value=0)

        return coverage

    def visualize_date_coverage(self, save_path=None):
        # Close any existing figures to prevent resource leaks
        plt.close('all')

        # Define backend explicitly to avoid MacOSX backend issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        coverage = self.get_date_coverage_report()

        if coverage.empty:
            print("No data available for date coverage visualization")
            return

        fig = plt.figure(figsize=(15, 8))

        plt.imshow(coverage.T, aspect='auto', cmap='viridis')

        plt.yticks(range(len(coverage.columns)), coverage.columns)

        date_ticks = range(0, len(coverage), max(1, len(coverage) // 15))
        date_labels = [coverage.index[i].strftime('%Y-%m-%d') for i in date_ticks]
        plt.xticks(date_ticks, date_labels, rotation=45, ha='right')

        if self.flare_dates:
            for flare_date in self.flare_dates:
                try:
                    flare_dt = pd.to_datetime(flare_date).date()
                    if flare_dt in coverage.index:
                        flare_idx = coverage.index.get_loc(flare_dt)
                        plt.axvline(x=flare_idx, color='red', alpha=0.3, linestyle='--')
                except Exception as e:
                    print(f"Error marking flare date {flare_date}: {e}")

        plt.colorbar(label='Number of files')
        plt.title('Dataset Coverage by Date and Type')

        if self.training_period:
            try:
                train_start = pd.to_datetime(self.training_period[0]).date()
                train_end = pd.to_datetime(self.training_period[1]).date()

                if train_start in coverage.index and train_end in coverage.index:
                    start_idx = coverage.index.get_loc(train_start)
                    end_idx = coverage.index.get_loc(train_end)
                    plt.axvspan(start_idx - 0.5, end_idx + 0.5, color='green', alpha=0.2, label='Training Period')
                    plt.legend(loc='upper right')
            except Exception as e:
                print(f"Error marking training period: {e}")

        plt.tight_layout()

        if save_path:
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Explicitly close figure
        else:
            plt.savefig(self.base_dir / "date_coverage.png", dpi=300, bbox_inches='tight')
            plt.close(fig)  # Always close to prevent resource leaks

    def estimate_download_size(self, date_range, data_types, cadence='12h'):
        start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(date_range[1], '%Y-%m-%d')

        days = (end_date - start_date).days + 1

        sizes = {
            'hmi': 15,
            'aia': 8,
            'goes': 0.5,
            'soho': 5
        }

        if cadence.endswith('h'):
            hours_per_sample = float(cadence[:-1])
        elif cadence.endswith('d'):
            hours_per_sample = float(cadence[:-1]) * 24
        else:
            hours_per_sample = 12

        samples_per_day = 24 / hours_per_sample

        total_mb = 0
        for data_type in data_types:
            if data_type == 'aia':
                wavelengths = 8
                total_mb += days * samples_per_day * sizes[data_type] * wavelengths
            elif data_type == 'soho':
                instruments = 2
                total_mb += days * samples_per_day * sizes[data_type] * instruments
            else:
                total_mb += days * samples_per_day * sizes[data_type]

        total_gb = total_mb / 1024

        return total_gb

    def download_incremental_data(self, download_func, start_date, end_date, step_days=30, **kwargs):
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        final_date = datetime.strptime(end_date, '%Y-%m-%d')

        all_files = []

        data_type = kwargs.get('data_type', 'unknown')

        protected = False
        if self.training_period:
            train_start = datetime.strptime(self.training_period[0], '%Y-%m-%d')
            train_end = datetime.strptime(self.training_period[1], '%Y-%m-%d')

            if not (current_date > train_end or final_date < train_start):
                protected = True
                print(f"Files will be protected as they overlap with training period")

        while current_date <= final_date:
            chunk_end = current_date + timedelta(days=step_days)
            if chunk_end > final_date:
                chunk_end = final_date

            print(
                f"Downloading {data_type} data from {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")

            est_size_gb = self.estimate_download_size(
                (current_date.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')),
                [data_type],
                kwargs.get('sample_cadence', '12h')
            )
            print(f"Estimated size: {est_size_gb:.2f} GB")

            if not self.check_storage_available(est_size_gb * 1024 ** 3):
                if not protected and not self.make_space(est_size_gb * 1024 ** 3):
                    print("WARNING: Could not free up enough space. Skipping this chunk.")
                    current_date = chunk_end + timedelta(days=1)
                    continue

            try:
                download_args = kwargs.copy()
                download_args['start_date'] = current_date.strftime('%Y-%m-%d')
                download_args['end_date'] = chunk_end.strftime('%Y-%m-%d')

                if 'data_type' in download_args:
                    del download_args['data_type']

                download_args.pop('priority', None)

                files = download_func(**download_args)

                self.register_files(
                    files,
                    data_type,
                    kwargs.get('priority', 5),
                    protected=protected
                )

                all_files.extend(files)

            except Exception as e:
                print(f"Error downloading chunk: {e}")

            current_date = chunk_end + timedelta(days=1)

        return all_files

    def import_flare_events(self, flare_file_path=None):
        if flare_file_path is None:
            flare_file_path = self.dirs['processed'] / 'flare_events.csv'

        try:
            if os.path.exists(flare_file_path):
                flares = pd.read_csv(flare_file_path)
                if 'time' in flares.columns:
                    flares['time'] = pd.to_datetime(flares['time'])
                    self.flare_dates = flares['time'].tolist()
                    print(f"Imported {len(self.flare_dates)} flare events")

                    self.update_importance_scores()
                    return True
            else:
                print(f"Flare file not found at {flare_file_path}")
        except Exception as e:
            print(f"Error importing flare events: {e}")

        return False

    def set_training_period(self, start_date, end_date):
        self.training_period = (start_date, end_date)
        print(f"Training period set to {start_date} to {end_date}")

        self.update_importance_scores()

        return True
