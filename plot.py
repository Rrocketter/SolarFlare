import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# Define paths to processed data
PROCESSED_ROOT = 'data/processed'
OUTPUT_DIR = 'plots'

# Create output directory for plots
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_magnetogram_data():
    """Plot and save example HMI magnetogram data"""
    print("Plotting HMI magnetogram data...")

    # Load magnetogram metadata
    metadata_path = os.path.join(PROCESSED_ROOT, 'magnetograms', 'magnetogram_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: Magnetogram metadata file not found at {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    if metadata_df.empty:
        print("Error: Magnetogram metadata is empty")
        return

    # Select a sample of magnetograms to plot (up to 3)
    sample_size = min(3, len(metadata_df))
    sample_indices = np.linspace(0, len(metadata_df) - 1, sample_size, dtype=int)

    for i, idx in enumerate(sample_indices):
        try:
            row = metadata_df.iloc[idx]
            filename = row['filename']
            timestamp = row['timestamp']

            # Load magnetogram data
            magnetogram = np.load(filename)

            # Create figure
            plt.figure(figsize=(10, 8))

            # Determine appropriate vmax based on data range (symmetrical for magnetograms)
            vmax = np.percentile(np.abs(magnetogram), 99)

            # Plot magnetogram using diverging colormap
            plt.imshow(magnetogram, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(label='Magnetic Field Strength (Gauss)')
            plt.title(f'HMI Magnetogram\n{timestamp}')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')

            # Save figure
            output_file = os.path.join(OUTPUT_DIR, f'magnetogram_sample_{i + 1}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved magnetogram plot to {output_file}")

        except Exception as e:
            print(f"Error plotting magnetogram {idx}: {e}")


def plot_aia_data():
    """Plot and save AIA data in both grid and individual formats"""
    print("Plotting AIA image data...")

    # Load AIA metadata
    metadata_path = os.path.join(PROCESSED_ROOT, 'aia_images', 'aia_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: AIA metadata file not found at {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    if metadata_df.empty:
        print("Error: AIA metadata is empty")
        return

    # Get unique wavelengths and sort them
    wavelengths = sorted(metadata_df['wavelength'].unique())
    if not wavelengths:
        print("No AIA wavelengths found in metadata")
        return

    # Create grid plot
    create_aia_grid(metadata_df, wavelengths)

    # Create individual plots
    create_individual_plots(metadata_df, wavelengths)


def create_aia_grid(metadata_df, wavelengths):
    """Create a grid plot of all AIA wavelengths"""
    print("Creating combined wavelength grid...")

    cols = 4
    rows = int(np.ceil(len(wavelengths) / cols))
    fig = plt.figure(figsize=(20, 5 * rows))

    for i, wavelength in enumerate(wavelengths, 1):
        ax = fig.add_subplot(rows, cols, i)
        try:
            # Get middle sample
            wavelength_df = metadata_df[metadata_df['wavelength'] == wavelength]
            middle_idx = len(wavelength_df) // 2
            row = wavelength_df.iloc[middle_idx]

            # Load data
            with np.load(row['filename']) as npz_file:
                aia_img = npz_file['data']

            # Plot parameters
            vmin = np.percentile(aia_img, 1)
            vmax = np.percentile(aia_img, 99.5)
            cmap, norm = get_aia_colormap(wavelength, vmin, vmax)

            # Plot to grid
            img = ax.imshow(aia_img, norm=norm, cmap=cmap)
            plt.colorbar(img, ax=ax, label='DN/s', fraction=0.046, pad=0.04)
            ax.set_title(f'AIA {wavelength}Å\n{row["timestamp"]}', fontsize=10)
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{wavelength}Å", ha='center', va='center')
            ax.axis('off')
            print(f"Grid error for {wavelength}Å: {str(e)[:50]}...")

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'aia_all_wavelengths_grid.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved grid plot to {output_file}")


def create_individual_plots(metadata_df, wavelengths):
    """Create individual plots for each wavelength"""
    print("Creating individual wavelength plots...")

    for wavelength in wavelengths:
        try:
            # Get middle sample
            wavelength_df = metadata_df[metadata_df['wavelength'] == wavelength]
            middle_idx = len(wavelength_df) // 2
            row = wavelength_df.iloc[middle_idx]

            # Load data
            with np.load(row['filename']) as npz_file:
                aia_img = npz_file['data']

            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            # Plot parameters
            vmin = np.percentile(aia_img, 1)
            vmax = np.percentile(aia_img, 99.5)
            cmap, norm = get_aia_colormap(wavelength, vmin, vmax)

            # Plot image
            img = ax.imshow(aia_img, norm=norm, cmap=cmap)
            plt.colorbar(img, label='Intensity (DN/s)')
            ax.set_title(f'AIA {wavelength}Å Image\n{row["timestamp"]}')
            ax.axis('off')

            # Save individual plot
            output_file = os.path.join(OUTPUT_DIR,
                                       f'aia_{wavelength}A_{row["timestamp"].replace(":", "").replace(" ", "_")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved individual plot for {wavelength}Å to {output_file}")

        except Exception as e:
            print(f"Individual plot error for {wavelength}Å: {str(e)[:50]}...")


def get_aia_colormap(wavelength, vmin, vmax):
    """Return appropriate colormap and normalization for AIA wavelength"""
    from matplotlib.colors import LogNorm

    cmap_options = {
        94: ['sdoaia94', 'viridis'],
        131: ['sdoaia131', 'plasma'],
        171: ['sdoaia171', 'magma'],
        193: ['sdoaia193', 'inferno'],
        211: ['sdoaia211', 'cividis'],
        304: ['sdoaia304', 'pink'],
        335: ['sdoaia335', 'hot'],
        1600: ['sdoaia1600', 'bone']
    }

    cmaps = cmap_options.get(wavelength, ['viridis'])
    for cmap in cmaps:
        try:
            plt.get_cmap(cmap)
            return cmap, LogNorm(vmin=max(vmin, 1e-2), vmax=vmax)
        except ValueError:
            continue
    return 'viridis', LogNorm(vmin=max(vmin, 1e-2), vmax=vmax)


# def plot_goes_data():
#     """Plot and save GOES X-ray flux data and flare events"""
#     print("Plotting GOES X-ray flux data...")
#
#     # Load GOES data
#     goes_path = os.path.join(PROCESSED_ROOT, 'goes_xray', 'goes_xray_flux.csv')
#     flares_path = os.path.join(PROCESSED_ROOT, 'goes_xray', 'goes_flare_events.csv')
#
#     if not os.path.exists(goes_path):
#         print(f"Error: GOES flux data not found at {goes_path}")
#         return
#
#     # Load GOES X-ray flux data
#     goes_df = pd.read_csv(goes_path)
#     if goes_df.empty:
#         print("Error: GOES flux data is empty")
#         return
#
#     # Convert timestamp to datetime
#     goes_df['timestamp'] = pd.to_datetime(goes_df['timestamp'])
#
#     # Check if we have flare data
#     have_flares = os.path.exists(flares_path)
#     flares_df = None
#     if have_flares:
#         flares_df = pd.read_csv(flares_path)
#         if not flares_df.empty:
#             flares_df['peak_time'] = pd.to_datetime(flares_df['peak_time'])
#
#     # Create time windows to plot (up to 3)
#     # Choose windows that include flares if possible
#     if have_flares and not flares_df.empty:
#         # Find time windows around flares
#         time_windows = []
#         for i, row in flares_df.iloc[::max(1, len(flares_df) // 3)].iterrows():
#             if len(time_windows) >= 3:
#                 break
#             peak_time = row['peak_time']
#             # Create a 24-hour window around the flare
#             start_time = peak_time - pd.Timedelta(hours=12)
#             end_time = peak_time + pd.Timedelta(hours=12)
#             time_windows.append((start_time, end_time, row['class']))
#     else:
#         # If no flares, just divide the data into 3 equal parts
#         min_time = goes_df['timestamp'].min()
#         max_time = goes_df['timestamp'].max()
#         total_days = (max_time - min_time).total_seconds() / (24 * 3600)
#
#         if total_days < 1:
#             # Less than 1 day of data, just plot all of it
#             time_windows = [(min_time, max_time, None)]
#         else:
#             # Create up to 3 windows
#             window_size = pd.Timedelta(days=total_days / 3)
#             time_windows = []
#             for i in range(min(3, int(total_days))):
#                 start_time = min_time + i * window_size
#                 end_time = start_time + window_size
#                 time_windows.append((start_time, end_time, None))
#
#     # Plot each time window
#     for i, (start_time, end_time, flare_class) in enumerate(time_windows):
#         try:
#             # Filter data for this time window
#             window_data = goes_df[(goes_df['timestamp'] >= start_time) &
#                                   (goes_df['timestamp'] <= end_time)]
#
#             # Create figure
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#             # Plot X-ray flux (both channels)
#             ax.semilogy(window_data['timestamp'], window_data['xray_long'], 'r-',
#                         label='1-8Å (GOES Long/X-ray)')
#             ax.semilogy(window_data['timestamp'], window_data['xray_short'], 'b-',
#                         label='0.5-4Å (GOES Short/X-ray)')
#
#             # Add horizontal lines for flare classes
#             flare_levels = {
#                 'A': 1e-8,
#                 'B': 1e-7,
#                 'C': 1e-6,
#                 'M': 1e-5,
#                 'X': 1e-4
#             }
#
#             for cls, level in flare_levels.items():
#                 ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
#                 ax.text(start_time + pd.Timedelta(hours=1), level * 1.1, cls, color='gray')
#
#             # Mark flares if available
#             if have_flares and not flares_df.empty:
#                 flares_in_window = flares_df[(flares_df['peak_time'] >= start_time) &
#                                              (flares_df['peak_time'] <= end_time)]
#
#                 for _, flare in flares_in_window.iterrows():
#                     ax.axvline(x=flare['peak_time'], color='magenta', linestyle='-', alpha=0.7)
#                     ax.text(flare['peak_time'], flare['peak_flux'] * 1.5,
#                             flare['class'], color='magenta')
#
#             # Format the plot
#             ax.set_yscale('log')
#             ax.set_ylim(1e-9, 1e-3)
#             ax.set_xlim(start_time, end_time)
#
#             date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
#             ax.xaxis.set_major_formatter(date_format)
#             fig.autofmt_xdate()
#
#             # Add title and labels
#             if flare_class:
#                 title = f'GOES X-ray Flux with {flare_class} Flare\n{start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}'
#             else:
#                 title = f'GOES X-ray Flux\n{start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}'
#
#             ax.set_title(title)
#             ax.set_xlabel('Time (UTC)')
#             ax.set_ylabel('X-ray Flux (W/m²)')
#             ax.grid(True, which='both', alpha=0.3)
#             ax.legend(loc='upper right')
#
#             # Save figure
#             output_file = os.path.join(OUTPUT_DIR, f'goes_xray_sample_{i + 1}.png')
#             plt.savefig(output_file, dpi=300, bbox_inches='tight')
#             plt.close()
#
#             print(f"Saved GOES X-ray flux plot to {output_file}")
#
#         except Exception as e:
#             print(f"Error plotting GOES data for window {i + 1}: {e}")


def plot_goes_data():
    """
    Create clean and informative GOES X-ray flux plots
    for a specific date range
    """
    print("Generating refined GOES X-ray flux visualizations...")

    # Load GOES data
    goes_path = os.path.join(PROCESSED_ROOT, 'goes_xray', 'goes_xray_flux.csv')
    flares_path = os.path.join(PROCESSED_ROOT, 'goes_xray', 'goes_flare_events.csv')

    # Validate data exists
    if not os.path.exists(goes_path):
        print(f"Error: GOES flux data not found at {goes_path}")
        return

    # Load data with explicit conversion
    goes_df = pd.read_csv(goes_path)
    goes_df['timestamp'] = pd.to_datetime(goes_df['timestamp'])

    # Clean numeric columns
    numeric_columns = ['xray_long', 'xray_short']
    for col in numeric_columns:
        goes_df[col] = pd.to_numeric(goes_df[col], errors='coerce')

    # Remove any rows with NaN values
    goes_df = goes_df.dropna(subset=numeric_columns)

    # Specify date range
    start_date = pd.Timestamp('2017-09-01')
    end_date = pd.Timestamp('2017-09-09')

    # Filter data for specified date range
    goes_df_filtered = goes_df[(goes_df['timestamp'] >= start_date) &
                               (goes_df['timestamp'] <= end_date)]

    # Load flares if available
    flares_df = None
    if os.path.exists(flares_path):
        flares_df = pd.read_csv(flares_path)
        if not flares_df.empty:
            flares_df['peak_time'] = pd.to_datetime(flares_df['peak_time'])

            # Filter flares for the same date range
            flares_df = flares_df[
                (flares_df['peak_time'] >= start_date) &
                (flares_df['peak_time'] <= end_date)
                ]

    # Enhanced color palette
    colors = {
        'long_xray': '#FF4500',  # Vibrant orange-red
        'short_xray': '#4169E1',  # Royal blue
        'background': '#F0F0F0'  # Light gray background
    }

    # Plotting function with clean design
    def plot_goes_window(goes_data, flares=None, window_title=None):
        plt.figure(figsize=(16, 8), facecolor=colors['background'])
        plt.style.use('seaborn-v0_8-whitegrid')

        # Simple moving average
        window_size = min(31, len(goes_data) // 10)  # Adaptive window size
        smooth_long = goes_data['xray_long'].rolling(window=window_size, center=True, min_periods=1).mean()
        smooth_short = goes_data['xray_short'].rolling(window=window_size, center=True, min_periods=1).mean()

        # Plot smoothed X-ray flux
        plt.semilogy(goes_data['timestamp'], smooth_long,
                     color=colors['long_xray'],
                     linewidth=2.5,
                     label='1-8Å (Long/X-ray)')
        plt.semilogy(goes_data['timestamp'], smooth_short,
                     color=colors['short_xray'],
                     linewidth=2.5,
                     label='0.5-4Å (Short/X-ray)')

        # Simplified flare classification thresholds
        flare_levels = {
            'C': 1e-6,  # Only show C, M, X class thresholds
            'M': 1e-5,
            'X': 1e-4
        }

        for cls, level in flare_levels.items():
            plt.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
            plt.text(goes_data['timestamp'].max(),
                     level * 1.1,
                     f'{cls}-Class',
                     color='gray',
                     ha='right',
                     fontsize=9)

        # Simplified flare marking
        if flares is not None and not flares.empty:
            # Only mark significant flares (M and X class)
            significant_flares = flares[flares['class'].str[0].isin(['M', 'X'])]
            for _, flare in significant_flares.iterrows():
                plt.axvline(x=flare['peak_time'],
                            color='magenta',
                            linestyle='-',
                            linewidth=2,
                            alpha=0.7)
                plt.text(flare['peak_time'],
                         flare['peak_flux'] * 1.5,
                         flare['class'],
                         color='magenta',
                         ha='center',
                         fontsize=10)

        # Improved axis formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate(rotation=45)

        # Enhanced title and labels
        plt.title(window_title or 'GOES X-ray Flux Observations',
                  fontsize=15,
                  fontweight='bold')
        plt.xlabel('Time (UTC)', fontsize=12)
        plt.ylabel('X-ray Flux (W/m²)', fontsize=12)

        # Refined plot limits and legend
        plt.ylim(1e-8, 1e-3)  # Adjusted to show more detail
        plt.legend(loc='upper right', framealpha=0.8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        # Save with high resolution
        output_file = os.path.join(OUTPUT_DIR,
                                   f'goes_xray_refined_{window_title.replace(" ", "_") if window_title else "plot"}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Refined visualization saved to {output_file}")

    # Plot for the specified date range
    plot_goes_window(goes_df_filtered, flares_df,
                     f'GOES X-ray Flux (2017-09-01 to 2017-09-09)')

    print("GOES X-ray flux visualization complete.")

def plot_feature_data():
    """Plot extracted features and their relationship to flares"""
    print("Plotting physics-based features data...")

    # Load feature data
    features_path = os.path.join(PROCESSED_ROOT, 'features', 'magnetogram_features.csv')

    if not os.path.exists(features_path):
        print(f"Error: Features data not found at {features_path}")
        return

    features_df = pd.read_csv(features_path)
    if features_df.empty:
        print("Error: Features data is empty")
        return

    # Check if we have the 'has_flare' column (depends on data processing)
    has_flare_column = 'has_flare' in features_df.columns

    # Convert timestamp to datetime if it exists
    if 'timestamp' in features_df.columns:
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    # Select relevant physics-based features if they exist
    possible_features = [
        'total_unsigned_flux', 'max_field_strength', 'r_value',
        'magnetic_shear', 'current_helicity', 'ising_energy', 'lorentz_force'
    ]

    # Filter to features that actually exist in our dataset
    plot_features = [f for f in possible_features if f in features_df.columns]

    if not plot_features:
        print("No physics-based features found to plot")
        return

    # Plot time series of selected features
    if 'timestamp' in features_df.columns:
        # Create a figure for time series of features
        plt.figure(figsize=(14, 10))

        for i, feature in enumerate(plot_features[:4]):  # Limit to 4 features for clarity
            plt.subplot(2, 2, i + 1)
            plt.plot(features_df['timestamp'], features_df[feature], 'b-')

            # Mark flare events if available
            if has_flare_column:
                flare_times = features_df[features_df['has_flare']]['timestamp']
                for t in flare_times:
                    plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)

            plt.title(f'{feature} over Time')
            plt.xlabel('Time')
            plt.ylabel(feature)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, 'features_timeseries.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved features time series plot to {output_file}")

    # Create correlation plot between features
    if len(plot_features) > 1:
        try:
            # Calculate correlation matrix
            corr = features_df[plot_features].corr()

            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

            # Add text annotations
            for i in range(len(corr)):
                for j in range(len(corr)):
                    text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black")

            plt.colorbar(label='Correlation')
            plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr)), corr.index)
            plt.title('Feature Correlation Matrix')

            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'features_correlation.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved feature correlation plot to {output_file}")

        except Exception as e:
            print(f"Error creating correlation plot: {e}")


def main():
    """Run all plotting functions"""
    print("Starting to plot processed solar data...")

    # Plot each data type
    plot_magnetogram_data()
    plot_aia_data()
    plot_goes_data()
    plot_feature_data()

    print(f"All plots saved to {OUTPUT_DIR} directory")


if __name__ == "__main__":
    main()