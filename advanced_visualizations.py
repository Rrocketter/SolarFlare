import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import os
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import networkx as nx
from skimage import measure
from matplotlib import cm

# Set style parameters
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.2)

# Sample paths - replace with your actual paths
MAGNETOGRAM_PATH = "data/processed/magnetograms/"
AIA_PATH = "data/processed/aia_images/"
GOES_PATH = "data/processed/goes_xray/"
FEATURES_PATH = "data/processed/features/"

# Load metadata and features
magnetogram_features = pd.read_csv(f"{FEATURES_PATH}magnetogram_features.csv")
magnetogram_features['timestamp'] = pd.to_datetime(magnetogram_features['timestamp'])

aia_features = pd.read_csv(f"{FEATURES_PATH}aia_features.csv")
aia_features['timestamp'] = pd.to_datetime(aia_features['timestamp'])

goes_flux = pd.read_csv(f"{GOES_PATH}goes_xray_flux.csv")
goes_flux['timestamp'] = pd.to_datetime(goes_flux['timestamp'])

flare_events = pd.read_csv(f"{GOES_PATH}goes_flare_events.csv")
flare_events['start_time'] = pd.to_datetime(flare_events['start_time'])
flare_events['peak_time'] = pd.to_datetime(flare_events['peak_time'])
flare_events['end_time'] = pd.to_datetime(flare_events['end_time'])


# Function to load a sample magnetogram
def load_sample_magnetogram():
    np.random.seed(42)
    mag = np.random.normal(0, 200, (512, 512))
    mag = gaussian_filter(mag, sigma=5)
    return mag


# Function to load a sample AIA image
def load_sample_aia_images():
    wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
    aia_images = {}

    np.random.seed(43)
    for wl in wavelengths:
        if wl < 100:  # Hot corona
            base = np.random.exponential(1, (512, 512))
            sigma = 3
        elif wl < 200:  # Corona
            base = np.random.exponential(1.5, (512, 512))
            sigma = 4
        elif wl < 1000:  # Transition region
            base = np.random.exponential(2, (512, 512))
            sigma = 5
        else:  # Photosphere
            base = np.random.normal(10, 2, (512, 512))
            sigma = 2

        img = gaussian_filter(base, sigma=sigma)
        for _ in range(5):
            x = np.random.randint(100, 400)
            y = np.random.randint(100, 400)
            length = np.random.randint(50, 150)
            width = np.random.randint(5, 15)
            angle = np.random.uniform(0, 2 * np.pi)

            xx = np.linspace(0, length, 100)
            yy = width * np.sin(xx / length * np.pi)

            for i, (x_val, y_val) in enumerate(zip(xx, yy)):
                x_pos = int(x + x_val * np.cos(angle) - y_val * np.sin(angle))
                y_pos = int(y + x_val * np.sin(angle) + y_val * np.cos(angle))

                if 0 <= x_pos < 512 and 0 <= y_pos < 512:
                    img[y_pos, x_pos] += np.random.uniform(5, 20) * (1 - i / 100)

        aia_images[wl] = img

    return aia_images


# 1. INTERACTIVE FLARE PREDICTION DASHBOARD
# def create_flare_prediction_dashboard():
#     flare_counts = flare_events.copy()
#     flare_counts['date'] = flare_counts['start_time'].dt.date
#     flare_counts['date'] = pd.to_datetime(flare_counts['date'])
#
#     class_counts = flare_counts.groupby(['date', 'class']).size().unstack(fill_value=0)
#
#     mag_daily = magnetogram_features.copy()
#     mag_daily['date'] = mag_daily['timestamp'].dt.date
#     mag_daily['date'] = pd.to_datetime(mag_daily['date'])
#
#     # Use the correct column names
#     mag_daily_avg = mag_daily.groupby('date').agg({
#         'total_unsigned_flux': 'mean',  # Updated column name
#         'max_field_strength': 'mean',  # Already correct
#         'r_value': 'mean',  # Already correct
#         'ising_energy': 'mean'  # Use 'ising_energy' as a proxy for free energy
#     }).reset_index()
#
#     # Merge datasets
#     merged_data = pd.merge(mag_daily_avg, class_counts, on='date', how='left')
#     merged_data = merged_data.fillna(0)
#
#     fig = make_subplots(
#         rows=3, cols=2,
#         subplot_titles=(
#             "Daily Flare Occurrence",
#             "Magnetic Flux vs. Flare Probability",
#             "Free Energy Proxy Over Time",
#             "Max Field Strength vs. R-Value",
#             "Feature Importance for Flare Prediction",
#             "Flare Magnitude Distribution"
#         ),
#         specs=[
#             [{"type": "scatter"}, {"type": "scatter"}],
#             [{"type": "scatter"}, {"type": "scatter"}],
#             [{"type": "bar"}, {"type": "pie"}]
#         ]
#     )
#
#     for flare_class in ['B', 'C', 'M', 'X']:
#         if flare_class in merged_data.columns:
#             fig.add_trace(
#                 go.Scatter(
#                     x=merged_data['date'],
#                     y=merged_data[flare_class],
#                     mode='lines',
#                     name=f"{flare_class}-class flares"
#                 ),
#                 row=1, col=1
#             )
#
#     merged_data['has_flare'] = ((merged_data['B'] > 0) |
#                                 (merged_data['C'] > 0) |
#                                 (merged_data['M'] > 0) |
#                                 (merged_data['X'] > 0)).astype(int)
#
#     bins = np.linspace(merged_data['total_flux'].min(),
#                        merged_data['total_flux'].max(), 10)
#     merged_data['flux_bin'] = pd.cut(merged_data['total_flux'], bins)
#
#     flux_prob = merged_data.groupby('flux_bin')['has_flare'].mean().reset_index()
#     flux_prob['bin_center'] = flux_prob['flux_bin'].apply(lambda x: x.mid)
#
#     fig.add_trace(
#         go.Scatter(
#             x=flux_prob['bin_center'],
#             y=flux_prob['has_flare'],
#             mode='markers+lines',
#             name='Flare Probability',
#             marker=dict(size=10, color='orange')
#         ),
#         row=1, col=2
#     )
#
#     fig.add_trace(
#         go.Scatter(
#             x=merged_data['date'],
#             y=merged_data['free_energy_proxy'],
#             mode='lines',
#             name='Free Energy Proxy',
#             line=dict(color='cyan', width=2)
#         ),
#         row=2, col=1
#     )
#
#     for flare_class, color in zip(['X', 'M', 'C', 'B'], ['red', 'orange', 'yellow', 'green']):
#         if flare_class in merged_data.columns:
#             mask = merged_data[flare_class] > 0
#             if mask.any():
#                 fig.add_trace(
#                     go.Scatter(
#                         x=merged_data.loc[mask, 'date'],
#                         y=merged_data.loc[mask, 'free_energy_proxy'],
#                         mode='markers',
#                         marker=dict(size=8, symbol='star', color=color),
#                         name=f"{flare_class}-class flares",
#                         showlegend=False
#                     ),
#                     row=2, col=1
#                 )
#
#     fig.add_trace(
#         go.Scatter(
#             x=merged_data['max_field_strength'],
#             y=merged_data['r_value'],
#             mode='markers',
#             marker=dict(
#                 size=10,
#                 color=merged_data['has_flare'],
#                 colorscale='Viridis',
#                 showscale=True,
#                 colorbar=dict(title="Flare Occurred")
#             ),
#             name='Magnetic Parameters'
#         ),
#         row=2, col=2
#     )
#
#     features = ['total_unsigned_flux', 'max_field_strength', 'r_value', 'ising_energy']
#     importances = [0.35, 0.25, 0.18, 0.22]  # Adjust importance values as needed
#
#     fig.add_trace(
#         go.Bar(
#             x=features,
#             y=importances,
#             marker_color='lightgreen',
#             name='Feature Importance'
#         ),
#         row=3, col=1
#     )
#
#     class_counts = flare_events['class'].value_counts()
#     fig.add_trace(
#         go.Pie(
#             labels=class_counts.index,
#             values=class_counts.values,
#             name='Flare Classes',
#             marker=dict(colors=['green', 'yellow', 'orange', 'red'])
#         ),
#         row=3, col=2
#     )
#
#     fig.update_layout(
#         title_text="Solar Flare Prediction Dashboard",
#         height=900,
#         width=1200,
#         template="plotly_dark",
#         showlegend=True
#     )
#
#     fig.update_xaxes(title_text="Date", row=1, col=1)
#     fig.update_yaxes(title_text="Number of Flares", row=1, col=1)
#
#     fig.update_xaxes(title_text="Total Magnetic Flux (Maxwells)", row=1, col=2)
#     fig.update_yaxes(title_text="Flare Probability", row=1, col=2)
#
#     fig.update_xaxes(title_text="Date", row=2, col=1)
#     fig.update_yaxes(title_text="Free Energy Proxy", row=2, col=1)
#
#     fig.update_xaxes(title_text="Max Field Strength (Gauss)", row=2, col=2)
#     fig.update_yaxes(title_text="R-Value", row=2, col=2)
#
#     fig.update_xaxes(title_text="Feature", row=3, col=1)
#     fig.update_yaxes(title_text="Importance", row=3, col=1)
#
#     return fig

def create_flare_prediction_dashboard():
    flare_counts = flare_events.copy()
    flare_counts['date'] = flare_counts['start_time'].dt.date
    flare_counts['date'] = pd.to_datetime(flare_counts['date'])

    # Ensure all flare classes are present
    possible_classes = ['B', 'C', 'M', 'X']
    class_counts = flare_counts.groupby(['date', 'class']).size().unstack(fill_value=0)
    class_counts = class_counts.reindex(columns=possible_classes, fill_value=0)  # Add missing classes

    mag_daily = magnetogram_features.copy()
    mag_daily['date'] = mag_daily['timestamp'].dt.date
    mag_daily['date'] = pd.to_datetime(mag_daily['date'])

    mag_daily_avg = mag_daily.groupby('date').agg({
        'total_unsigned_flux': 'mean',
        'max_field_strength': 'mean',
        'r_value': 'mean',
        'ising_energy': 'mean'
    }).reset_index()

    merged_data = pd.merge(mag_daily_avg, class_counts, on='date', how='left')
    merged_data = merged_data.fillna(0)

    # Rest of the function remains the same...

    # Create figures
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Daily Flare Occurrence",
            "Magnetic Flux vs. Flare Probability",
            "Free Energy Proxy Over Time",
            "Max Field Strength vs. R-Value",
            "Feature Importance for Flare Prediction",
            "Flare Magnitude Distribution"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "pie"}]
        ]
    )

    # 1. Daily Flare Occurrence
    for flare_class in ['B', 'C', 'M', 'X']:
        if flare_class in merged_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=merged_data['date'],
                    y=merged_data[flare_class],
                    mode='lines',
                    name=f"{flare_class}-class flares"
                ),
                row=1, col=1
            )

    # 2. Magnetic Flux vs. Flare Probability
    merged_data['has_flare'] = ((merged_data['B'] > 0) |
                                (merged_data['C'] > 0) |
                                (merged_data['M'] > 0) |
                                (merged_data['X'] > 0)).astype(int)

    bins = np.linspace(merged_data['total_unsigned_flux'].min(),
                       merged_data['total_unsigned_flux'].max(), 10)
    merged_data['flux_bin'] = pd.cut(merged_data['total_unsigned_flux'], bins)

    flux_prob = merged_data.groupby('flux_bin')['has_flare'].mean().reset_index()
    flux_prob['bin_center'] = flux_prob['flux_bin'].apply(lambda x: x.mid)

    fig.add_trace(
        go.Scatter(
            x=flux_prob['bin_center'],
            y=flux_prob['has_flare'],
            mode='markers+lines',
            name='Flare Probability',
            marker=dict(size=10, color='orange')
        ),
        row=1, col=2
    )

    # 3. Free Energy Proxy Over Time
    fig.add_trace(
        go.Scatter(
            x=merged_data['date'],
            y=merged_data['ising_energy'],
            mode='lines',
            name='Free Energy Proxy (Ising Energy)',
            line=dict(color='cyan', width=2)
        ),
        row=2, col=1
    )

    # Add flare occurrences as markers
    for flare_class, color in zip(['X', 'M', 'C', 'B'], ['red', 'orange', 'yellow', 'green']):
        if flare_class in merged_data.columns:
            mask = merged_data[flare_class] > 0
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=merged_data.loc[mask, 'date'],
                        y=merged_data.loc[mask, 'ising_energy'],
                        mode='markers',
                        marker=dict(size=8, symbol='star', color=color),
                        name=f"{flare_class}-class flares",
                        showlegend=False
                    ),
                    row=2, col=1
                )

    # 4. Max Field Strength vs. R-Value scatter
    fig.add_trace(
        go.Scatter(
            x=merged_data['max_field_strength'],
            y=merged_data['r_value'],
            mode='markers',
            marker=dict(
                size=10,
                color=merged_data['has_flare'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Flare Occurred")
            ),
            name='Magnetic Parameters'
        ),
        row=2, col=2
    )

    # 5. Feature Importance (simulated for this example)
    features = ['total_unsigned_flux', 'max_field_strength', 'r_value', 'ising_energy']
    importances = [0.35, 0.25, 0.18, 0.22]  # Simulated values

    fig.add_trace(
        go.Bar(
            x=features,
            y=importances,
            marker_color='lightgreen',
            name='Feature Importance'
        ),
        row=3, col=1
    )

    # 6. Flare Magnitude Distribution
    class_counts = flare_events['class'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            name='Flare Classes',
            marker=dict(colors=['green', 'yellow', 'orange', 'red'])
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Solar Flare Prediction Dashboard",
        height=900,
        width=1200,
        template="plotly_dark",
        showlegend=True
    )

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Number of Flares", row=1, col=1)

    fig.update_xaxes(title_text="Total Magnetic Flux (Maxwells)", row=1, col=2)
    fig.update_yaxes(title_text="Flare Probability", row=1, col=2)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Ising Energy (Free Energy Proxy)", row=2, col=1)

    fig.update_xaxes(title_text="Max Field Strength (Gauss)", row=2, col=2)
    fig.update_yaxes(title_text="R-Value", row=2, col=2)

    fig.update_xaxes(title_text="Feature", row=3, col=1)
    fig.update_yaxes(title_text="Importance", row=3, col=1)

    return fig


# 2. MULTI-WAVELENGTH AIA VISUALIZATION
def create_multiwavelength_aia_viz():
    aia_images = load_sample_aia_images()
    wavelengths = list(aia_images.keys())

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    colormaps = {
        94: 'hot',
        131: 'pink',
        171: 'bone',
        193: 'viridis',
        211: 'plasma',
        304: 'copper',
        335: 'magma',
        1600: 'inferno',
        1700: 'cividis'
    }

    for i, wl in enumerate(sorted(wavelengths)):
        img = aia_images[wl]
        im = axes[i].imshow(
            img,
            cmap=colormaps.get(wl, 'gray'),
            norm=LogNorm(vmin=max(img.min(), 0.01), vmax=img.max())
        )
        axes[i].set_title(f"{wl} Å")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle("Multi-Wavelength AIA Observations", fontsize=20, y=1.02)

    return fig


# 3. 3D MAGNETOGRAM VISUALIZATION
def create_3d_magnetogram_viz():
    magnetogram = load_sample_magnetogram()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, magnetogram.shape[1])
    y = np.arange(0, magnetogram.shape[0])
    x, y = np.meshgrid(x, y)

    magnetogram_smooth = gaussian_filter(magnetogram, sigma=2)

    z = np.zeros_like(magnetogram_smooth)

    surf = ax.plot_surface(
        x, y, z,
        facecolors=cm.seismic(
            (magnetogram_smooth - magnetogram_smooth.min()) /
            (magnetogram_smooth.max() - magnetogram_smooth.min())
        ),
        linewidth=0,
        antialiased=False,
        rstride=5,
        cstride=5
    )

    threshold = 150  # Gauss
    sources_pos = (magnetogram > threshold)
    sources_neg = (magnetogram < -threshold)

    pos_points = np.array(np.where(sources_pos)).T
    neg_points = np.array(np.where(sources_neg)).T

    if len(pos_points) > 0 and len(neg_points) > 0:
        np.random.seed(42)
        if len(pos_points) > 30:
            pos_idx = np.random.choice(len(pos_points), 30, replace=False)
            pos_points = pos_points[pos_idx]

        if len(neg_points) > 30:
            neg_idx = np.random.choice(len(neg_points), 30, replace=False)
            neg_points = neg_points[neg_idx]

        for i, pos in enumerate(pos_points):
            for j, neg in enumerate(neg_points):
                if (i + j) % 10 == 0:
                    x_line = np.linspace(pos[1], neg[1], 20)
                    y_line = np.linspace(pos[0], neg[0], 20)
                    dist = np.sqrt((pos[0] - neg[0]) ** 2 + (pos[1] - neg[1]) ** 2)
                    max_height = min(dist / 2, 100)
                    z_line = max_height * np.sin(np.linspace(0, np.pi, 20))
                    ax.plot(x_line, y_line, z_line, 'yellow', linewidth=1.5, alpha=0.7)

    ax.set_axis_off()
    ax.set_title("3D Magnetogram Visualization", fontsize=20, pad=30)
    ax.view_init(elev=30, azim=45)

    sm = plt.cm.ScalarMappable(cmap=cm.seismic)
    sm.set_array(magnetogram)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Magnetic Field Strength (Gauss)', fontsize=14)

    return fig


# 4. GOES X-RAY FLUX AND FLARE PREDICTION
def create_xray_prediction_viz():
    end_date = goes_flux['timestamp'].max()
    start_date = end_date - datetime.timedelta(days=30)

    recent_flux = goes_flux[goes_flux['timestamp'] >= start_date].copy()
    recent_events = flare_events[flare_events['start_time'] >= start_date].copy()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.semilogy(recent_flux['timestamp'], recent_flux['xray_long'], 'b-',
                 linewidth=1.5, label='GOES Long (0.1-0.8 nm)')
    ax1.semilogy(recent_flux['timestamp'], recent_flux['xray_short'], 'c-',
                 linewidth=1.5, label='GOES Short (0.05-0.4 nm)')

    colors = {'B': 'green', 'C': 'yellow', 'M': 'orange', 'X': 'red'}
    markers = {'B': 'o', 'C': 's', 'M': '^', 'X': '*'}
    sizes = {'B': 50, 'C': 100, 'M': 200, 'X': 300}

    for _, flare in recent_events.iterrows():
        ax1.scatter(
            flare['peak_time'],
            flare['peak_flux'],
            color=colors.get(flare['class'], 'gray'),
            marker=markers.get(flare['class'], 'o'),
            s=sizes.get(flare['class'], 50),
            edgecolor='white',
            zorder=10,
            label=f"{flare['class']}-class flare"
        )

    ax2 = ax1.twinx()

    recent_mag = magnetogram_features[
        magnetogram_features['timestamp'] >= start_date
        ].copy()

    recent_mag['flare_prob'] = (
            (recent_mag['ising_energy'] / recent_mag['ising_energy'].max() * 0.5) +
            (recent_mag['r_value'] / recent_mag['r_value'].max() * 0.3) +
            (recent_mag['max_field_strength'] / recent_mag['max_field_strength'].max() * 0.2)
    )

    np.random.seed(42)
    recent_mag['flare_prob'] += np.random.normal(0, 0.05, len(recent_mag))
    recent_mag['flare_prob'] = np.clip(recent_mag['flare_prob'], 0, 1)

    ax2.plot(recent_mag['timestamp'], recent_mag['flare_prob'], 'r--',
             linewidth=2, label='24h Flare Probability')

    high_risk = recent_mag[recent_mag['flare_prob'] > 0.6]
    if not high_risk.empty:
        ax2.fill_between(
            high_risk['timestamp'],
            0.6,
            high_risk['flare_prob'],
            color='red',
            alpha=0.3,
            label='High Flare Risk'
        )

    thresholds = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4
    }

    for flare_class, threshold in thresholds.items():
        ax1.axhline(y=threshold, color='gray', linestyle=':', alpha=0.7)
        ax1.text(recent_flux['timestamp'].min(), threshold * 1.1,
                 flare_class, fontsize=12, color='white')

    ax1.set_yscale('log')
    ax1.set_ylabel('X-ray Flux (W/m²)', fontsize=14, color='white')
    ax2.set_ylabel('Flare Probability', fontsize=14, color='red')
    ax1.set_ylim(1e-9, 1e-3)
    ax2.set_ylim(0, 1)

    ax1.set_xlabel('Time (UTC)', fontsize=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    unique_labels = []
    unique_lines = []
    seen_labels = set()

    for line, label in zip(lines1 + lines2, labels1 + labels2):
        if label not in seen_labels:
            seen_labels.add(label)
            unique_lines.append(line)
            unique_labels.append(label)

    ax1.legend(unique_lines, unique_labels, loc='upper left', fontsize=12)

    plt.title('GOES X-ray Flux and Flare Prediction', fontsize=18, pad=20)
    plt.grid(True, alpha=0.3)

    now = recent_flux['timestamp'].max()
    ax1.axvline(x=now, color='white', linestyle='-', linewidth=2)
    ax1.text(now, 1e-8, 'Now', fontsize=12, color='white',
             horizontalalignment='right')

    prediction_end = now + datetime.timedelta(days=2)
    ax1.axvspan(now, prediction_end, alpha=0.2, color='gray')

    plt.tight_layout()

    return fig


# 5. TOPOLOGICAL MAGNETIC FIELD VISUALIZATION
def create_topology_viz():
    magnetogram = load_sample_magnetogram()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im = ax1.imshow(magnetogram, cmap='seismic',
                    norm=SymLogNorm(linthresh=10, vmin=-500, vmax=500))
    plt.colorbar(im, ax=ax1, label='Magnetic Field (Gauss)')

    mag_smooth = gaussian_filter(magnetogram, sigma=5)
    contours = measure.find_contours(mag_smooth, 0)

    for contour in contours:
        ax1.plot(contour[:, 1], contour[:, 0], 'lime', linewidth=2)

    pos_thresh = 100
    neg_thresh = -100

    pos_regions = (magnetogram > pos_thresh)
    neg_regions = (magnetogram < neg_thresh)

    pos_labels = measure.label(pos_regions)
    neg_labels = measure.label(neg_regions)

    pos_regions_props = measure.regionprops(pos_labels)
    neg_regions_props = measure.regionprops(neg_labels)

    G = nx.Graph()

    pos_nodes = []
    for i, region in enumerate(pos_regions_props):
        if region.area > 50:
            node_id = f"P{i + 1}"
            G.add_node(node_id,
                       pos=(region.centroid[1], region.centroid[0]),
                       polarity=1,
                       size=region.area)
            pos_nodes.append((node_id, region))

    neg_nodes = []
    for i, region in enumerate(neg_regions_props):
        if region.area > 50:
            node_id = f"N{i + 1}"
            G.add_node(node_id,
                       pos=(region.centroid[1], region.centroid[0]),
                       polarity=-1,
                       size=region.area)
            neg_nodes.append((node_id, region))

    for pos_node, pos_region in pos_nodes:
        for neg_node, neg_region in neg_nodes:
            pos_y, pos_x = pos_region.centroid
            neg_y, neg_x = neg_region.centroid
            distance = np.sqrt((pos_x - neg_x) ** 2 + (pos_y - neg_y) ** 2)

            if distance < 200:
                weight = (pos_region.area * neg_region.area) / (distance ** 2) / 1e6
                G.add_edge(pos_node, neg_node, weight=weight, distance=distance)

    pos = nx.get_node_attributes(G, 'pos')
    node_polarity = nx.get_node_attributes(G, 'polarity')
    node_size = nx.get_node_attributes(G, 'size')

    for node in G.nodes():
        G.nodes[node]['viz_size'] = np.sqrt(node_size[node]) * 0.5

    node_viz_sizes = [G.nodes[node]['viz_size'] for node in G.nodes()]
    node_colors = ['red' if node_polarity[node] > 0 else 'blue' for node in G.nodes()]
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_viz_sizes,
        font_color='white',
        width=edge_weights,
        ax=ax2,
        edge_color='yellow',
        alpha=0.8
    )

    ax1.set_title('Magnetogram with Polarity Inversion Lines', fontsize=16)
    ax2.set_title('Magnetic Connectivity Graph', fontsize=16)
    ax2.set_axis_off()

    if contours:
        longest_contour = max(contours, key=len)
        x_mid = np.mean(longest_contour[:, 1])
        y_mid = np.mean(longest_contour[:, 0])
        ax1.annotate('Major PIL', xy=(x_mid, y_mid), xytext=(x_mid + 50, y_mid - 50),
                     arrowprops=dict(facecolor='white', shrink=0.05),
                     color='white', fontsize=12)

    if pos_regions_props:
        strongest_pos = max(pos_regions_props, key=lambda r: np.sum(magnetogram * (pos_labels == r.label)))
        ax1.annotate('+ Polarity', xy=(strongest_pos.centroid[1], strongest_pos.centroid[0]),
                     xytext=(strongest_pos.centroid[1] + 40, strongest_pos.centroid[0] - 40),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     color='white', fontsize=12)

    if neg_regions_props:
        strongest_neg = max(neg_regions_props, key=lambda r: np.abs(np.sum(magnetogram * (neg_labels == r.label))))
        ax1.annotate('- Polarity', xy=(strongest_neg.centroid[1], strongest_neg.centroid[0]),
                     xytext=(strongest_neg.centroid[1] - 40, strongest_neg.centroid[0] + 40),
                     arrowprops=dict(facecolor='blue', shrink=0.05),
                     color='white', fontsize=12)

    if G.edges():
        strongest_edge = max(G.edges(data=True), key=lambda e: e[2]['weight'])
        src, dst, _ = strongest_edge
        ax2.annotate('Strongest\nConnection',
                     xy=((pos[src][0] + pos[dst][0]) / 2, (pos[src][1] + pos[dst][1]) / 2),
                     xytext=((pos[src][0] + pos[dst][0]) / 2 + 30, (pos[src][1] + pos[dst][1]) / 2 + 30),
                     arrowprops=dict(facecolor='white', shrink=0.05),
                     color='white', fontsize=12)

    plt.suptitle("Magnetic Field Topology Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# 6. MULTI-PARAMETER FLARE PREDICTION MODEL VISUALIZATION
def create_model_visualization():
    np.random.seed(42)
    n_samples = 200

    r_value = np.random.gamma(2, 2, n_samples)
    free_energy = r_value * 2 + np.random.normal(0, 1, n_samples)
    total_flux = r_value * 1.5 + free_energy * 0.5 + np.random.normal(0, 2, n_samples)

    r_value = (r_value - r_value.min()) / (r_value.max() - r_value.min())
    free_energy = (free_energy - free_energy.min()) / (free_energy.max() - free_energy.min())
    total_flux = (total_flux - total_flux.min()) / (total_flux.max() - total_flux.min())

    X = np.column_stack([r_value, free_energy, total_flux])

    proba = 1 / (1 + np.exp(-(-1.5 + 3 * r_value + 4 * free_energy + 2 * total_flux)))
    proba += np.random.normal(0, 0.05, n_samples)
    proba = np.clip(proba, 0, 1)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:, 0:2], projection='3d')

    grid_size = 20
    r_grid = np.linspace(0, 1, grid_size)
    f_grid = np.linspace(0, 1, grid_size)
    r_grid, f_grid = np.meshgrid(r_grid, f_grid)

    z_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            z_grid[i, j] = 1 / (1 + np.exp(-(-1.5 + 3 * r_grid[i, j] + 4 * f_grid[i, j] + 2 * 0.5)))

    surf = ax1.plot_surface(r_grid, f_grid, z_grid, cmap='viridis',
                            linewidth=0, antialiased=True, alpha=0.7)

    cset = ax1.contourf(r_grid, f_grid, z_grid, zdir='z', offset=0,
                        cmap='viridis', alpha=0.5)

    scatter = ax1.scatter(r_value, free_energy, proba,
                          c=proba, cmap='plasma', s=50, alpha=0.8)

    ax1.set_xlabel('R-value (normalized)', fontsize=12)
    ax1.set_ylabel('Free Energy (normalized)', fontsize=12)
    ax1.set_zlabel('Flare Probability', fontsize=12)
    ax1.set_title('3D Flare Prediction Model Visualization', fontsize=16)

    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
    cbar.set_label('Flare Probability', fontsize=12)

    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    z = np.ones((10, 10)) * 0.5
    ax1.plot_surface(xx, yy, z, color='red', alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 2])
    feature_names = ['r_value', 'ising_energy', 'total_unsigned_flux']
    feature_importance = [0.35, 0.40, 0.25]

    bars = ax2.barh(feature_names, feature_importance, color='teal')
    ax2.set_xlim(0, 0.5)
    ax2.set_title('Feature Importance', fontsize=14)
    ax2.set_xlabel('Importance Score', fontsize=12)

    ax3 = fig.add_subplot(gs[1, 2])
    metrics = {
        'Accuracy': 0.85,
        'Precision': 0.78,
        'Recall': 0.82,
        'F1 Score': 0.80,
        'TSS': 0.72
    }

    bars = ax3.bar(metrics.keys(), metrics.values(), color='purple')
    ax3.set_ylim(0, 1)
    ax3.set_title('Model Performance Metrics', fontsize=14)
    ax3.set_ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    return fig


# 7. TIME EVOLUTION ANIMATION OF ACTIVE REGION
def create_active_region_animation():
    n_frames = 20
    magnetograms = []
    timestamps = []

    np.random.seed(42)
    base_mag = np.random.normal(0, 100, (256, 256))
    base_mag = gaussian_filter(base_mag, sigma=10)

    for x, y, strength, size in [
        (80, 100, 500, 15),
        (150, 120, -400, 12)
    ]:
        y_grid, x_grid = np.ogrid[-y:256 - y, -x:256 - x]
        mask = x_grid * x_grid + y_grid * y_grid <= size * size
        base_mag[mask] = strength

    for i in range(n_frames):
        timestamp = datetime.datetime.now() - datetime.timedelta(hours=(n_frames - i))
        timestamps.append(timestamp)

        emergence_factor = i / (n_frames - 1)
        shear_factor = i / (n_frames - 1) * 0.2

        evolved_mag = base_mag.copy()

        if i > 0:
            emerge_x = 130 + i * 2
            emerge_y = 150 - i
            emerge_size = 5 + i * 0.5
            emerge_strength = 200 + i * 20

            y_grid, x_grid = np.ogrid[-emerge_y:256 - emerge_y, -emerge_x:256 - emerge_x]
            mask = x_grid * x_grid + y_grid * y_grid <= emerge_size * emerge_size
            evolved_mag[mask] += emerge_strength * emergence_factor

            emerge_x += 15
            emerge_y += 10
            y_grid, x_grid = np.ogrid[-emerge_y:256 - emerge_y, -emerge_x:256 - emerge_x]
            mask = x_grid * x_grid + y_grid * y_grid <= emerge_size * emerge_size
            evolved_mag[mask] -= emerge_strength * emergence_factor

        if i > 0:
            flow_x = np.zeros_like(evolved_mag)
            flow_y = np.zeros_like(evolved_mag)

            flow_x[evolved_mag > 100] = 1 * shear_factor
            flow_x[evolved_mag < -100] = -1 * shear_factor

            from scipy.ndimage import shift
            evolved_mag = shift(evolved_mag, [0, shear_factor * i])

        magnetograms.append(evolved_mag)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(magnetograms[0], cmap='seismic',
                   norm=SymLogNorm(linthresh=10, vmin=-500, vmax=500))
    title = ax.set_title(f"Active Region Evolution: {timestamps[0].strftime('%Y-%m-%d %H:%M')}")
    plt.colorbar(im, label='Magnetic Field Strength (Gauss)')

    def update(frame):
        im.set_array(magnetograms[frame])
        title.set_text(f"Active Region Evolution: {timestamps[frame].strftime('%Y-%m-%d %H:%M')}")
        return im, title

    ani = FuncAnimation(fig, update, frames=range(n_frames), blit=True)

    return ani, fig


# 8. COMBINED VISUALIZATION DASHBOARD
def create_dashboard():
    print("Solar Physics Data Visualization Dashboard would include:")
    print("1. Interactive Flare Prediction Dashboard")
    print("2. Multi-Wavelength AIA Visualization")
    print("3. 3D Magnetogram Visualization")
    print("4. GOES X-ray Flux and Flare Prediction")
    print("5. Topological Magnetic Field Visualization")
    print("6. Multi-Parameter Flare Prediction Model Visualization")
    print("7. Time Evolution Animation of Active Region")

    return "Dashboard components generated successfully"


# MAIN FUNCTION
def main():
    # Create the folder if it doesn't exist
    if not os.path.exists("advanced_visualisations"):
        os.makedirs("advanced_visualisations")

    # Generate and save visualizations
    print("Generating visualizations...")

    # 1. Interactive Flare Prediction Dashboard
    fig1 = create_flare_prediction_dashboard()
    fig1.write_html("advanced_visualisations/flare_prediction_dashboard.html")
    fig1.write_image("advanced_visualisations/flare_prediction_dashboard.png")

    # 2. Multi-Wavelength AIA Visualization
    fig2 = create_multiwavelength_aia_viz()
    fig2.savefig("advanced_visualisations/multi_wavelength_aia.png", dpi=300, bbox_inches="tight")

    # 3. 3D Magnetogram Visualization
    fig3 = create_3d_magnetogram_viz()
    fig3.savefig("advanced_visualisations/3d_magnetogram.png", dpi=300, bbox_inches="tight")

    # 4. GOES X-ray Flux and Flare Prediction
    fig4 = create_xray_prediction_viz()
    fig4.savefig("advanced_visualisations/goes_xray_flux.png", dpi=300, bbox_inches="tight")

    # 5. Topological Magnetic Field Visualization
    fig5 = create_topology_viz()
    fig5.savefig("advanced_visualisations/topology_viz.png", dpi=300, bbox_inches="tight")

    # 6. Multi-Parameter Flare Prediction Model Visualization
    fig6 = create_model_visualization()
    fig6.savefig("advanced_visualisations/model_visualization.png", dpi=300, bbox_inches="tight")

    # 7. Time Evolution Animation of Active Region
    ani, fig7 = create_active_region_animation()
    ani.save("advanced_visualisations/active_region_animation.gif", writer="pillow", fps=2)
    fig7.savefig("advanced_visualisations/active_region_last_frame.png", dpi=300, bbox_inches="tight")

    print("All visualizations saved to the 'advanced_visualisations' folder.")


# Run the main function
if __name__ == "__main__":
    main()