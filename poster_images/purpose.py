import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, ConnectionPatch, FancyArrowPatch
from matplotlib import rcParams

# Set up font configuration with fallbacks
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI']
rcParams['axes.titlesize'] = 28
rcParams['axes.labelsize'] = 25

# Create figure with white background
fig = plt.figure(figsize=(24, 16), facecolor='white')
ax_main = fig.add_subplot(111)
ax_main.set_aspect('equal')
ax_main.axis('off')

# Set background to white
ax_main.set_facecolor('white')

# Solar System Elements --------------------------------------------------------
sun_pos = (-9, 0)
sun_radius = 2
sun = Circle(sun_pos, sun_radius, color='#ff9933', ec='#ff3300', lw=4, zorder=10)
ax_main.add_patch(sun)

# Solar flare eruption
flare_angles = [np.deg2rad(40), np.deg2rad(50), np.deg2rad(60)]
for i, angle in enumerate(flare_angles):
    flare_length = 4 + i * 0.5
    flare_x = sun_pos[0] + sun_radius * np.cos(angle)
    flare_y = sun_pos[1] + sun_radius * np.sin(angle)
    ax_main.annotate('', xy=(flare_x, flare_y),
                     xytext=(flare_x - flare_length * np.cos(angle),
                             flare_y - flare_length * np.sin(angle)),
                     arrowprops=dict(arrowstyle='wedge', color='#ff0000',
                                     lw=4 - i * 0.5, alpha=0.6 - 0.1 * i))

# Earth system
earth_pos = (9, 0)
earth_radius = 0.8
earth = Circle(earth_pos, earth_radius, color='#3399ff', zorder=10)
ax_main.add_patch(earth)

# Magnetosphere
theta = np.linspace(0, 2 * np.pi, 150)
r = np.linspace(earth_radius, earth_radius * 5, 50)
theta_grid, r_grid = np.meshgrid(theta, r)
x_mag = earth_pos[0] + r_grid * np.cos(theta_grid) * (2 - 0.8 * np.cos(theta_grid))
y_mag = earth_pos[1] + r_grid * np.sin(theta_grid)
ax_main.contour(x_mag, y_mag, r_grid, 10, colors='#0000ff', alpha=0.3, linewidths=1)

# Impact Pathways -------------------------------------------------------------
pathways = {
    'CME (1-3 days)': {'color': '#FF4500', 'rad': 0.3, 'width': 4},
    'Energetic Particles (15-60 min)': {'color': '#FFA500', 'rad': 0.15, 'width': 3},
    'X-rays (8 min)': {'color': '#FFD700', 'rad': 0, 'width': 2}
}

for name, props in pathways.items():
    arrow = FancyArrowPatch(sun_pos, earth_pos,
                            connectionstyle=f"arc3,rad={props['rad']}",
                            color=props['color'], lw=props['width'],
                            alpha=0.8, zorder=5)
    ax_main.add_patch(arrow)

# Satellite Constellation ------------------------------------------------------
def detailed_satellite(ax, pos, scale=1):
    body = Ellipse(pos, 0.3 * scale, 0.15 * scale, angle=45, color='#555555', zorder=15)
    ax.add_patch(body)
    panel_length = 0.6 * scale
    ax.plot([pos[0] - 0.15 * scale, pos[0] - 0.15 * scale - panel_length * np.cos(np.deg2rad(45))],
            [pos[1], pos[1] - panel_length * np.sin(np.deg2rad(45))],
            color='#777777', lw=2 * scale, solid_capstyle='round')
    ax.plot([pos[0] + 0.15 * scale, pos[0] + 0.15 * scale + panel_length * np.cos(np.deg2rad(45))],
            [pos[1], pos[1] + panel_length * np.sin(np.deg2rad(45))],
            color='#777777', lw=2 * scale, solid_capstyle='round')
    ax.plot([pos[0], pos[0]], [pos[1] + 0.1 * scale, pos[1] + 0.3 * scale],
            color='#999999', lw=1 * scale)

# Adjusted satellite positions
orbit_radius = earth_radius * 6
for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    sat_pos = (earth_pos[0] + orbit_radius * np.cos(angle),
               earth_pos[1] + orbit_radius * np.sin(angle))
    detailed_satellite(ax_main, sat_pos)

# Impact Annotations ----------------------------------------------------------
impacts = [
    {'label': 'Satellite Damage', 'pos': (13, 4), 'color': '#FF4444'},
    {'label': 'Radio Blackouts', 'pos': (13, 2), 'color': '#FF8844'},
    {'label': 'Power Grid Instability', 'pos': (13, 0), 'color': '#FF4444'},
    {'label': 'GPS Disruption', 'pos': (13, -2), 'color': '#FF8844'}
]

for impact in impacts:
    conn = ConnectionPatch(earth_pos, impact['pos'], 'data', 'data',
                           arrowstyle='->', color=impact['color'],
                           alpha=0.7, linewidth=2, zorder=20)
    ax_main.add_patch(conn)
    ax_main.text(impact['pos'][0], impact['pos'][1],
                 impact['label'],
                 fontsize=22, ha='left', va='center', color=impact['color'],
                 linespacing=1.2, zorder=30,
                 bbox=dict(facecolor='white', edgecolor=impact['color'],
                           boxstyle='round,pad=0.5'))

# HELIOS Timeline -------------------------------------------------------------
ax_timeline = fig.add_axes([0.15, 0.08, 0.7, 0.12], facecolor='white')
ax_timeline.set_xlim(-36, 0)
ax_timeline.set_ylim(0, 1)
ax_timeline.axis('off')

# Timeline elements
timeline_y = 0.5
ax_timeline.plot([-36, 0], [timeline_y] * 2, color='black', lw=3)

# Time markers
for hours, label in [(-24, '24h'), (-18, '18h'), (-12, '12h'), (-6, '6h'), (0, 'Impact')]:
    ax_timeline.plot([hours] * 2, [timeline_y - 0.1, timeline_y + 0.1], color='black', lw=2)
    ax_timeline.text(hours, timeline_y + 0.15, label, color='black', ha='center', fontsize=25)

# Prediction ranges
helios_arrow = FancyArrowPatch((-36, 0.7), (-12, 0.7),
                               arrowstyle='wedge,tail_width=0.7',
                               color='#4CAF50', lw=0)
ax_timeline.add_patch(helios_arrow)
ax_timeline.text(-24, 0.15, 'HELIOS Prediction Window', color='#4CAF50',
                 ha='center', fontsize=22)

current_arrow = FancyArrowPatch((-12, 0.3), (0, 0.3),
                                arrowstyle='wedge,tail_width=0.7',
                                color='#F44336', lw=0)
ax_timeline.add_patch(current_arrow)
ax_timeline.text(-6, 0.15, 'Current Systems', color='#F44336',
                 ha='center', fontsize=22)

# Protective measures text
ax_timeline.plot([-12] * 2, [0, 1], color='#2196F3', linestyle='--', lw=2)
ax_timeline.text(-6, 1.1, 'Minimum Response Time for\nProtection Measures',  # Adjusted x and y coordinates
                 color='#2196F3', fontsize=15, ha='center', va='center')  # Right-aligned

# Titles and Text -------------------------------------------------------------
ax_main.text(0, 8.5, 'Solar Flare Impact Pathways on Terrestrial Infrastructure',
             color='black', fontsize=34, ha='center', va='center', weight='bold')
ax_main.text(0, 7.5, 'Demonstrating HELIOS Extended Warning System Effectiveness',
             color='#1a53ff', fontsize=28, ha='center', va='center', style='italic')

# Data sources
ax_main.text(-11, -7, 'Data Sources: NASA Solar Dynamics Observatory | NOAA SWPC\n'
                      'Simulation Parameters: MHD Model (ENLIL v3.2)',
             color='#444444', fontsize=20)

plt.savefig('purpose.png', bbox_inches='tight', dpi=300)
plt.close()