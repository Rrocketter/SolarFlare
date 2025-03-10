import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up figure and font size
plt.rcParams.update({'font.size': 22})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Generate sample solar disk data (a smooth Gaussian to mimic the solar disk)
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
solar_data = np.exp(-(X**2 + Y**2)/0.2)

# Conventional Model (Left) - Restricted Field-of-View
ax1.imshow(solar_data, cmap='hot', extent=[-1, 1, -1, 1])
# Define radii based on heliocentric angles (using r = sin(theta))
inner_radius = np.sin(np.deg2rad(30))   # ~0.5: Optimal region (±30°)
outer_radius = np.sin(np.deg2rad(70))   # ~0.94: Maximum usable region (±70°)

# Add the boundaries for the restricted field-of-view
ax1.add_patch(patches.Circle((0, 0), outer_radius, edgecolor='red',
             linewidth=3, fill=False, linestyle='--'))
ax1.add_patch(patches.Circle((0, 0), inner_radius, edgecolor='orange',
             linewidth=3, fill=False, linestyle='-.'))

ax1.set_title('Conventional Models\n(Restricted FOV)', pad=20)
ax1.text(0, -1.3, "• Observations limited to within ±70° of the central meridian\n"
                   "• Highest fidelity within the inner ±30°\n"
                   "• Regional feature tracking and partial disk analyses",
         ha='center', va='center')

# HELIOS Approach (Right) - Full-Disk Analysis
ax2.imshow(solar_data, cmap='hot', extent=[-1, 1, -1, 1])
# Full-disk boundary (using nearly the full radius)
ax2.add_patch(patches.Circle((0, 0), 1.0, edgecolor='lime',
             linewidth=4, fill=False, linestyle='-'))
ax2.set_title('HELIOS Approach\n(Full-Disk Analysis)', pad=20)
ax2.text(0, -1.3, "• Complete solar disk coverage\n"
                   "• Global feature tracking across all latitudes\n"
                   "• Integration of whole-Sun magnetic field data",
         ha='center', va='center')

# Common formatting for both axes
for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Overall figure title
fig.text(0.5, 0.95, 'Figure 2: Comparative Solar Analysis Approaches',
         ha='center', va='center', fontsize=26, fontweight='bold')

# Annotations to indicate the FOV boundaries on the conventional model
ax1.annotate('Optimal\n(±30°)', xy=(inner_radius, 0), xytext=(inner_radius+0.2, 0.2),
             arrowprops=dict(arrowstyle="->", color='orange', lw=2),
             color='orange')
ax1.annotate('Restricted FOV\n(±70° limit)', xy=(outer_radius, 0), xytext=(outer_radius-0.4, 0.4),
             arrowprops=dict(arrowstyle="->", color='red', lw=2),
             color='red')

ax2.annotate('Full-Disk\nCoverage', xy=(0.7, 0.7), xytext=(0.3, 0.3),
             arrowprops=dict(arrowstyle="->", color='white', lw=2),
             color='white')

plt.tight_layout(pad=4.0)
plt.savefig('question.png', dpi=300, bbox_inches='tight')
plt.show()
