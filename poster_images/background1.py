import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.patheffects as path_effects

# Set up the figure with a white background and professional styling
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor='white')
fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.88)

# Flatten the axes for easier iteration
axes = axes.flatten()

# Define all AIA wavelengths and associated information
wavelengths = [
    "94 Å\n6,000,000 K\nFlaring Regions",
    "131 Å\n10,000,000 K\nFlaring Plasma",
    "171 Å\n600,000 K\nCorona/Transition Region",
    "193 Å\n1,000,000 K\nCorona/Hot Loops",
    "211 Å\n2,000,000 K\nActive Regions",
    "304 Å\n50,000 K\nChromosphere",
    "335 Å\n2,500,000 K\nActive Regions/Loops",
    "1600 Å\n5,000 K\nLower Atmosphere",
    "1700 Å\n5,000 K\nPhotosphere (UV Continuum)",
    "4500 Å\nVisible Light\nPhotospheric Continuum"
]

# Define colormaps for the wavelengths
colormaps = [
    cm.Greens,  # 94 Å
    cm.hot,  # 131 Å
    cm.Blues,  # 171 Å
    cm.YlOrBr,  # 193 Å
    cm.pink,  # 211 Å
    cm.Reds,  # 304 Å
    cm.Purples,  # 335 Å
    cm.Oranges,  # 1600 Å
    cm.YlGn,  # 1700 Å
    cm.gray  # 4500 Å
]

# Random seed for reproducibility
np.random.seed(42)

# Create the grid for simulation
grid_size = 256
x, y = np.meshgrid(np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size))
r = np.sqrt(x ** 2 + y ** 2)
base = np.exp(-0.5 * r ** 2)  # common base profile

# Generate simulated active region data for each wavelength
for i, (wavelength, cmap) in enumerate(zip(wavelengths, colormaps)):
    data = None  # initialize data for each wavelength

    if "94" in wavelength:
        # Flaring regions: bright core and additional small-scale brightenings
        data = base + 2 * np.exp(-0.5 * ((r - 1.5) / 0.3) ** 2) * (r < 2.5)
        for _ in range(5):
            x_pos = np.random.uniform(-3, 3)
            y_pos = np.random.uniform(-3, 3)
            r_local = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
            data += 2 * np.exp(-0.5 * (r_local / 0.2) ** 2)

    elif "131" in wavelength:
        # Hot plasma loops
        data = base * 0.7
        for _ in range(3):
            theta = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.1, 0.3)
            r_loop = np.sqrt((x - np.cos(theta)) ** 2 + (y - np.sin(theta)) ** 2)
            data += 1.5 * np.exp(-0.5 * (r_loop / width) ** 2)

    elif "171" in wavelength:
        # Cooler coronal loops
        data = base * 0.5
        for j in range(7):
            theta = j * np.pi / 4
            width = 0.15 + 0.05 * np.sin(j)
            height = 2 + 0.5 * np.cos(j)
            amp = 0.8 + 0.3 * np.random.random()
            data += amp * np.exp(-0.5 * ((x * np.cos(theta) + y * np.sin(theta)) / height) ** 2
                                 - 0.5 * ((y * np.cos(theta) - x * np.sin(theta)) / width) ** 2)

    elif "193" in wavelength:
        # Corona with hot loops and moss
        data = base * 0.6
        for j in range(5):
            theta = j * np.pi / 3
            width = 0.2 + 0.1 * np.sin(j)
            height = 2.5 + 0.7 * np.cos(j)
            amp = 0.9 + 0.4 * np.random.random()
            data += amp * np.exp(-0.5 * ((x * np.cos(theta) + y * np.sin(theta)) / height) ** 2
                                 - 0.5 * ((y * np.cos(theta) - x * np.sin(theta)) / width) ** 2)

    elif "211" in wavelength:
        # Active region loops with radial shells
        data = base * 0.8
        for j in range(4):
            r_shell = np.abs(r - (1.5 + 0.5 * j))
            data += (0.7 - 0.1 * j) * np.exp(-0.5 * (r_shell / 0.3) ** 2)

    elif "304" in wavelength:
        # Chromosphere: textured with many small-scale features
        data = base * 0.7
        for _ in range(100):
            x_pos = np.random.uniform(-4, 4)
            y_pos = np.random.uniform(-4, 4)
            feature_size = np.random.uniform(0.05, 0.2)
            r_local = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
            data += 0.3 * np.exp(-0.5 * (r_local / feature_size) ** 2)

    elif "335" in wavelength:
        # 335 Å: Active region loops with a slightly different morphology
        data = base * 0.65
        for j in range(6):
            theta = j * np.pi / 3
            width = 0.25 + 0.1 * np.sin(j)
            height = 2.0 + 0.5 * np.cos(j)
            amp = 1.0 + 0.3 * np.random.random()
            data += amp * np.exp(-0.5 * ((x * np.cos(theta) + y * np.sin(theta)) / height) ** 2
                                 - 0.5 * ((y * np.cos(theta) - x * np.sin(theta)) / width) ** 2)

    elif "1600" in wavelength:
        # 1600 Å: Lower atmosphere, with network-like small-scale brightenings
        data = base * 0.8
        for _ in range(50):
            x_pos = np.random.uniform(-4, 4)
            y_pos = np.random.uniform(-4, 4)
            feature_size = np.random.uniform(0.1, 0.4)
            r_local = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
            data += 0.5 * np.exp(-0.5 * (r_local / feature_size) ** 2)

    elif "1700" in wavelength:
        # 1700 Å: Photospheric UV continuum; simulate granulation with many small cells
        data = base * 0.9
        for _ in range(70):
            x_pos = np.random.uniform(-4.5, 4.5)
            y_pos = np.random.uniform(-4.5, 4.5)
            feature_size = np.random.uniform(0.05, 0.2)
            r_local = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
            data += 0.4 * np.exp(-0.5 * (r_local / feature_size) ** 2)

    elif "4500" in wavelength:
        # 4500 Å: Visible continuum; a smoother granulation pattern
        data = base * 1.0
        for _ in range(80):
            x_pos = np.random.uniform(-5, 5)
            y_pos = np.random.uniform(-5, 5)
            feature_size = np.random.uniform(0.1, 0.3)
            r_local = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
            data += 0.3 * np.exp(-0.5 * (r_local / feature_size) ** 2)

    # Add a slight noise characteristic to each wavelength
    noise_level = 0.05 + 0.02 * i
    data += np.random.normal(0, noise_level, size=(grid_size, grid_size))
    data = np.clip(data, 0, None)  # Ensure no negative values

    # Display the image with appropriate colormap and logarithmic normalization
    im = axes[i].imshow(data, cmap=cmap, norm=LogNorm(vmin=0.01, vmax=data.max() * 1.2),
                        extent=[-5, 5, -5, 5])

    # Use the subplot title for labeling; the title appears above each image
    axes[i].set_title(wavelength, fontsize=14, color='black', fontweight='bold', pad=10)
    # Add a white background to the title to ensure clarity
    axes[i].title.set_bbox(dict(facecolor='white', edgecolor='none', pad=3, alpha=0.8))

    # Remove axis ticks and labels for a clean presentation
    axes[i].set_xticks([])
    axes[i].set_yticks([])

    # Set the axes background to white and add a subtle border
    axes[i].set_facecolor('white')
    for spine in axes[i].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

# Add an overall title to the figure in black text
fig.suptitle('Figure 3: Multi-Wavelength View of a Solar Active Region\nRevealing Temperature-Dependent Plasma Structures',
             color='black', fontsize=26, y=0.96)

# Add a source note at the bottom
fig.text(0.5, 0.01, 'Simulated data based on SDO/AIA observations', ha='center', fontsize=16, color='gray')

plt.tight_layout()
plt.savefig('background1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
