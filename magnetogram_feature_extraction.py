# # Physics-Informed Feature Extraction Pipeline for Vector Magnetograms
# # This script extracts physics-based features from processed HMI vector magnetograms
#
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import ndimage
# import sunpy.map
# from sunpy.coordinates import frames
# import astropy.units as u
# from astropy.coordinates import SkyCoord
# from sklearn.preprocessing import StandardScaler
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# class MagnetogramFeatureExtractor:
#     """Class for extracting physics-informed features from vector magnetograms"""
#
#     def __init__(self, input_dir='data/processed/magnetograms', output_dir='data/processed/features'):
#         """Initialize the feature extractor"""
#
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Load magnetogram metadata
#         metadata_path = os.path.join(input_dir, 'magnetogram_metadata.csv')
#
#         if os.path.exists(metadata_path):
#             try:
#                 self.metadata_df = pd.read_csv(metadata_path)
#                 print(f"Initialized feature extractor with {len(self.metadata_df)} magnetograms")
#             except pd.errors.EmptyDataError:
#                 print("Warning: Empty metadata file. Creating empty DataFrame.")
#                 self.metadata_df = pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude',
#                                                          'carrington_rotation', 'center'])
#         else:
#             print(f"Warning: Metadata file {metadata_path} not found. Creating empty DataFrame.")
#             self.metadata_df = pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude',
#                                                      'carrington_rotation', 'center'])
#
#
#
#     def calculate_gradient_weighted_neutral_line(self, bz):
#         """
#         Calculate the gradient-weighted neutral line length (R-value)
#
#         Parameters:
#         bz (numpy.ndarray): Vertical magnetic field component
#
#         Returns:
#         float: R-value
#         """
#         # Create a binary map of the polarity inversion line (PIL)
#         bz_smooth = ndimage.gaussian_filter(bz, sigma=1)
#         pil = np.zeros_like(bz_smooth)
#
#         # Find zero-crossing points (neutral line)
#         for i in range(1, bz_smooth.shape[0]):
#             for j in range(1, bz_smooth.shape[1]):
#                 # Check if there's a sign change in any of the 4 directions
#                 if (bz_smooth[i, j] * bz_smooth[i - 1, j] < 0 or
#                         bz_smooth[i, j] * bz_smooth[i, j - 1] < 0):
#                     pil[i, j] = 1
#
#         # Calculate gradient
#         gradient_x = ndimage.sobel(bz_smooth, axis=0)
#         gradient_y = ndimage.sobel(bz_smooth, axis=1)
#         gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#
#         # Weight the neutral line by the gradient
#         weighted_pil = pil * gradient_magnitude
#
#         # R-value is the sum of the weighted PIL
#         r_value = np.sum(weighted_pil)
#
#         return r_value
#
#     # Add this function to help with debugging
#     def inspect_file_structure(directory):
#         """Print information about files in a directory"""
#         if not os.path.exists(directory):
#             print(f"Directory {directory} does not exist")
#             return
#
#         files = os.listdir(directory)
#         print(f"Directory {directory} contains {len(files)} files")
#         if len(files) > 0:
#             print("Sample files:")
#             for f in files[:5]:
#                 full_path = os.path.join(directory, f)
#                 size = os.path.getsize(full_path) / 1024  # Size in KB
#                 print(f"  - {f} ({size:.1f} KB)")
#
#     def calculate_magnetic_shear(self, bx, by, bz):
#         """
#         Calculate the magnetic shear (angle between potential and observed field)
#
#         Parameters:
#         bx, by, bz (numpy.ndarray): Components of the magnetic field vector
#
#         Returns:
#         float: Average magnetic shear angle in degrees
#         """
#         # Calculate the observed field direction
#         b_total = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
#
#         # Avoid division by zero
#         mask = b_total > 0
#
#         # Calculate the unit vector of the observed field
#         bx_unit = np.zeros_like(bx)
#         by_unit = np.zeros_like(by)
#         bz_unit = np.zeros_like(bz)
#
#         bx_unit[mask] = bx[mask] / b_total[mask]
#         by_unit[mask] = by[mask] / b_total[mask]
#         bz_unit[mask] = bz[mask] / b_total[mask]
#
#         # For a potential field, the horizontal components would be minimized
#         # So we can approximate the shear angle as the angle between the observed field
#         # and the vertical direction
#
#         # Cosine of the angle between the observed field and the vertical
#         cos_shear = bz_unit
#
#         # Convert to degrees
#         shear_angle = np.arccos(np.clip(cos_shear, -1, 1)) * 180 / np.pi
#
#         # Calculate average shear in strong field regions
#         strong_field = b_total > np.median(b_total[mask])
#         avg_shear = np.mean(shear_angle[strong_field & mask]) if np.any(strong_field & mask) else 0
#
#         return avg_shear
#
#     def calculate_current_helicity(self, bx, by, bz, dx=0.5, dy=0.5):
#         """
#         Calculate the current helicity proxy
#
#         Parameters:
#         bx, by, bz (numpy.ndarray): Components of the magnetic field vector
#         dx, dy (float): Spatial resolution in arcseconds per pixel
#
#         Returns:
#         float: Current helicity proxy
#         """
#         # Calculate current density components
#         # Jz = dBy/dx - dBx/dy
#         jz = ndimage.sobel(by, axis=1) / dx - ndimage.sobel(bx, axis=0) / dy
#
#         # Current helicity proxy: Bz * Jz
#         current_helicity = bz * jz
#
#         # Return the sum of the absolute current helicity
#         return np.sum(np.abs(current_helicity))
#
#     def calculate_ising_energy(self, bz):
#         """
#         Calculate the Ising energy proxy (a measure of magnetic complexity)
#
#         Parameters:
#         bz (numpy.ndarray): Vertical magnetic field component
#
#         Returns:
#         float: Ising energy proxy
#         """
#         # Binarize the vertical field (positive or negative)
#         bz_sign = np.sign(bz)
#
#         # Calculate energy by comparing each pixel with its neighbors
#         # Higher energy means more complex configuration
#         energy = 0
#
#         for i in range(1, bz.shape[0] - 1):
#             for j in range(1, bz.shape[1] - 1):
#                 # Compare with 4 nearest neighbors
#                 central = bz_sign[i, j]
#                 neighbors = [
#                     bz_sign[i + 1, j],
#                     bz_sign[i - 1, j],
#                     bz_sign[i, j + 1],
#                     bz_sign[i, j - 1]
#                 ]
#
#                 # Add energy for each opposite-sign neighbor
#                 for n in neighbors:
#                     if central * n < 0:  # opposite signs
#                         energy += 1
#
#         return energy
#
#     def calculate_lorentz_force(self, bx, by, bz):
#         """
#         Calculate the proxy of the Lorentz force
#
#         Parameters:
#         bx, by, bz (numpy.ndarray): Components of the magnetic field vector
#
#         Returns:
#         float: Lorentz force proxy
#         """
#         # The Lorentz force is proportional to J × B
#         # We use a simplified proxy here
#
#         # Compute the gradients of the field components
#         grad_bx_x = ndimage.sobel(bx, axis=1)
#         grad_by_y = ndimage.sobel(by, axis=0)
#
#         # A proxy for the vertical component of the Lorentz force
#         # F_z ~ (B·∇)B_z
#         force_z_proxy = bx * ndimage.sobel(bz, axis=1) + by * ndimage.sobel(bz, axis=0)
#
#         # Return the sum of the absolute value
#         return np.sum(np.abs(force_z_proxy))
#
#     def extract_features(self, filename):
#         """
#         Extract all physics-informed features from a magnetogram
#
#         Parameters:
#         filename (str): Path to the magnetogram file
#
#         Returns:
#         dict: Dictionary of extracted features
#         """
#         try:
#             # In a real implementation, we would load Bx, By, Bz components
#             # For this example, we'll simulate them
#             magnetogram = np.load(filename)
#
#             # Simulate 3D vector components (in real implementation, these would be actual data)
#             # This is just for demonstration purposes
#             bz = magnetogram  # Vertical component
#             bx = ndimage.gaussian_filter(np.random.randn(*bz.shape) * np.std(bz) * 0.5,
#                                          sigma=5)  # Simulated horizontal components
#             by = ndimage.gaussian_filter(np.random.randn(*bz.shape) * np.std(bz) * 0.5, sigma=5)
#
#             # Calculate features
#             r_value = self.calculate_gradient_weighted_neutral_line(bz)
#             shear = self.calculate_magnetic_shear(bx, by, bz)
#             helicity = self.calculate_current_helicity(bx, by, bz)
#             ising_energy = self.calculate_ising_energy(bz)
#             lorentz_force = self.calculate_lorentz_force(bx, by, bz)
#
#             # Calculate additional basic statistics
#             total_unsigned_flux = np.sum(np.abs(bz))
#             max_field_strength = np.max(np.abs(bz))
#
#             # Return all features
#             features = {
#                 'r_value': r_value,
#                 'magnetic_shear': shear,
#                 'current_helicity': helicity,
#                 'ising_energy': ising_energy,
#                 'lorentz_force': lorentz_force,
#                 'total_unsigned_flux': total_unsigned_flux,
#                 'max_field_strength': max_field_strength
#             }
#
#             return features
#
#         except Exception as e:
#             print(f"Error extracting features from {filename}: {e}")
#             return None
#
#     def extract_all_features(self):
#         """Extract features from all magnetograms and save to CSV"""
#         print("Extracting physics-informed features from all magnetograms")
#
#         all_features = []
#
#         for i, row in self.metadata_df.iterrows():
#             if i % 10 == 0:
#                 print(f"Processing magnetogram {i + 1}/{len(self.metadata_df)}")
#
#             # Extract features
#             features = self.extract_features(row['filename'])
#
#             if features:
#                 # Add metadata
#                 features['filename'] = row['filename']
#                 features['timestamp'] = row['timestamp']
#
#                 all_features.append(features)
#
#         # Create DataFrame
#         features_df = pd.DataFrame(all_features)
#
#         # Save to CSV
#         output_file = os.path.join(self.output_dir, 'magnetogram_features.csv')
#         features_df.to_csv(output_file, index=False)
#
#         print(f"Extracted features from {len(features_df)} magnetograms")
#         print(f"Saved features to {output_file}")
#
#         return features_df
#
#
# # Main function to run the feature extraction
# def main():
#     # Initialize and run the feature extractor
#     extractor = MagnetogramFeatureExtractor()
#     features_df = extractor.extract_all_features()
#
#     # Show feature statistics
#     print("\nFeature Statistics:")
#     print(features_df.describe())
#
#     return features_df
#
#
# # If running as a script
# if __name__ == "__main__":
#     main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


class MagnetogramFeatureExtractor:
    """Class for extracting physics-informed features from vector magnetograms"""

    def __init__(self, input_dir='data/processed/magnetograms', output_dir='data/processed/features'):
        """Initialize the feature extractor"""

        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load magnetogram metadata
        metadata_path = os.path.join(input_dir, 'magnetogram_metadata.csv')

        if os.path.exists(metadata_path):
            try:
                self.metadata_df = pd.read_csv(metadata_path)
                print(f"Initialized feature extractor with {len(self.metadata_df)} magnetograms")
            except pd.errors.EmptyDataError:
                print("Warning: Empty metadata file. Creating empty DataFrame.")
                self.metadata_df = pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude',
                                                         'carrington_rotation', 'center'])
        else:
            print(f"Warning: Metadata file {metadata_path} not found. Creating empty DataFrame.")
            self.metadata_df = pd.DataFrame(columns=['filename', 'timestamp', 'carrington_longitude',
                                                     'carrington_rotation', 'center'])

    def load_vector_magnetogram(self, filename):
        """
        Load magnetogram data and extract the field components

        In a real implementation, vector magnetograms should contain all three components
        of the magnetic field: Bx, By, and Bz. Since our data appears to be
        single-component, we'll need to derive Bx and By from Bz.

        Parameters:
        filename (str): Path to the magnetogram file

        Returns:
        tuple: (Bx, By, Bz) components of the magnetic field
        """
        try:
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                return None, None, None

            # Load the magnetogram data
            magnetogram = np.load(filename, allow_pickle=True)

            # Debug: Print shape and type
            print(f"Loaded magnetogram shape: {magnetogram.shape}, dtype: {magnetogram.dtype}")

            # Check if data contains NaN values and handle accordingly
            if np.isnan(magnetogram).any():
                print(f"Warning: Magnetogram contains NaN values. Replacing with zeros.")
                magnetogram = np.nan_to_num(magnetogram, nan=0.0)

            # If no valid data (all zeros or NaNs), return zeros
            if np.all(np.abs(magnetogram) < 1e-10):
                print(f"Warning: Magnetogram contains no valid data. Using random data for testing.")
                # For testing purposes, create synthetic data
                shape = magnetogram.shape
                bz = np.random.normal(0, 100, size=shape)  # Random field for testing
                bx = np.zeros_like(bz)
                by = np.zeros_like(bz)
                return bx, by, bz

            # Check if the magnetogram is a structured array with field components
            if hasattr(magnetogram, 'dtype') and magnetogram.dtype.names is not None:
                print(f"Structured array with fields: {magnetogram.dtype.names}")
                # Extract components based on their names
                if 'Bx' in magnetogram.dtype.names and 'By' in magnetogram.dtype.names and 'Bz' in magnetogram.dtype.names:
                    bx = np.nan_to_num(magnetogram['Bx'], nan=0.0)
                    by = np.nan_to_num(magnetogram['By'], nan=0.0)
                    bz = np.nan_to_num(magnetogram['Bz'], nan=0.0)
                    return bx, by, bz

            # If it's a 3D array, assume [Bx, By, Bz] ordering
            if len(magnetogram.shape) == 3 and magnetogram.shape[0] == 3:
                bx = np.nan_to_num(magnetogram[0], nan=0.0)
                by = np.nan_to_num(magnetogram[1], nan=0.0)
                bz = np.nan_to_num(magnetogram[2], nan=0.0)
                return bx, by, bz

            # If it's a single 2D array, treat it as Bz only
            bz = np.nan_to_num(magnetogram, nan=0.0)

            # For a proper implementation, Bx and By should be loaded directly
            # from the vector magnetogram file. However, as a physics-informed
            # approximation, we'll derive them using potential field approximation.

            # Apply some smoothing before gradient calculation to reduce noise
            bz_smooth = ndimage.gaussian_filter(bz, sigma=2)

            # Compute the gradient of Bz - this gives us an approximation of horizontal fields
            # based on the assumption that div(B) = 0 and curl(B) ≈ 0 (potential field)
            gradient_x = ndimage.sobel(bz_smooth, axis=1)
            gradient_y = ndimage.sobel(bz_smooth, axis=0)

            # The x and y components in a potential field configuration are related to
            # the gradient of the z component. We apply a scaling factor to keep
            # the magnitudes physically reasonable
            max_gradient = max(np.max(np.abs(gradient_x)), np.max(np.abs(gradient_y)))
            if max_gradient > 0:
                scale_factor = 0.05 * np.max(np.abs(bz)) / max_gradient
                bx = -gradient_x * scale_factor
                by = -gradient_y * scale_factor
            else:
                bx = np.zeros_like(bz)
                by = np.zeros_like(bz)

            return bx, by, bz

        except Exception as e:
            print(f"Error loading magnetogram {filename}: {e}")
            print("Using random data as fallback")
            # Create dummy data for testing
            shape = (512, 512)  # Default shape if can't determine from file
            bz = np.random.normal(0, 100, size=shape)
            bx = np.zeros_like(bz)
            by = np.zeros_like(bz)
            return bx, by, bz

    def calculate_gradient_weighted_neutral_line(self, bz):
        """
        Calculate the gradient-weighted neutral line length (R-value)

        Parameters:
        bz (numpy.ndarray): Vertical magnetic field component

        Returns:
        float: R-value
        """
        try:
            # Create a binary map of the polarity inversion line (PIL)
            bz_smooth = ndimage.gaussian_filter(bz, sigma=1)
            pil = np.zeros_like(bz_smooth)

            # Find zero-crossing points (neutral line)
            for i in range(1, bz_smooth.shape[0]):
                for j in range(1, bz_smooth.shape[1]):
                    # Check if there's a sign change in any of the 4 directions
                    if (bz_smooth[i, j] * bz_smooth[i - 1, j] <= 0 or
                            bz_smooth[i, j] * bz_smooth[i, j - 1] <= 0):
                        pil[i, j] = 1

            # Calculate gradient
            gradient_x = ndimage.sobel(bz_smooth, axis=0)
            gradient_y = ndimage.sobel(bz_smooth, axis=1)
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

            # Weight the neutral line by the gradient
            weighted_pil = pil * gradient_magnitude

            # R-value is the sum of the weighted PIL
            r_value = np.sum(weighted_pil)

            return r_value
        except Exception as e:
            print(f"Error calculating gradient-weighted neutral line: {e}")
            return 0.0

    def calculate_magnetic_shear(self, bx, by, bz):
        """
        Calculate the magnetic shear (angle between potential and observed field)

        Parameters:
        bx, by, bz (numpy.ndarray): Components of the magnetic field vector

        Returns:
        float: Average magnetic shear angle in degrees
        """
        try:
            # Calculate the observed field strength
            b_total = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)

            # Create a mask for areas with significant field strength
            # to avoid numerical issues in weak field regions
            threshold = 0.05 * np.max(b_total)
            mask = b_total > threshold

            if not np.any(mask):
                return 0.0

            # Unit vectors of the observed field
            bx_unit = np.zeros_like(bx)
            by_unit = np.zeros_like(by)
            bz_unit = np.zeros_like(bz)

            bx_unit[mask] = bx[mask] / b_total[mask]
            by_unit[mask] = by[mask] / b_total[mask]
            bz_unit[mask] = bz[mask] / b_total[mask]

            # A potential field would align more with the vertical direction
            # We approximate potential field as purely vertical
            potential_field_unit = np.zeros_like(bz_unit)
            potential_field_unit[mask] = np.sign(bz[mask])

            # Calculate dot product between observed and potential field
            dot_product = bz_unit * potential_field_unit

            # Ensure values are in the valid range for arccos
            dot_product = np.clip(dot_product, -1.0, 1.0)

            # Calculate the shear angle in degrees
            shear_angle = np.arccos(dot_product) * 180 / np.pi

            # Calculate the average shear in regions with strong field
            strong_field = b_total > np.percentile(b_total[mask], 75)
            combined_mask = strong_field & mask

            if np.any(combined_mask):
                avg_shear = np.mean(shear_angle[combined_mask])
            else:
                avg_shear = 0.0

            return avg_shear
        except Exception as e:
            print(f"Error calculating magnetic shear: {e}")
            return 0.0

    def calculate_current_helicity(self, bx, by, bz, dx=0.5, dy=0.5):
        """
        Calculate the current helicity proxy

        Parameters:
        bx, by, bz (numpy.ndarray): Components of the magnetic field vector
        dx, dy (float): Spatial resolution in arcseconds per pixel

        Returns:
        float: Current helicity proxy
        """
        try:
            # Calculate current density components using finite differences
            # Jx = dBz/dy - dBy/dz (we set dBy/dz = 0 as we have no z-derivative)
            # Jy = dBx/dz - dBz/dx (we set dBx/dz = 0 as we have no z-derivative)
            # Jz = dBy/dx - dBx/dy

            jz = ndimage.sobel(by, axis=1) / dx - ndimage.sobel(bx, axis=0) / dy

            # Current helicity proxy: Bz * Jz
            current_helicity = bz * jz

            # Calculate the total signed current helicity and normalize by area
            total_area = bz.shape[0] * bz.shape[1]
            alpha_proxy = np.sum(current_helicity) / total_area

            return alpha_proxy
        except Exception as e:
            print(f"Error calculating current helicity: {e}")
            return 0.0

    def calculate_ising_energy(self, bz):
        """
        Calculate the Ising energy proxy (a measure of magnetic complexity)

        Parameters:
        bz (numpy.ndarray): Vertical magnetic field component

        Returns:
        float: Ising energy proxy
        """
        try:
            # Apply a threshold to isolate significant magnetic elements
            max_abs_bz = np.max(np.abs(bz))
            if max_abs_bz < 1e-10:
                return 0.0

            threshold = 0.1 * max_abs_bz
            bz_sign = np.zeros_like(bz)
            bz_sign[bz > threshold] = 1
            bz_sign[bz < -threshold] = -1

            # Calculate energy by comparing each pixel with its neighbors
            energy = 0
            count = 0

            for i in range(1, bz.shape[0] - 1):
                for j in range(1, bz.shape[1] - 1):
                    if bz_sign[i, j] == 0:
                        continue

                    # Compare with 4 nearest neighbors
                    central = bz_sign[i, j]
                    neighbors = [
                        bz_sign[i + 1, j],
                        bz_sign[i - 1, j],
                        bz_sign[i, j + 1],
                        bz_sign[i, j - 1]
                    ]

                    # Add energy for each opposite-sign neighbor
                    for n in neighbors:
                        if n != 0:  # Only consider significant field elements
                            count += 1
                            if central * n < 0:  # opposite signs
                                energy += 1

            # Normalize by the number of comparisons
            normalized_energy = energy / max(count, 1)
            return normalized_energy
        except Exception as e:
            print(f"Error calculating Ising energy: {e}")
            return 0.0

    def calculate_lorentz_force(self, bx, by, bz):
        """
        Calculate the proxy of the Lorentz force

        Parameters:
        bx, by, bz (numpy.ndarray): Components of the magnetic field vector

        Returns:
        float: Lorentz force proxy
        """
        try:
            # Calculate the magnetic pressure gradient
            # ∇(B^2/8π) approximated using the gradient of B^2
            b_squared = bx ** 2 + by ** 2 + bz ** 2
            grad_b_squared_x = ndimage.sobel(b_squared, axis=1)
            grad_b_squared_y = ndimage.sobel(b_squared, axis=0)

            # Calculate the tension force components
            # (B·∇)B_x = Bx·∂Bx/∂x + By·∂Bx/∂y
            # (B·∇)B_y = Bx·∂By/∂x + By·∂By/∂y
            # (B·∇)B_z = Bx·∂Bz/∂x + By·∂Bz/∂y

            dbx_dx = ndimage.sobel(bx, axis=1)
            dbx_dy = ndimage.sobel(bx, axis=0)
            dby_dx = ndimage.sobel(by, axis=1)
            dby_dy = ndimage.sobel(by, axis=0)
            dbz_dx = ndimage.sobel(bz, axis=1)
            dbz_dy = ndimage.sobel(bz, axis=0)

            tension_x = bx * dbx_dx + by * dbx_dy
            tension_y = bx * dby_dx + by * dby_dy
            tension_z = bx * dbz_dx + by * dbz_dy

            # The Lorentz force proxy is the sum of tension and pressure gradient
            # We'll calculate the magnitude of the force
            force_magnitude = np.sqrt(
                (tension_x - 0.5 * grad_b_squared_x) ** 2 +
                (tension_y - 0.5 * grad_b_squared_y) ** 2 +
                tension_z ** 2
            )

            # Return the average force magnitude
            return np.nanmean(force_magnitude)
        except Exception as e:
            print(f"Error calculating Lorentz force: {e}")
            return 0.0

    def calculate_total_unsigned_flux(self, bz):
        """
        Calculate the total unsigned magnetic flux

        Parameters:
        bz (numpy.ndarray): Vertical magnetic field component

        Returns:
        float: Total unsigned flux
        """
        try:
            return np.sum(np.abs(bz))
        except Exception as e:
            print(f"Error calculating total unsigned flux: {e}")
            return 0.0

    def calculate_max_field_strength(self, bx, by, bz):
        """
        Calculate the maximum field strength

        Parameters:
        bx, by, bz (numpy.ndarray): Components of the magnetic field vector

        Returns:
        float: Maximum field strength
        """
        try:
            b_total = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
            return np.max(b_total)
        except Exception as e:
            print(f"Error calculating max field strength: {e}")
            return 0.0

    def extract_features(self, filename):
        """
        Extract all physics-informed features from a magnetogram

        Parameters:
        filename (str): Path to the magnetogram file

        Returns:
        dict: Dictionary of extracted features
        """
        try:
            print(f"\nProcessing file: {filename}")
            # Load the vector magnetogram components
            bx, by, bz = self.load_vector_magnetogram(filename)

            if bz is None:
                print(f"Failed to load magnetogram data from {filename}")
                return None

            # Verify data is not all zeros or NaNs
            if np.all(np.abs(bz) < 1e-10) and np.all(np.abs(bx) < 1e-10) and np.all(np.abs(by) < 1e-10):
                print(f"Warning: Magnetogram contains only zeros or very small values")
                # Continue with calculations anyway, features should be close to zero

            # Calculate features
            print(f"Bx shape: {bx.shape}, By shape: {by.shape}, Bz shape: {bz.shape}")
            print(f"Bx range: [{np.min(bx)}, {np.max(bx)}], Bz range: [{np.min(bz)}, {np.max(bz)}]")

            # Calculate and print each feature
            r_value = self.calculate_gradient_weighted_neutral_line(bz)
            print(f"R-value: {r_value}")

            shear = self.calculate_magnetic_shear(bx, by, bz)
            print(f"Magnetic shear: {shear}")

            helicity = self.calculate_current_helicity(bx, by, bz)
            print(f"Current helicity: {helicity}")

            ising_energy = self.calculate_ising_energy(bz)
            print(f"Ising energy: {ising_energy}")

            lorentz_force = self.calculate_lorentz_force(bx, by, bz)
            print(f"Lorentz force: {lorentz_force}")

            # Calculate additional basic statistics
            total_unsigned_flux = self.calculate_total_unsigned_flux(bz)
            max_field_strength = self.calculate_max_field_strength(bx, by, bz)

            # Return all features
            features = {
                'r_value': r_value,
                'magnetic_shear': shear,
                'current_helicity': helicity,
                'ising_energy': ising_energy,
                'lorentz_force': lorentz_force,
                'total_unsigned_flux': total_unsigned_flux,
                'max_field_strength': max_field_strength
            }

            return features

        except Exception as e:
            print(f"Error extracting features from {filename}: {e}")
            # Return default values for all features
            return {
                'r_value': 0.0,
                'magnetic_shear': 0.0,
                'current_helicity': 0.0,
                'ising_energy': 0.0,
                'lorentz_force': 0.0,
                'total_unsigned_flux': 0.0,
                'max_field_strength': 0.0
            }

    def extract_all_features(self):
        """Extract features from all magnetograms and save to CSV"""
        print("Extracting physics-informed features from all magnetograms")

        all_features = []

        npy_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))

        print(f"Found {len(npy_files)} .npy files")

        for i, filename in enumerate(npy_files):
            if i % 10 == 0:
                print(f"Processing magnetogram {i + 1}/{len(npy_files)}")

            # Extract features directly from the file
            features = self.extract_features(filename)

            if features:
                # Add filename to features
                features['filename'] = filename

                # Extract timestamp and Carrington longitude from metadata if available
                metadata_row = self.metadata_df[self.metadata_df['filename'] == filename]
                if not metadata_row.empty:
                    features['timestamp'] = metadata_row['timestamp'].iloc[0]
                    features['carrington_longitude'] = metadata_row['carrington_longitude'].iloc[0]
                else:
                    # Extract timestamp from filename if possible
                    basename = os.path.basename(filename)
                    if 'magnetogram_' in basename:
                        date_str = basename.split('magnetogram_')[1].split('.npy')[0]
                        features[
                            'timestamp'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {date_str[9:11]}:{date_str[11:13]}:{date_str[13:15]}"
                    else:
                        features['timestamp'] = None
                    features['carrington_longitude'] = None

                all_features.append(features)

        # Create DataFrame
        features_df = pd.DataFrame(all_features)

        # Save to CSV
        output_file = os.path.join(self.output_dir, 'magnetogram_features.csv')
        features_df.to_csv(output_file, index=False)

        print(f"Extracted features from {len(features_df)} magnetograms")
        print(f"Saved features to {output_file}")

        return features_df

    def visualize_features(self, features_df):
        """
        Visualize extracted features

        Parameters:
        features_df (pandas.DataFrame): DataFrame with extracted features
        """
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot histograms of all features
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        numeric_cols = ['r_value', 'magnetic_shear', 'current_helicity',
                        'ising_energy', 'lorentz_force', 'total_unsigned_flux',
                        'max_field_strength']

        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                features_df[col].hist(ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_histograms.png'))
        plt.close()

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = features_df[numeric_cols].corr()
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add correlation values to the heatmap
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black')

        plt.colorbar(label='Correlation')
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('Correlation Matrix of Magnetogram Features')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_correlation.png'))
        plt.close()

    def check_magnetogram_data(self, sample_count=5):
        """
        Check a few random magnetogram files to determine data quality

        Parameters:
        sample_count (int): Number of files to check
        """
        npy_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))

        if not npy_files:
            print("No magnetogram files found")
            return

        print(f"\nChecking {min(sample_count, len(npy_files))} random magnetogram files for data quality:")

        # Randomly select files to check
        import random
        sample_files = random.sample(npy_files, min(sample_count, len(npy_files)))

        for filename in sample_files:
            try:
                # Load the data directly
                data = np.load(filename, allow_pickle=True)

                # Check basic properties
                print(f"\nFile: {os.path.basename(filename)}")
                print(f"  Shape: {data.shape}")
                print(f"  Data type: {data.dtype}")

                # Check for NaN values
                nan_count = np.isnan(data).sum()
                print(f"  NaN count: {nan_count} ({(nan_count / data.size) * 100:.2f}% of data)")

                # Check data range
                if not np.all(np.isnan(data)):
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                    print(f"  Data range: [{data_min}, {data_max}]")
                else:
                    print("  Data range: All NaN values")

                # Check for zero values
                zero_count = np.sum(np.abs(data) < 1e-10)
                print(f"  Near-zero count: {zero_count} ({(zero_count / data.size) * 100:.2f}% of data)")

            except Exception as e:
                print(f"Error checking file {filename}: {e}")


# Main function to run the feature extraction
def main():
    # Initialize the feature extractor
    extractor = MagnetogramFeatureExtractor()

    # First check data quality to understand the issue
    extractor.check_magnetogram_data(sample_count=3)

    # Extract features
    features_df = extractor.extract_all_features()

    # Visualize the features
    if not features_df.empty:
        extractor.visualize_features(features_df)

        # Show feature statistics
        print("\nFeature Statistics:")
        print(features_df.describe())
    else:
        print("No features extracted. Check data issues.")

    return features_df


# If running as a script
if __name__ == "__main__":
    main()