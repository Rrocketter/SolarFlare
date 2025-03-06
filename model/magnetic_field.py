class MagneticFieldAnalysisModule(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()
        # Vector CNN for processing magnetogram data
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Physics-informed feature extractors
        self.current_helicity_extractor = PhysicsFeatureExtractor(
            feature_type="current_helicity", input_size=256, output_size=64)
        self.magnetic_shear_extractor = PhysicsFeatureExtractor(
            feature_type="magnetic_shear", input_size=256, output_size=64)
        self.lorentz_force_extractor = PhysicsFeatureExtractor(
            feature_type="lorentz_force", input_size=256, output_size=64)
        self.ising_energy_extractor = PhysicsFeatureExtractor(
            feature_type="ising_energy", input_size=256, output_size=64)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, x, metadata=None):
        # x: batch of magnetogram images [B, 1, H, W]
        # metadata: tensor of magnetogram features from CSV [B, n_features]

        # Extract CNN features
        cnn_features = self.encoder(x)

        # Extract physics-based features
        helicity_features = self.current_helicity_extractor(cnn_features, metadata)
        shear_features = self.magnetic_shear_extractor(cnn_features, metadata)
        lorentz_features = self.lorentz_force_extractor(cnn_features, metadata)
        ising_features = self.ising_energy_extractor(cnn_features, metadata)

        # Combine all features
        combined = torch.cat([
            cnn_features,
            helicity_features,
            shear_features,
            lorentz_features,
            ising_features
        ], dim=1)

        # Fuse features
        output = self.fusion(combined)

        return output