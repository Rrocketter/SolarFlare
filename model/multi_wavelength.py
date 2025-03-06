class AIAWavelengthEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class MultiWavelengthModule(nn.Module):
    def __init__(self, wavelengths=[94, 131, 171, 193, 211, 304, 335, 1600], feature_dim=256):
        super().__init__()
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)

        # Parallel CNN streams for each wavelength
        self.wavelength_encoders = nn.ModuleDict({
            f"wave_{wave}": AIAWavelengthEncoder(feature_dim=feature_dim)
            for wave in wavelengths
        })

        # Cross-wavelength attention
        self.cross_attention = MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )

        # Temperature gradient detection
        self.temp_gradient_net = nn.Sequential(
            nn.Linear(feature_dim * self.num_wavelengths, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # Output projection
        self.output_projection = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, wavelength_images, metadata=None):
        # wavelength_images: dictionary of tensors for each wavelength
        # {wave_94: [B,1,H,W], wave_131: [B,1,H,W], ...}

        # Process each wavelength through its encoder
        wavelength_features = []
        for wave in self.wavelengths:
            features = self.wavelength_encoders[f"wave_{wave}"](
                wavelength_images[f"wave_{wave}"]
            )
            wavelength_features.append(features)

        # Stack for cross-attention [num_wavelengths, batch, feature_dim]
        stacked_features = torch.stack(wavelength_features)

        # Apply cross-wavelength attention
        attn_output, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Process temperature gradient
        concatenated = torch.cat([f for f in wavelength_features], dim=1)
        temp_gradient = self.temp_gradient_net(concatenated)

        # Average the attention output across wavelengths
        wavelength_representation = attn_output.mean(dim=0)

        # Combine with temperature gradient features
        output = torch.cat([wavelength_representation, temp_gradient], dim=1)
        output = self.output_projection(output)

        return output