class TemporalEvolutionModule(nn.Module):
    def __init__(self, feature_dim=256, sequence_length=12, num_layers=3):
        super().__init__()

        # Transformer encoder for sequential magnetogram changes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            feature_dim,
            dropout=0.1,
            max_len=sequence_length
        )

        # Recurrent units for capturing precursor patterns
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # Time-to-event regression head
        self.time_to_event = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

        # Feature fusion
        self.fusion = nn.Linear(feature_dim * 3, feature_dim)

    def forward(self, x, mask=None):
        # x: sequence of features [batch, seq_len, feature_dim]
        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        if mask is None:
            transformer_out = self.transformer_encoder(x)
        else:
            transformer_out = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Recurrent processing
        gru_out, _ = self.gru(x)

        # Extract sequence representations
        transformer_seq = transformer_out[:, -1]  # Last token
        gru_seq = gru_out[:, -1, :feature_dim] + gru_out[:, -1, feature_dim:]  # Bidirectional concat

        # Global attention pool across sequence
        attention_weights = F.softmax(
            torch.bmm(
                transformer_out,
                transformer_seq.unsqueeze(2)
            ).squeeze(), dim=1
        )
        global_context = torch.bmm(
            attention_weights.unsqueeze(1),
            transformer_out
        ).squeeze()

        # Combine features
        combined = torch.cat([transformer_seq, gru_seq, global_context], dim=1)
        output = self.fusion(combined)

        # Time-to-event estimation
        tte = self.time_to_event(combined)

        return output, tte