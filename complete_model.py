import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from tqdm import tqdm
import cv2
import os

tfd = tfp.distributions
tfpl = tfp.layers


class SolarFlarePredictor:
    def __init__(self, config=None):
        self.config = {
            # Data parameters
            'magnetogram_shape': (1024, 1024, 1),  # HMI magnetogram dimensions
            'aia_shape': (512, 512, 1),  # Resized AIA images
            'aia_wavelengths': [94, 131, 171, 193, 211, 304, 335, 1600, 1700],
            'sequence_length': 24,  # 12 hours at 30-min cadence
            'batch_size': 8,  # Limited by GPU memory

            # Model hyperparameters
            'cnn_filters': [32, 64, 128, 256],
            'transformer_heads': 8,
            'transformer_dim': 256,
            'transformer_layers': 4,
            'dropout_rate': 0.3,
            'l2_reg': 1e-5,
            'learning_rate': 1e-4,
            'mc_dropout_samples': 20,

            # Training parameters
            'epochs': 100,
            'early_stopping_patience': 10,

            # Task weights for multi-objective loss
            'flare_class_weight': 1.0,
            'cme_prob_weight': 0.8,
            'xray_flux_weight': 0.7,
            'time_to_event_weight': 0.5
        }

        # Update with any user-provided config
        if config:
            self.config.update(config)

        # Initialize model components
        self.magnetogram_encoder = None
        self.aia_wavelength_encoders = {}
        self.temporal_encoder = None
        self.uncertainty_layers = None
        self.full_model = None

        self.evaluator = EnhancedSolarEvaluator(self, self.config)
        self.explainability = EnhancedSolarExplainability(self)

    def evaluate_model(self, test_data):
        """Comprehensive model evaluation with new metrics"""
        # Get predictions
        predictions = self.predict_with_uncertainty(test_data)

        # Extract labels
        test_batches = list(test_data.as_numpy_iterator())
        y_true = [np.concatenate([b[1][i] for b in test_batches], axis=0) for i in range(4)]

        # Calculate basic metrics
        results = {
            'accuracy': accuracy_score(y_true[0], predictions['flare_class_mean'].argmax(axis=1)),
            'tss': calculate_tss(y_true[0], predictions['flare_class_mean'].argmax(axis=1)),
            'bss': self.evaluator.calculate_brier_skill_score(y_true[0], predictions['flare_class_mean']),
            'reliability': self.evaluator.generate_reliability_diagram(y_true[0], predictions['flare_class_mean']),
            'lead_time_analysis': self.evaluator.analyze_lead_time_accuracy(test_data),
            'false_cases': self.evaluator.analyze_false_predictions(test_data)
        }

        # Generate visualizations
        self._plot_reliability_diagram(results['reliability'])
        self._plot_lead_time_analysis(results['lead_time_analysis'])

        return results

    def _plot_reliability_diagram(self, reliability_data):
        """Plot reliability diagram and calibration curves"""
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(reliability_data['prob_pred'], reliability_data['prob_true'],
                 's-', label='Model Output')
        plt.plot(reliability_data['prob_pred_iso'], reliability_data['prob_true_iso'],
                 'r-', label='Isotonic Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        plt.title(f'Reliability Diagram (ECE={reliability_data["ece"]:.3f}, MCE={reliability_data["mce"]:.3f})')
        plt.legend()
        plt.savefig('reliability_diagram.png')
        plt.close()

    def _plot_lead_time_analysis(self, lead_time_data):
        """Plot accuracy metrics vs lead time"""
        plt.figure(figsize=(12, 6))

        # Plot TSS
        plt.subplot(1, 3, 1)
        x = [f"{b[0]}-{b[1]}h" for b in [d['lead_time_range'] for d in lead_time_data]]
        y = [d['tss'] for d in lead_time_data]
        plt.bar(x, y)
        plt.xticks(rotation=45)
        plt.title('TSS vs Lead Time')

        # Plot BSS
        plt.subplot(1, 3, 2)
        y = [d['bss'] for d in lead_time_data]
        plt.bar(x, y)
        plt.xticks(rotation=45)
        plt.title('BSS vs Lead Time')

        # Plot Accuracy
        plt.subplot(1, 3, 3)
        y = [d['accuracy'] for d in lead_time_data]
        plt.bar(x, y)
        plt.xticks(rotation=45)
        plt.title('Accuracy vs Lead Time')

        plt.tight_layout()
        plt.savefig('lead_time_analysis.png')
        plt.close()

    def build_magnetogram_encoder(self):
        """Build the Vector CNN for processing 3D magnetogram data"""
        inputs = layers.Input(shape=self.config['magnetogram_shape'])
        x = inputs

        # Physics-informed feature extraction layers
        for filters in self.config['cnn_filters']:
            x = layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config['l2_reg'])
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=2)(x)

        # Calculate physics-based features
        # Current helicity approximation
        helicity = layers.Conv2D(
            16, kernel_size=5, padding='same', activation='tanh',
            name='helicity_filter'
        )(inputs)
        helicity = layers.GlobalAveragePooling2D()(helicity)

        # Magnetic shear approximation
        shear = layers.Conv2D(
            16, kernel_size=7, padding='same', activation='relu',
            name='shear_filter'
        )(inputs)
        shear = layers.GlobalAveragePooling2D()(shear)

        # Lorentz force proxy
        lorentz = layers.Conv2D(
            16, kernel_size=5, padding='same', activation='relu',
            name='lorentz_filter'
        )(inputs)
        lorentz = layers.GlobalAveragePooling2D()(lorentz)

        # Ising energy proxy
        ising = layers.Conv2D(
            16, kernel_size=3, padding='same', activation='tanh',
            name='ising_filter'
        )(inputs)
        ising = layers.GlobalAveragePooling2D()(ising)

        # Global features
        global_features = layers.GlobalAveragePooling2D()(x)

        # Combine CNN features with physics-based features
        combined = layers.Concatenate()([global_features, helicity, shear, lorentz, ising])

        # Final dense layers
        x = layers.Dense(512, activation='relu')(combined)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        x = layers.Dense(256, activation='relu')(x)

        self.magnetogram_encoder = Model(inputs=inputs, outputs=x, name='magnetogram_encoder')
        return self.magnetogram_encoder

    def build_aia_wavelength_encoder(self, wavelength):
        """Build CNN encoder for a specific AIA wavelength"""
        if wavelength in self.aia_wavelength_encoders:
            return self.aia_wavelength_encoders[wavelength]

        inputs = layers.Input(shape=self.config['aia_shape'])
        x = inputs

        # CNN layers for feature extraction
        for filters in self.config['cnn_filters']:
            x = layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config['l2_reg'])
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=2)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)

        encoder = Model(
            inputs=inputs,
            outputs=x,
            name=f'aia_{wavelength}_encoder'
        )
        self.aia_wavelength_encoders[wavelength] = encoder
        return encoder

    def build_cross_wavelength_attention(self, wavelength_features):
        """Build attention mechanism to correlate different AIA wavelengths"""
        # Combine all wavelength features
        if len(wavelength_features) == 1:
            return wavelength_features[0]

        # Multi-head attention across wavelengths
        concat_features = layers.Concatenate(axis=1)(wavelength_features)
        reshaped = layers.Reshape((len(wavelength_features), -1))(concat_features)

        # Self-attention to find correlations
        attention = layers.MultiHeadAttention(
            num_heads=min(self.config['transformer_heads'], len(wavelength_features)),
            key_dim=128 // min(self.config['transformer_heads'], len(wavelength_features))
        )(reshaped, reshaped)

        x = layers.Add()([reshaped, attention])
        x = layers.LayerNormalization()(x)

        # Feed forward
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dropout(self.config['dropout_rate'])(ff)
        ff = layers.Dense(reshaped.shape[-1])(ff)

        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)

        return x

    def build_temporal_transformer(self):
        """Build transformer encoder for sequential data processing"""
        # Input shape: [batch, sequence_length, features]
        inputs = layers.Input(shape=(
            self.config['sequence_length'],
            self.config['transformer_dim']
        ))

        # Positional encoding
        pos_encoding = self._positional_encoding(
            self.config['sequence_length'],
            self.config['transformer_dim']
        )
        x = inputs + pos_encoding

        # Transformer blocks
        for _ in range(self.config['transformer_layers']):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=self.config['transformer_heads'],
                key_dim=self.config['transformer_dim'] // self.config['transformer_heads']
            )(x, x)

            # Add & normalize
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)

            # Feed forward
            ff = layers.Dense(self.config['transformer_dim'] * 4, activation='relu')(x)
            ff = layers.Dropout(self.config['dropout_rate'])(ff)
            ff = layers.Dense(self.config['transformer_dim'])(ff)

            # Add & normalize
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)

        # Global features from sequence
        x = layers.GlobalAveragePooling1D()(x)

        # Add recurrent layer for explicit modeling of temporal dependencies
        gru = layers.GRU(128, return_sequences=True)(inputs)
        gru_features = layers.GlobalAveragePooling1D()(gru)

        # Combine transformer and RNN features
        combined = layers.Concatenate()([x, gru_features])
        x = layers.Dense(256, activation='relu')(combined)

        self.temporal_encoder = Model(inputs=inputs, outputs=x, name='temporal_encoder')
        return self.temporal_encoder

    def _positional_encoding(self, max_len, d_model):
        """Create positional encoding for transformer"""
        positions = np.arange(max_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dimensions // 2)) / d_model)

        # Apply sin to even indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        # Apply cos to odd indices
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build_uncertainty_quantification(self, features, output_dims):
        """Build Bayesian layers for uncertainty quantification"""
        # Prior distribution for the Bayesian layers
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(output_dims), scale=1),
            reinterpreted_batch_ndims=1
        )

        # Variational posterior with trainable parameters
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(self.config['dropout_rate'])(x)

        # Bayesian dense layer for classification uncertainty
        bayesian_output = tfpl.DenseVariational(
            units=tfpl.IndependentNormal.params_size(output_dims),
            make_prior_fn=lambda *args, **kwargs: prior,
            make_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            kl_weight=1 / self.config['batch_size'],
            activation=None
        )(x)

        distribution_output = tfpl.IndependentNormal(output_dims)(bayesian_output)

        return distribution_output

    def build_full_model(self):
        """Build the complete hybrid architecture"""
        # Input layers
        magnetogram_inputs = layers.Input(
            shape=(self.config['sequence_length'],) + self.config['magnetogram_shape'],
            name='magnetogram_sequence'
        )

        aia_inputs = {
            wavelength: layers.Input(
                shape=(self.config['sequence_length'],) + self.config['aia_shape'],
                name=f'aia_{wavelength}_sequence'
            )
            for wavelength in self.config['aia_wavelengths']
        }

        # Feature extraction for each time step
        magnetogram_features = []
        aia_wavelength_features = {wave: [] for wave in self.config['aia_wavelengths']}

        mag_encoder = self.build_magnetogram_encoder()

        # Process each time step
        for t in range(self.config['sequence_length']):
            # Extract magnetogram features

            mag_features = mag_encoder(magnetogram_inputs[:, t])
            magnetogram_features.append(mag_features)

            # Extract AIA features for each wavelength
            for wavelength in self.config['aia_wavelengths']:
                wave_encoder = self.build_aia_wavelength_encoder(wavelength)
                wave_features = wave_encoder(aia_inputs[wavelength][:, t])
                aia_wavelength_features[wavelength].append(wave_features)

        # Convert lists to tensors
        magnetogram_sequence = layers.Lambda(lambda x: tf.stack(x, axis=1))(magnetogram_features)

        aia_sequence_by_wavelength = {}
        for wavelength in self.config['aia_wavelengths']:
            aia_sequence_by_wavelength[wavelength] = layers.Lambda(
                lambda x: tf.stack(x, axis=1)
            )(aia_wavelength_features[wavelength])

        # Process each wavelength sequence with temporal encoder
        temporal_encoder = self.build_temporal_transformer()
        magnetogram_temporal = temporal_encoder(magnetogram_sequence)

        aia_temporal_features = []
        for wavelength in self.config['aia_wavelengths']:
            aia_temp = temporal_encoder(aia_sequence_by_wavelength[wavelength])
            aia_temporal_features.append(aia_temp)

        # Cross-wavelength attention
        aia_combined = self.build_cross_wavelength_attention(aia_temporal_features)

        # Combine all features
        combined_features = layers.Concatenate()([magnetogram_temporal, aia_combined])

        # High-level feature extraction
        x = layers.Dense(512, activation='relu')(combined_features)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)

        # Task-specific heads with uncertainty quantification

        # 1. Flare classification (B, C, M, X, None)
        flare_class_logits = layers.Dense(5, name='flare_class_logits')(x)
        flare_class_output = layers.Softmax(name='flare_class')(flare_class_logits)

        # 2. CME probability
        cme_prob_output = layers.Dense(1, activation='sigmoid', name='cme_probability')(x)

        # 3. X-ray brightness prediction (log scale)
        xray_flux_output = self.build_uncertainty_quantification(x, 1)
        # xray_flux_mean = tfpl.DistributionLambda(
        #     lambda p: tfd.Normal(loc=p.mean(), scale=1e-6),
        #     name='xray_flux'
        # )(xray_flux_output)
        xray_flux_mean = layers.Lambda(lambda p: p.mean(), name='xray_flux')(xray_flux_output)

        # 4. Time to event prediction with uncertainty
        time_to_event_output = self.build_uncertainty_quantification(x, 1)
        # time_to_event_mean = tfpl.DistributionLambda(
        #     lambda p: tfd.Normal(loc=p.mean(), scale=1e-6),
        #     name='time_to_event'
        # )(time_to_event_output)
        time_to_event_mean = layers.Lambda(lambda p: p.mean(), name='time_to_event')(time_to_event_output)

        # Combine all inputs and outputs
        model_inputs = [magnetogram_inputs] + list(aia_inputs.values())
        model_outputs = [
            flare_class_output,
            cme_prob_output,
            xray_flux_mean,
            time_to_event_mean
        ]

        self.full_model = Model(inputs=model_inputs, outputs=model_outputs)

        # Define custom loss function with physics constraints
        losses = {
            'flare_class': self.physics_guided_categorical_crossentropy,
            'cme_probability': 'binary_crossentropy',
            'xray_flux': self.physics_guided_mse_loss,
            'time_to_event': self.physics_guided_time_to_event_loss
        }

        loss_weights = {
            'flare_class': self.config['flare_class_weight'],
            'cme_probability': self.config['cme_prob_weight'],
            'xray_flux': self.config['xray_flux_weight'],
            'time_to_event': self.config['time_to_event_weight']
        }

        metrics = {
            'flare_class': ['accuracy', tf.keras.metrics.F1Score(average='macro')],
            'cme_probability': ['accuracy', tf.keras.metrics.AUC()],
            'xray_flux': [tf.keras.metrics.MeanAbsoluteError()],
            'time_to_event': [tf.keras.metrics.MeanAbsoluteError()]
        }

        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return self.full_model

    def physics_guided_categorical_crossentropy(self, y_true, y_pred):
        """Custom loss for flare classification with physics constraints"""
        # Base categorical crossentropy
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Add physics-based regularization: higher class flares should have higher probabilities
        # for classes near their true class (e.g., an X flare might be predicted as an M with
        # some probability, but should almost never be predicted as B)
        class_weights = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)  # None, B, C, M, X
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)

        true_weight = tf.gather(class_weights, y_true_class)
        pred_weight = tf.gather(class_weights, y_pred_class)

        # Penalty for significant class underestimation (more severe than overestimation)
        class_penalty = tf.maximum(0.0, tf.cast(true_weight - pred_weight, tf.float32)) * 0.1

        return base_loss + class_penalty

    def physics_guided_mse_loss(self, y_true, y_pred):
        """Custom loss for X-ray flux prediction with physics constraints"""
        # Base MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Add physics constraint: predicted flux should follow power-law relationships
        # This encourages realistic flux predictions that follow observed solar physics
        # We apply a small penalty when predictions deviate from expected power-law behavior

        # Simplified power-law constraint (would be more complex in full implementation)
        log_y_true = tf.math.log(tf.maximum(y_true, 1e-8))
        log_y_pred = tf.math.log(tf.maximum(y_pred, 1e-8))

        power_law_penalty = tf.abs(log_y_true - log_y_pred) * 0.05

        return mse_loss + power_law_penalty

    def physics_guided_time_to_event_loss(self, y_true, y_pred):
        """Custom loss for time-to-event prediction with physics constraints"""
        # Base MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Add physics constraint: time-to-event predictions should be positive
        # and predictions should be more precise for imminent events

        # Penalty for negative time predictions (physically impossible)
        negative_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred)) * 10.0

        # Higher precision required for imminent events
        imminent_mask = tf.cast(y_true < 6.0, tf.float32)  # events within 6 hours
        imminent_error = tf.abs(y_true - y_pred) * imminent_mask
        imminent_penalty = tf.reduce_mean(imminent_error) * 2.0

        return mse_loss + negative_penalty + imminent_penalty

    def train_model(self, train_data, val_data, test_data=None):
        """Train the model with cross-validation"""
        if not self.full_model:
            self.build_full_model()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_solar_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        # Train model
        history = self.full_model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['epochs'],
            callbacks=callbacks
        )

        # Evaluate if test data is provided
        if test_data:
            results = self.full_model.evaluate(test_data)
            return history, results

        return history

    def predict_with_uncertainty(self, data, samples=None):
        """Make predictions with uncertainty estimates using MC Dropout"""
        if samples is None:
            samples = self.config['mc_dropout_samples']

        # Enable dropout at inference time
        inference_model = tf.keras.models.clone_model(self.full_model)

        # Set all dropout and batch norm layers to training mode
        for layer in inference_model.layers:
            if isinstance(layer, layers.Dropout):
                layer.training = True

        # Make multiple predictions
        predictions = []
        for _ in range(samples):
            # preds = inference_model.predict(data)
            preds = self.full_model.predict(data, training=True)
            predictions.append(preds)

        # Process predictions to get means and standard deviations
        results = {}

        # Flare classification (mean probabilities and entropy)
        flare_class_samples = np.array([p[0] for p in predictions])
        results['flare_class_mean'] = np.mean(flare_class_samples, axis=0)
        results['flare_class_std'] = np.std(flare_class_samples, axis=0)
        results['flare_class_entropy'] = -np.sum(
            results['flare_class_mean'] * np.log(results['flare_class_mean'] + 1e-10),
            axis=1
        )

        # CME probability
        cme_prob_samples = np.array([p[1] for p in predictions])
        results['cme_prob_mean'] = np.mean(cme_prob_samples, axis=0)
        results['cme_prob_std'] = np.std(cme_prob_samples, axis=0)

        # X-ray flux
        xray_flux_samples = np.array([p[2] for p in predictions])
        results['xray_flux_mean'] = np.mean(xray_flux_samples, axis=0)
        results['xray_flux_std'] = np.std(xray_flux_samples, axis=0)

        # Time to event
        time_to_event_samples = np.array([p[3] for p in predictions])
        results['time_to_event_mean'] = np.mean(time_to_event_samples, axis=0)
        results['time_to_event_std'] = np.std(time_to_event_samples, axis=0)

        return results

    def find_last_conv_layer(self, model):
        """Recursively search for last convolutional layer in model/submodels"""
        conv_layers = []
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                conv_layers.append(layer)
            elif hasattr(layer, 'layers'):  # Check for nested models
                nested_conv = self.find_last_conv_layer(layer)
                if nested_conv is not None:
                    conv_layers.append(nested_conv)

        return conv_layers[-1] if conv_layers else None

    def generate_feature_importance(self, data, target_output_idx=0):
        """Generate feature importance using Grad-CAM"""
        # Create a model that outputs the target and the last conv layer
        # last_conv_layer = None
        # for layer in reversed(self.full_model.layers):
        #     if isinstance(layer, layers.Conv2D):
        #         last_conv_layer = layer
        #         break

        last_conv_layer = self.find_last_conv_layer(self.full_model)

        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer")

        target_output = self.full_model.outputs[target_output_idx]

        # Create a model that maps the input to the activations of the last conv layer and output
        grad_model = Model(
            inputs=self.full_model.inputs,
            outputs=[last_conv_layer.output, target_output]
        )

        # Calculate gradients of output with respect to last conv layer
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(data)
            if target_output_idx == 0:  # Classification task
                class_idx = tf.argmax(predictions[0])
                target_output = predictions[0, class_idx]
            else:  # Regression tasks
                target_output = tf.reduce_mean(predictions)

        # Gradients of the output wrt the last conv layer
        grads = tape.gradient(target_output, conv_output)

        # Average gradients spatially
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Build a weighted combination of the feature maps
        cam = tf.reduce_sum(
            tf.multiply(weights, conv_output),
            axis=-1
        )

        # Normalize and convert to heatmap
        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())

        return cam.numpy()

    def optimize_hyperparams(self, train_data, val_data, param_bounds, n_iter=25):
        """Optimize hyperparameters using Bayesian optimization"""

        def objective_function(learning_rate, dropout_rate, l2_reg, transformer_heads):
            # Convert to appropriate types and ranges
            config = {
                'learning_rate': 10 ** learning_rate,  # log scale
                'dropout_rate': dropout_rate,
                'l2_reg': 10 ** l2_reg,  # log scale
                'transformer_heads': int(transformer_heads)
            }

            # Update model config
            self.config.update(config)

            # Rebuild and train model with new config
            # Use a smaller number of epochs for optimization
            temp_config = self.config.copy()
            temp_config['epochs'] = 10

            self.full_model = None  # Force rebuild
            self.build_full_model()

            # Train with early stopping
            history = self.train_model(train_data, val_data)

            # Get best validation performance
            best_val_loss = min(history.history['val_loss'])

            # Return negative loss for maximization
            return -best_val_loss

        # Define parameter bounds
        pbounds = {
            'learning_rate': (-5, -2),  # log scale: 1e-5 to 1e-2
            'dropout_rate': (0.1, 0.5),
            'l2_reg': (-6, -3),  # log scale: 1e-6 to 1e-3
            'transformer_heads': (2, 12.999)  # Will be converted to int: 2 to 12
        }

        # Run Bayesian optimization
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42
        )

        optimizer.maximize(init_points=5, n_iter=n_iter)

        # Get best parameters
        best_params = optimizer.max['params']
        best_params['learning_rate'] = 10 ** best_params['learning_rate']
        best_params['l2_reg'] = 10 ** best_params['l2_reg']
        best_params['transformer_heads'] = int(best_params['transformer_heads'])

        return best_params


# Data loading and preprocessing
class SolarDataGenerator(tf.keras.utils.Sequence):
    """Data generator for loading and preprocessing solar data"""

    def __init__(
            self,
            magnetogram_dir,
            aia_dir,
            magnetogram_features_csv,
            aia_features_csv,
            goes_xray_csv,
            flare_events_csv,
            sequence_length=24,
            prediction_window=24,
            batch_size=8,
            shuffle=True,
            augment=False,
            class_weights=None
    ):
        self.magnetogram_dir = magnetogram_dir
        self.aia_dir = aia_dir
        self.sequence_length = sequence_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_weights = class_weights

        # Load metadata and features
        self.magnetogram_features = pd.read_csv(magnetogram_features_csv)
        self.aia_features = pd.read_csv(aia_features_csv)
        self.goes_xray = pd.read_csv(goes_xray_csv)
        self.flare_events = pd.read_csv(flare_events_csv)

        # Convert timestamps to datetime
        self.magnetogram_features['timestamp'] = pd.to_datetime(self.magnetogram_features['timestamp'])
        self.aia_features['timestamp'] = pd.to_datetime(self.aia_features['timestamp'])
        self.goes_xray['timestamp'] = pd.to_datetime(self.goes_xray['timestamp'])
        self.flare_events['start_time'] = pd.to_datetime(self.flare_events['start_time'])
        self.flare_events['peak_time'] = pd.to_datetime(self.flare_events['peak_time'])
        self.flare_events['end_time'] = pd.to_datetime(self.flare_events['end_time'])

        # Create sequences
        self.create_sequences()

        # Initialize indexes
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def create_sequences(self):
        """Create sequences of data for training"""
        # Group by timestamp to align data
        self.sequences = []

        # Get unique timestamps from magnetogram data (base timeline)
        unique_timestamps = self.magnetogram_features['timestamp'].unique()
        unique_timestamps.sort()

        for i in range(len(unique_timestamps) - self.sequence_length - self.prediction_window):
            # Get sequence timestamps
            seq_timestamps = unique_timestamps[i:i + self.sequence_length]
            target_timestamp = unique_timestamps[i + self.sequence_length]

            # Find flares in the prediction window
            prediction_start = target_timestamp
            prediction_end = unique_timestamps[i + self.sequence_length + self.prediction_window - 1]

            # Get flares in prediction window
            flares_in_window = self.flare_events[
                (self.flare_events['start_time'] >= prediction_start) &
                (self.flare_events['start_time'] <= prediction_end)
                ]

            # Skip if no magnetogram data for this sequence
            mag_seq_data = self.magnetogram_features[
                self.magnetogram_features['timestamp'].isin(seq_timestamps)
            ]
            if len(mag_seq_data) < self.sequence_length:
                continue

            # Create sequence metadata
            sequence_info = {
                'sequence_start': seq_timestamps[0],
                'sequence_end': seq_timestamps[-1],
                'target_timestamp': target_timestamp,
                'prediction_end': prediction_end,
                'magnetogram_indices': mag_seq_data.index.tolist(),
                'has_flare': len(flares_in_window) > 0
            }

            # Add flare information if exists
            if len(flares_in_window) > 0:
                max_flare = flares_in_window.loc[flares_in_window['peak_flux'].idxmax()]
                sequence_info['flare_class'] = max_flare['class']
                sequence_info['flare_magnitude'] = max_flare['magnitude']
                sequence_info['flare_start_time'] = max_flare['start_time']
                sequence_info['time_to_flare'] = (max_flare['start_time'] - target_timestamp).total_seconds() / 3600.0

                # Map flare class to numeric
                flare_class_map = {'B': 1, 'C': 2, 'M': 3, 'X': 4}
                sequence_info['flare_class_numeric'] = flare_class_map.get(max_flare['class'][0], 0)

                # Check for associated CME (this would require additional data)
                # For now, use a placeholder probability based on flare class
                cme_probs = {'B': 0.1, 'C': 0.3, 'M': 0.6, 'X': 0.8}
                sequence_info['cme_probability'] = cme_probs.get(max_flare['class'][0], 0.0)

                # Get peak X-ray flux
                sequence_info['peak_xray_flux'] = max_flare['peak_flux']
            else:
                sequence_info['flare_class'] = 'None'
                sequence_info['flare_magnitude'] = 0.0
                sequence_info['flare_class_numeric'] = 0
                sequence_info['time_to_flare'] = -1.0  # No flare
                sequence_info['cme_probability'] = 0.0
                sequence_info['peak_xray_flux'] = 0.0

            self.sequences.append(sequence_info)


    def __len__(self):
        """Return the number of batches"""
        return int(np.ceil(len(self.sequences) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_sequences = [self.sequences[i] for i in batch_indexes]

        # Generate data
        X, y = self._generate_data(batch_sequences)

        return X, y


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def _generate_data(self, batch_sequences):
        """Generate data for a batch of sequences"""
        # Initialize input arrays
        magnetogram_batch = np.zeros(
            (len(batch_sequences), self.sequence_length, 1024, 1024, 1),
            dtype=np.float32
        )

        # Initialize AIA wavelength inputs
        aia_wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
        aia_batches = {}
        for wavelength in aia_wavelengths:
            aia_batches[wavelength] = np.zeros(
                (len(batch_sequences), self.sequence_length, 512, 512, 1),
                dtype=np.float32
            )

        # Initialize target arrays
        flare_class_batch = np.zeros((len(batch_sequences), 5), dtype=np.float32)  # None, B, C, M, X
        cme_prob_batch = np.zeros((len(batch_sequences), 1), dtype=np.float32)
        xray_flux_batch = np.zeros((len(batch_sequences), 1), dtype=np.float32)
        time_to_event_batch = np.zeros((len(batch_sequences), 1), dtype=np.float32)

        # Load data for each sequence
        for i, sequence in enumerate(batch_sequences):
            # Load magnetogram data
            for t, mag_idx in enumerate(sequence['magnetogram_indices'][:self.sequence_length]):
                # Get the filename from metadata
                mag_filename = f"{self.magnetogram_dir}/{self.magnetogram_features.loc[mag_idx, 'filename']}"

                # Load magnetogram data
                try:
                    mag_data = np.load(mag_filename)
                    # Normalize if needed
                    mag_data = (mag_data - np.mean(mag_data)) / (np.std(mag_data) + 1e-8)
                    # Reshape for input
                    magnetogram_batch[i, t, :, :, 0] = mag_data
                except Exception as e:
                    print(f"Error loading magnetogram {mag_filename}: {e}")

            # Load AIA data for each wavelength
            for wavelength in aia_wavelengths:
                # Get AIA data for the sequence timeframes
                seq_start = sequence['sequence_start']
                seq_end = sequence['sequence_end']

                aia_seq_data = self.aia_features[
                    (self.aia_features['timestamp'] >= seq_start) &
                    (self.aia_features['timestamp'] <= seq_end) &
                    (self.aia_features['wavelength'] == wavelength)
                    ]

                if len(aia_seq_data) == 0:
                    # No data for this wavelength, leave as zeros
                    continue

                # Sort by timestamp
                aia_seq_data = aia_seq_data.sort_values('timestamp')

                for t, (_, row) in enumerate(aia_seq_data.iterrows()):
                    if t >= self.sequence_length:
                        break

                    # Get the filename
                    aia_filename = f"{self.aia_dir}/{row['filename']}"

                    # Load AIA data
                    try:
                        aia_data = np.load(aia_filename)
                        # Log-scale preprocessing
                        aia_data = np.log10(aia_data + 1e-3)
                        # Resize to 512x512 if needed (could use scipy.ndimage)
                        # For this example, we'll assume it's already 512x512
                        aia_batches[wavelength][i, t, :, :, 0] = aia_data
                    except Exception as e:
                        print(f"Error loading AIA {wavelength} image {aia_filename}: {e}")

            # Set target values
            flare_class = sequence['flare_class_numeric']
            flare_class_batch[i, flare_class] = 1.0  # One-hot encoding

            cme_prob_batch[i, 0] = sequence['cme_probability']

            xray_flux_batch[i, 0] = sequence['peak_xray_flux']

            time_to_event = sequence['time_to_flare']
            if time_to_event > 0:
                time_to_event_batch[i, 0] = time_to_event
            else:
                # No flare in the window, set to prediction window length
                time_to_event_batch[i, 0] = self.prediction_window * 0.5  # hours

            # Apply data augmentation if enabled
            if self.augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    magnetogram_batch[i] = magnetogram_batch[i, :, :, ::-1, :]
                    for wavelength in aia_wavelengths:
                        aia_batches[wavelength][i] = aia_batches[wavelength][i, :, :, ::-1, :]

                # Random rotation (90, 180, 270 degrees)
                k = np.random.randint(0, 4)  # 0: no rotation, 1: 90 deg, 2: 180 deg, 3: 270 deg
                if k > 0:
                    magnetogram_batch[i] = np.rot90(magnetogram_batch[i], k=k, axes=(1, 2))
                    for wavelength in aia_wavelengths:
                        aia_batches[wavelength][i] = np.rot90(aia_batches[wavelength][i], k=k, axes=(1, 2))

        # Prepare inputs and outputs
        X = [magnetogram_batch] + [aia_batches[wavelength] for wavelength in aia_wavelengths]
        y = [flare_class_batch, cme_prob_batch, xray_flux_batch, time_to_event_batch]

        return X, y


class EnhancedSolarEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def calculate_brier_skill_score(self, y_true, y_pred_probs):
        """Calculate Brier Skill Score (BSS)"""
        # Convert multiclass to binary for BSS calculation
        y_true_binary = np.any(y_true[:, 1:], axis=1).astype(int)  # Any flare (B, C, M, X)
        y_pred_binary = 1 - y_pred_probs[:, 0]  # Probability of any flare

        # Calculate Brier score and reference score
        bs = brier_score_loss(y_true_binary, y_pred_binary)
        climatology = np.mean(y_true_binary)
        bs_ref = brier_score_loss(y_true_binary, np.ones_like(y_pred_binary) * climatology)

        return 1 - (bs / bs_ref)

    def generate_reliability_diagram(self, y_true, y_pred_probs, n_bins=10):
        """Generate reliability diagram and calibration metrics"""
        y_true_binary = np.any(y_true[:, 1:], axis=1).astype(int)
        y_pred_binary = 1 - y_pred_probs[:, 0]

        # Reliability diagram data
        prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_binary, n_bins=n_bins)

        # Isotonic regression calibration
        iso_reg = IsotonicRegression(out_of_bounds='clip').fit(y_pred_binary, y_true_binary)
        prob_pred_iso = np.linspace(0, 1, 100)
        prob_true_iso = iso_reg.predict(prob_pred_iso)

        # Calculate calibration metrics
        ece = np.mean(np.abs(prob_true - prob_pred))
        mce = np.max(np.abs(prob_true - prob_pred))

        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'prob_pred_iso': prob_pred_iso,
            'prob_true_iso': prob_true_iso,
            'ece': ece,
            'mce': mce
        }

    def analyze_lead_time_accuracy(self, test_data, time_bins=[0, 6, 12, 24, 48]):
        """Analyze accuracy vs forecast lead time"""
        # Get predictions and ground truth
        predictions = self.model.predict_with_uncertainty(test_data)
        sequences = test_data.sequences

        # Extract true flare times and predictions
        lead_times = []
        flare_occurred = []
        pred_probs = []

        for seq, pred in zip(sequences, predictions['flare_class_mean']):
            if seq['time_to_flare'] > 0:
                lead_times.append(seq['time_to_flare'])
                flare_occurred.append(1)
                pred_probs.append(1 - pred[0])  # Probability of any flare
            else:
                lead_times.append(self.config['prediction_window'])
                flare_occurred.append(0)
                pred_probs.append(1 - pred[0])

        # Bin by lead time
        binned_results = []
        for i in range(len(time_bins) - 1):
            mask = (lead_times >= time_bins[i]) & (lead_times < time_bins[i + 1])
            binned_true = np.array(flare_occurred)[mask]
            binned_pred = np.array(pred_probs)[mask]

            if len(binned_true) == 0:
                continue

            # Calculate metrics
            acc = accuracy_score(binned_true, binned_pred > 0.5)
            tss = self.calculate_tss(binned_true, binned_pred > 0.5)
            bss = self.calculate_brier_skill_score(binned_true, binned_pred)

            binned_results.append({
                'lead_time_range': (time_bins[i], time_bins[i + 1]),
                'accuracy': acc,
                'tss': tss,
                'bss': bss,
                'sample_count': len(binned_true)
            })

        return binned_results

    def analyze_false_predictions(self, test_data, n_cases=5):
        """Analyze top false positives/negatives with explanations"""
        # Get predictions and ground truth
        predictions = self.model.predict_with_uncertainty(test_data)
        sequences = test_data.sequences

        # Identify false positives/negatives
        false_positives = []
        false_negatives = []

        for i, (seq, pred) in enumerate(zip(sequences, predictions['flare_class_mean'])):
            true_class = seq['flare_class_numeric']
            pred_class = np.argmax(pred)

            if pred_class > 0 and true_class == 0:
                # False positive
                false_positives.append({
                    'index': i,
                    'confidence': pred[pred_class],
                    'predicted_class': pred_class,
                    'time_to_flare': seq['time_to_flare'],
                    'metadata': seq
                })
            elif pred_class == 0 and true_class > 0:
                # False negative
                false_negatives.append({
                    'index': i,
                    'confidence': pred[0],
                    'true_class': true_class,
                    'time_to_flare': seq['time_to_flare'],
                    'metadata': seq
                })

        # Sort by confidence and select top cases
        fp_cases = sorted(false_positives, key=lambda x: x['confidence'], reverse=True)[:n_cases]
        fn_cases = sorted(false_negatives, key=lambda x: x['confidence'], reverse=True)[:n_cases]

        # Generate explanations for each case
        for case in fp_cases + fn_cases:
            data = test_data[case['index']]
            case['gradcam'] = self.model.generate_feature_importance(data[0], 0)
            case['integrated_grads'] = self.compute_integrated_gradients(data[0], 0)

        return {'false_positives': fp_cases, 'false_negatives': fn_cases}


class EnhancedSolarExplainability:
    def __init__(self, model):
        self.model = model

    def compute_integrated_gradients(self, inputs, target_class_idx, baseline=None, steps=50):
        """Compute integrated gradients for feature attribution"""
        # Create baseline if not provided
        if baseline is None:
            baseline = 0.0 * inputs  # Black image baseline

        # Interpolate between baseline and input
        interpolated_inputs = [baseline + (float(i) / steps) * (inputs - baseline)
                               for i in range(steps + 1)]

        # Compute gradients
        gradients = []
        for x in interpolated_inputs:
            with tf.GradientTape() as tape:
                tape.watch(x)
                preds = self.model(x)
                target = preds[target_class_idx][:, target_class_idx]
            grad = tape.gradient(target, x)
            gradients.append(grad.numpy())

        # Average gradients across path
        avg_grads = np.mean(gradients[:-1], axis=0)
        integrated_grads = (inputs - baseline) * avg_grads
        return integrated_grads.numpy()

    def generate_counterfactuals(self, input_sample, target_class, lr=0.01, max_iter=100, lambda_reg=0.1):
        """Generate counterfactual explanations via optimization"""
        # Convert to TensorFlow variables
        original_input = tf.identity(input_sample)
        perturbed_input = tf.Variable(input_sample)

        # Define optimization
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                # Get prediction for perturbed input
                preds = self.model(perturbed_input)
                current_prob = preds[0][0, target_class]

                # Loss components
                classification_loss = 1 - current_prob  # Maximize target class probability
                similarity_loss = tf.reduce_mean(tf.square(perturbed_input - original_input))
                total_loss = classification_loss + lambda_reg * similarity_loss

            # Compute gradients
            grads = tape.gradient(total_loss, [perturbed_input])
            opt.apply_gradients(zip(grads, [perturbed_input]))

            # Project to valid input range
            perturbed_input.assign(tf.clip_by_value(perturbed_input, 0.0, 1.0))

        return perturbed_input.numpy()

    def compute_feature_ablation(self, inputs, target_class_idx, feature_mask):
        """Compute feature importance through systematic ablation"""
        original_pred = self.model(inputs)[0][0, target_class_idx]

        # Ablate features according to mask
        ablated_inputs = inputs * feature_mask
        ablated_pred = self.model(ablated_inputs)[0][0, target_class_idx]

        return original_pred - ablated_pred


# Training and Evaluation
def train_and_evaluate_model(config=None):
    """Main function to train and evaluate the model"""
    # Default configuration
    if config is None:
        config = {
            'data_dir': './data/processed',
            'batch_size': 8,
            'sequence_length': 24,  # 12 hours at 30-min cadence
            'prediction_window': 48,  # 24 hours
            'learning_rate': 1e-4,
            'epochs': 100,
            'val_split': 0.15,
            'test_split': 0.15,
            'cross_validation': True,
            'n_folds': 5,
            'optimize_hyperparams': True,
            'mc_dropout_samples': 20
        }

    # Set up paths
    magnetogram_dir = f"{config['data_dir']}/magnetograms"
    aia_dir = f"{config['data_dir']}/aia_images"
    magnetogram_features_csv = f"{config['data_dir']}/features/magnetogram_features.csv"
    aia_features_csv = f"{config['data_dir']}/features/aia_features.csv"
    goes_xray_csv = f"{config['data_dir']}/goes_xray/goes_xray_flux.csv"
    flare_events_csv = f"{config['data_dir']}/goes_xray/goes_flare_events.csv"

    # Create data generator
    data_gen = SolarDataGenerator(
        magnetogram_dir=magnetogram_dir,
        aia_dir=aia_dir,
        magnetogram_features_csv=magnetogram_features_csv,
        aia_features_csv=aia_features_csv,
        goes_xray_csv=goes_xray_csv,
        flare_events_csv=flare_events_csv,
        sequence_length=config['sequence_length'],
        prediction_window=config['prediction_window'],
        batch_size=config['batch_size'],
        shuffle=True,
        augment=True
    )

    # Split data into train, validation, and test sets
    indices = np.arange(len(data_gen.sequences))
    np.random.shuffle(indices)

    if config['cross_validation']:
        # Use stratified K-fold cross-validation
        # Extract labels for stratification
        labels = np.array([s['flare_class_numeric'] for s in data_gen.sequences])

        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
        fold_results = []
        explainability_dir = 'crossval_explanations'
        os.makedirs(explainability_dir, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
            print(f"\nTraining fold {fold + 1}/{config['n_folds']}")

            # Further split train into train and validation
            train_val_labels = labels[train_idx]
            train_val_indices = indices[train_idx]

            # Stratified split for validation
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=config['val_split'],
                stratify=train_val_labels,
                random_state=fold
            )

            test_indices = indices[test_idx]

            # # Create data generators for each set
            # train_gen = tf.data.Dataset.from_generator(
            #     lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
            #              for batch in
            #              np.array_split(train_indices, max(1, len(train_indices) // config['batch_size']))),
            #     output_signature=(
            #         (
            #             tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
            #             *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
            #               for _ in range(9)]  # 9 AIA wavelengths
            #         ),
            #         (
            #             tf.TensorSpec(shape=(None, 5), dtype=tf.float32),  # Flare class
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # CME probability
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # X-ray flux
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Time to event
            #         )
            #     )
            # ).prefetch(tf.data.AUTOTUNE)
            #
            # val_gen = tf.data.Dataset.from_generator(
            #     lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
            #              for batch in np.array_split(val_indices, max(1, len(val_indices) // config['batch_size']))),
            #     output_signature=(
            #         (
            #             tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
            #             *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
            #               for _ in range(9)]
            #         ),
            #         (
            #             tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            #         )
            #     )
            # ).prefetch(tf.data.AUTOTUNE)
            #
            # test_gen = tf.data.Dataset.from_generator(
            #     lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
            #              for batch in np.array_split(test_indices, max(1, len(test_indices) // config['batch_size']))),
            #     output_signature=(
            #         (
            #             tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
            #             *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
            #               for _ in range(9)]
            #         ),
            #         (
            #             tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            #             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            #         )
            #     )
            # ).prefetch(tf.data.AUTOTUNE)

            # Create TensorFlow datasets
            train_gen = create_tf_dataset(data_gen, train_indices, config)
            val_gen = create_tf_dataset(data_gen, val_indices, config)
            test_gen = create_tf_dataset(data_gen, test_idx, config)

            # Create model
            model_params = {
                'learning_rate': config['learning_rate'],
                'sequence_length': config['sequence_length'],
                'batch_size': config['batch_size']
            }

            # Add memory optimization for RTX 3060 (8GB VRAM)
            # This involves using mixed precision and gradient accumulation
            # Set memory growth to avoid OOM errors
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Memory growth enabled for GPU")

                # Enable mixed precision for better memory efficiency
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision enabled")

                # Further memory optimizations by reducing model size
                # Reduce magnetogram resolution if needed
                if config.get('reduce_magnetogram_resolution', True):
                    model_params['magnetogram_shape'] = (256, 256, 1)  # Reduced resolution
                    model_params['aia_shape'] = (128, 128, 1)  # Reduced resolution
                    print("Using reduced resolution for model inputs")

            solar_model = SolarFlarePredictor(model_params)

            # Optimize hyperparameters if enabled
            if config['optimize_hyperparams'] and fold == 0:
                print("Optimizing hyperparameters...")
                best_params = solar_model.optimize_hyperparams(
                    train_gen,
                    val_gen,
                    {
                        'learning_rate': (-5, -3),
                        'dropout_rate': (0.1, 0.5),
                        'l2_reg': (-6, -4),
                        'transformer_heads': (2, 8.999)
                    },
                    n_iter=10  # Reduced for resources
                )

                # Update model with optimized parameters
                solar_model.config.update(best_params)
                print(f"Optimized params: {best_params}")

            # Build and train model
            solar_model.build_full_model()

            # Check model size and parameters
            solar_model.full_model.summary()

            # Train the model
            history, results = solar_model.train_model(train_gen, val_gen, test_gen)

            # Evaluate model performance
            test_predictions = solar_model.predict_with_uncertainty(test_gen)

            # Calculate metrics
            # Extract test labels
            test_batches = list(test_gen.as_numpy_iterator())
            test_labels = [np.concatenate([b[1][i] for b in test_batches], axis=0)
                           for i in range(4)]  # 4 outputs

            # Classification metrics for flare class
            flare_pred_class = np.argmax(test_predictions['flare_class_mean'], axis=1)
            flare_true_class = np.argmax(test_labels[0], axis=1)

            accuracy = accuracy_score(flare_true_class, flare_pred_class)
            f1 = f1_score(flare_true_class, flare_pred_class, average='macro')

            # True Skill Statistic (TSS)
            tss = calculate_tss(flare_true_class, flare_pred_class)

            # Heidke Skill Score (HSS)
            hss = calculate_hss(flare_true_class, flare_pred_class)

            # MSE for regression tasks
            xray_mse = mean_squared_error(test_labels[2], test_predictions['xray_flux_mean'])
            time_mse = mean_squared_error(test_labels[3], test_predictions['time_to_event_mean'])

            # Print metrics
            print(f"Fold {fold + 1} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"True Skill Statistic (TSS): {tss:.4f}")
            print(f"Heidke Skill Score (HSS): {hss:.4f}")
            print(f"X-ray Flux MSE: {xray_mse:.6f}")
            print(f"Time to Event MSE: {time_mse:.6f}")

            # Save fold results
            fold_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'f1': f1,
                'tss': tss,
                'hss': hss,
                'xray_mse': xray_mse,
                'time_mse': time_mse
            })

            # Save model for this fold
            solar_model.full_model.save(f"solar_model_fold_{fold + 1}.h5")

            # Generate and save feature attributions
            feature_importance = solar_model.generate_feature_importance(
                next(iter(test_gen))[0],
                target_output_idx=0  # Flare classification
            )

            np.save(f"feature_importance_fold_{fold + 1}.npy", feature_importance)

            # Clear session to free memory
            tf.keras.backend.clear_session()

        # Compute and print average performance across folds
        print("\nCross-validation Results:")
        metrics = ['accuracy', 'f1', 'tss', 'hss', 'xray_mse', 'time_mse']
        for metric in metrics:
            values = [r[metric] for r in fold_results]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"Mean {metric}: {mean_value:.4f} ± {std_value:.4f}")

        return fold_results

    else:
        # Simple train/val/test split
        val_size = int(config['val_split'] * len(indices))
        test_size = int(config['test_split'] * len(indices))
        train_size = len(indices) - val_size - test_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create data generators
        train_gen = tf.data.Dataset.from_generator(
            lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
                     for batch in np.array_split(train_indices, max(1, len(train_indices) // config['batch_size']))),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
                    *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
                      for _ in range(9)]
                ),
                (
                    tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            )
        ).prefetch(tf.data.AUTOTUNE)

        val_gen = tf.data.Dataset.from_generator(
            lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
                     for batch in np.array_split(val_indices, max(1, len(val_indices) // config['batch_size']))),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
                    *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
                      for _ in range(9)]
                ),
                (
                    tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            )
        ).prefetch(tf.data.AUTOTUNE)

        test_gen = tf.data.Dataset.from_generator(
            lambda: ((data_gen._generate_data([data_gen.sequences[i] for i in batch]))
                     for batch in np.array_split(test_indices, max(1, len(test_indices) // config['batch_size']))),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),
                    *[tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
                      for _ in range(9)]
                ),
                (
                    tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            )
        ).prefetch(tf.data.AUTOTUNE)

        # Create model
        model_params = {
            'learning_rate': config['learning_rate'],
            'sequence_length': config['sequence_length'],
            'batch_size': config['batch_size']
        }

        # Add memory optimization for RTX 3060
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Memory growth enabled for GPU")

            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")

            # Reduce resolution if needed
            if config.get('reduce_magnetogram_resolution', True):
                model_params['magnetogram_shape'] = (256, 256, 1)
                model_params['aia_shape'] = (128, 128, 1)
                print("Using reduced resolution for model inputs")

        solar_model = SolarFlarePredictor(model_params)

        # Optimize hyperparameters if enabled
        if config['optimize_hyperparams']:
            print("Optimizing hyperparameters...")
            best_params = solar_model.optimize_hyperparams(
                train_gen,
                val_gen,
                {
                    'learning_rate': (-5, -3),
                    'dropout_rate': (0.1, 0.5),
                    'l2_reg': (-6, -4),
                    'transformer_heads': (2, 8.999)
                },
                n_iter=10
            )

            # Update model with optimized parameters
            solar_model.config.update(best_params)
            print(f"Optimized params: {best_params}")

        # Build and train model
        solar_model.build_full_model()
        solar_model.full_model.summary()

        # Train the model
        history, results = solar_model.train_model(train_gen, val_gen, test_gen)

        # Evaluate model and calculate metrics
        test_predictions = solar_model.predict_with_uncertainty(test_gen)

        # Extract test labels
        test_batches = list(test_gen.as_numpy_iterator())
        test_labels = [np.concatenate([b[1][i] for b in test_batches], axis=0)
                       for i in range(4)]

        # Calculate metrics
        flare_pred_class = np.argmax(test_predictions['flare_class_mean'], axis=1)
        flare_true_class = np.argmax(test_labels[0], axis=1)

        accuracy = accuracy_score(flare_true_class, flare_pred_class)
        f1 = f1_score(flare_true_class, flare_pred_class, average='macro')
        tss = calculate_tss(flare_true_class, flare_pred_class)
        hss = calculate_hss(flare_true_class, flare_pred_class)

        xray_mse = mean_squared_error(test_labels[2], test_predictions['xray_flux_mean'])
        time_mse = mean_squared_error(test_labels[3], test_predictions['time_to_event_mean'])

        # Print metrics
        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Skill Statistic (TSS): {tss:.4f}")
        print(f"Heidke Skill Score (HSS): {hss:.4f}")
        print(f"X-ray Flux MSE: {xray_mse:.6f}")
        print(f"Time to Event MSE: {time_mse:.6f}")

        # Save the model
        solar_model.full_model.save("solar_model_final.h5")

        # Generate and save feature attributions
        feature_importance = solar_model.generate_feature_importance(
            next(iter(test_gen))[0],
            target_output_idx=0  # Flare classification
        )

    np.save("feature_importance_final.npy", feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_importance, cmap='viridis')
    plt.colorbar(label='Feature Importance')
    plt.title('Feature Importance Map')
    plt.savefig('feature_importance_map.png')
    plt.close()

    # Return final results
    return {
        'accuracy': accuracy,
        'f1': f1,
        'tss': tss,
        'hss': hss,
        'xray_mse': xray_mse,
        'time_mse': time_mse,
        'history': history.history,
        'test_predictions': test_predictions,
        'test_labels': test_labels
    }


# Helper functions for skill scores
def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS)"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    return tss


def calculate_hss(y_true, y_pred):
    """Calculate Heidke Skill Score (HSS)"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return hss


# Main execution
if __name__ == "__main__":
    # Run training and evaluation
    results = train_and_evaluate_model()

    # Print final results
    print("\nFinal Model Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"True Skill Statistic (TSS): {results['tss']:.4f}")
    print(f"Heidke Skill Score (HSS): {results['hss']:.4f}")
    print(f"X-ray Flux MSE: {results['xray_mse']:.6f}")
    print(f"Time to Event MSE: {results['time_mse']:.6f}")

    # Plot training history
    plt.figure(figsize=(12, 8))
    for metric in results['history']:
        if not metric.startswith('val_'):
            plt.plot(results['history'][metric], label=metric)
            if f'val_{metric}' in results['history']:
                plt.plot(results['history'][f'val_{metric}'], label=f'val_{metric}')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()


def plot_input(metadata, filename):
    """Visualize input sequence"""
    plt.figure(figsize=(20, 5))
    for t in range(3):  # Show first 3 time steps
        plt.subplot(1, 3, t + 1)
        plt.imshow(metadata['magnetogram'][t], cmap='gray')
        plt.title(f"T-{(3 - t) * 0.5}h")
    plt.savefig(filename)
    plt.close()


def plot_attributions(case, filename):
    """Visualize explanation maps"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(case['gradcam'], cmap='hot')
    plt.title('Grad-CAM')

    plt.subplot(1, 3, 2)
    plt.imshow(case['integrated_grads'], cmap='hot')
    plt.title('Integrated Gradients')

    plt.subplot(1, 3, 3)
    # Plot difference between original and counterfactual
    plt.imshow(np.abs(case['original'] - case['counterfactual']), cmap='hot')
    plt.title('Counterfactual Differences')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()