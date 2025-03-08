import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from complete_model import SolarFlarePredictor, SolarDataGenerator  # Adjust import path as needed


def load_test_data(config):
    """Load test dataset using SolarDataGenerator"""
    data_gen = SolarDataGenerator(
        magnetogram_dir=f"{config['data_dir']}/magnetograms",
        aia_dir=f"{config['data_dir']}/aia_images",
        magnetogram_features_csv=f"{config['data_dir']}/features/magnetogram_features.csv",
        aia_features_csv=f"{config['data_dir']}/features/aia_features.csv",
        goes_xray_csv=f"{config['data_dir']}/goes_xray/goes_xray_flux.csv",
        flare_events_csv=f"{config['data_dir']}/goes_xray/goes_flare_events.csv",
        sequence_length=config['sequence_length'],
        prediction_window=config['prediction_window'],
        batch_size=config['batch_size'],
        shuffle=False,
        augment=False
    )
    test_indices = np.arange(len(data_gen.sequences))  # Adjust with actual test indices
    return create_tf_dataset(data_gen, test_indices, config)


def create_tf_dataset(data_gen, indices, config):
    """Create TensorFlow dataset from data generator"""
    return tf.data.Dataset.from_generator(
        # Generator function: yield data batches
        lambda: (
            data_gen._generate_data([data_gen.sequences[i] for i in batch])
            for batch in np.array_split(indices, max(1, len(indices) // config['batch_size']))
        ),
        output_signature=(
            # Input structure: one magnetogram and nine AIA images
            (
                tf.TensorSpec(shape=(None, config['sequence_length'], 1024, 1024, 1), dtype=tf.float32),  # Magnetogram
                *(tf.TensorSpec(shape=(None, config['sequence_length'], 512, 512, 1), dtype=tf.float32)
                  for _ in range(9)),
            ),
            # Output structure: flare_class, cme_probability, xray_flux, time_to_event
            (
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32),   # flare_class
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),   # cme_probability
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),   # xray_flux
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)    # time_to_event
            )
        )
    ).prefetch(tf.data.AUTOTUNE)



def plot_roc_curves(y_true_labels, y_pred_probs, class_names):
    """Plot ROC curves for each class with AUC scores"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_labels, classes=np.arange(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()


def main():
    # Configuration must match training settings
    config = {
        'data_dir': './data/processed',
        'batch_size': 8,
        'sequence_length': 24,
        'prediction_window': 48,
        'magnetogram_shape': (1024, 1024, 1),
        'aia_shape': (512, 512, 1),
        # Add other parameters from SolarFlarePredictor's config
    }

    # Initialize predictor and load model
    solar_predictor = SolarFlarePredictor(config=config)
    solar_predictor.full_model = tf.keras.models.load_model(
        'solar_model_final.h5',
        custom_objects={
            'physics_guided_categorical_crossentropy': solar_predictor.physics_guided_categorical_crossentropy,
            'physics_guided_mse_loss': solar_predictor.physics_guided_mse_loss,
            'physics_guided_time_to_event_loss': solar_predictor.physics_guided_time_to_event_loss
        }
    )

    # Load test data and generate predictions
    test_data = load_test_data(config)
    predictions = solar_predictor.predict_with_uncertainty(test_data)
    y_pred_probs = predictions['flare_class_mean']

    # Extract true labels
    y_true = np.concatenate([y[0].numpy() for _, y in test_data], axis=0)
    y_true_labels = np.argmax(y_true, axis=1)

    # Generate ROC curves
    plot_roc_curves(y_true_labels, y_pred_probs, ['None', 'B', 'C', 'M', 'X'])


if __name__ == '__main__':
    main()