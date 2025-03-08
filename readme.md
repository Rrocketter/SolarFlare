


# Solar Flare Prediction Model - Setup and Execution Guide

## Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended) with CUDA 11.2 and cuDNN 8.1
- 50GB+ free disk space for datasets
- 16GB+ RAM (32GB recommended)

## Installation

### 1. Install Python Requirements
Create a virtual environment and install dependencies:

```bash
python -m venv solar-env
source solar-env/bin/activate  # Linux/MacOS
# or solar-env\Scripts\activate  # Windows

pip install -r requirements.txt
```

Example `requirements.txt`:
```
tensorflow==2.10.0
tensorflow-probability==0.18.0
tensorflow-addons==0.19.0
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
bayesian-optimization==1.4.3
matplotlib==3.7.1
astropy==5.2.2
sunpy==4.1.3
```

### 2. Dataset Setup

1. Download dataset parts to a directory named `dataset_parts/`
2. Combine and extract dataset:

```bash
# Combine split files
cat dataset_parts/solar_dataset.tar.gz.* > solar_dataset.tar.gz

# Verify integrity
md5sum solar_dataset.tar.gz  # Compare with provided checksum

# Extract dataset
tar -xzvf solar_dataset.tar.gz
```

Directory structure after extraction:
```
data/
├── processed/
│   ├── magnetograms/
│   ├── aia_images/
│   └── features/
│   └── combined/
│   └── soho/
│   └── goes_xray/

logs/
saved_models/
```

## Running the Model

### 1. Training

### RUN THIS COMMAND FOR TRAINING HENRY:
```bash
CUDA_VISIBLE_DEVICES=0 python train_solar_flare_model.py
```

### IGNORE THIS
```bash
python train_solar_flare_model.py \
  --data_dir ./data/processed \
  --batch_size 8 \
  --epochs 100 \
  --sequence_length 24 \
  --output_dir ./saved_models
```

Key Arguments:
- `--data_dir`: Path to processed data directory
- `--batch_size`: Reduce if facing memory issues (default: 8)
- `--use_mixed_precision`: Enable for GPU acceleration (recommended)
- `--cache_dataset`: Enable dataset caching for faster training


#### Sample Command to run code with GPU:
```bash 
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python -u solar_flare_train.py \
  --data_dir ./data/processed \
  --batch_size 8 \
  --epochs 100 \
  --sequence_length 24 \
  --mixed_precision \
  --xla_compile \
  --output_dir ./saved_models \
  --log_dir ./logs \
  --cache_dataset
````

### 2. Inference
Run predictions using a trained model:

```bash
python predict_solar_flare.py \
  --model_path ./saved_models/best_model.h5 \
  --input_data ./data/processed/new_observations \
  --output_file predictions.csv
```

### 3. Hyperparameter Optimization
```bash
python optimize_hyperparams.py \
  --data_dir ./data/processed \
  --n_iter 50 \
  --init_points 10 \
  --output_file optimal_params.json
```

## Configuration

### Recommended Hardware Settings
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8GB     | 24GB+       |
| RAM       | 16GB    | 64GB        |
| Storage   | HDD     | NVMe SSD    |

### Runtime Flags
| Flag                      | Description                                  |
|---------------------------|----------------------------------------------|
| `--reduce_resolution`     | Use 512x512 images instead of 1024x1024     |
| `--use_amp`               | Enable Automatic Mixed Precision            |
| `--memory_growth`         | Enable GPU memory growth allocation         |
| `--profiler`              | Enable TensorFlow profiler for optimization |

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**:
   - Reduce batch size (start with 4)
   - Enable mixed precision with `--use_amp`
   - Use `--reduce_resolution` flag

2. **Dataset Loading Errors**:
   - Verify directory structure
   - Check file permissions
   - Ensure all sequence files are present

3. **Dependency Conflicts**:
   - Use exact versions from requirements.txt
   - Create fresh virtual environment

### Performance Tips
- Enable dataset caching with `--cache_dataset` for 2-3x speedup
- Use `TF_XLA_FLAGS=--tf_xla_auto_jit=2` for XLA compilation
- Set `CUDA_VISIBLE_DEVICES` to limit GPU usage

## Output Files
| File/Folder         | Description                          |
|---------------------|--------------------------------------|
| `logs/`             | Training logs and TensorBoard files  |
| `saved_models/`     | Best model checkpoints               |
| `predictions/`      | CSV files with prediction results    |
| `calibration_plots/`| Reliability diagrams and metrics     |

## License
This model is provided under the NASA Open Source Agreement (NOSA) version 1.3
```

This markdown file provides comprehensive instructions covering installation, dataset preparation, execution commands, configuration options, and troubleshooting. The structure is organized for clarity and includes code blocks for commands as well as tables for important configurations and hardware requirements.
