# Multi-task Segmentation and Classification Model

Multi-task segmentation and classification pipeline for ISIC2018 Challenge's HAM10000 dataset. Built with **PyTorch**, **Hydra**, and **MLflow** tracking.

## Features

- 🔬 **Multi-class classification**: MEL, NV, BCC, AKIEC, BKL, DF, VASC
- 🎯 **Multi-class segmentation**: Lesion mask segmentation
- 🏗️ **Segmentation models**: Pre-trained semantic segmentation encoder-based architectures from **SMP**
- 🧠 **Classification head**: Adaptive classification head via segmentation model encoder level outputs
- ⚖️ **Multi-task loss criterion**: Flexible loss creation and weighting
- ⚙️ **Hydra configuration**: Hierarchical YAML configs with experiment management
- 📊 **MLflow tracking**: Automatic logging of metrics, hyperparameters, and artifacts
- 💾 **Checkpointing**: Patience-based saving with training resumption
- ✅ **DevContainer ready**: VS Code development environment included

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU training, optional but recommended)
- **RAM**: 16GB minimum
- **Disk Space**: ~3GB for dataset, ~2GB for model checkpoints
- **Docker**: (Optional) For DevContainer support

## Installation

**Option 1: Local installation**
```bash
# Clone repository
git clone https://github.com/user/project.git
cd project

# Install Python dependencies
pip install -r requirements.txt
```

**Option 2: VS Code DevContainer**

1. Install Docker and VS Code with Remote-Containers extension
2. Open project in VS Code
3. Press F1 → Select "Remote-Containers: Reopen in Container"
4. Wait for container to build (first time only)

The DevContainer includes all dependencies and GPU support pre-configured.

## Dataset

### Download Dataset

**Source**: [Skin cancer: HAM10000 (Kaggle: Suraj Ghuwalewala)](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)

Download and place extracted dataset with images, binary segmentation masks, metadata in `data/` directory:

### Expected Directory Structure

 ```bash
data/
├── images/
├── masks/
└── metadata.csv
 ``` 

**Classes**:

- **AKIEC**: Actinic keratoses and intraepithelial carcinoma / Bowen's disease
- **BCC**: Basal cell carcinoma
- **BKL**: Benign keratosis-like lesions
- **DF**: Dermatofibroma
- **MEL**: Melanoma
- **NV**: Melanocytic nevi
- **VASC**: Vascular lesions


## Quickstart

### Training

Train and test with default configuration:

```bash
# Train and test with default configuration
python src/train.py
```

Train with custom settings:

```bash
# Use different model architecture
python src/train.py model.model=unet model.encoder_name=resnet34

# Adjust training parameters
python src/train.py epochs=50 data.batch_size=16 optimizer.learning_rate=0.0001

# Enable backbone freezing for transfer learning
python src/train.py model.freeze_backbone=true epochs=20
```

### Testing

Test a trained model:

```bash
# Test latest checkpoint
python src/test.py

# Test specific model
python src/test.py run_name=unet_resnet50_20251203_0016

# Save outputs to custom directory
python src/test.py local_save_dir=results/
```

Results are saved to `results/test/{run_name}/`:

- `classification_report.txt`: Per-class precision, recall, F1
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for each class
- `metrics.json`: Dice score, IoU
- `predictions/`: Sample predictions with overlays/outlines


### Inference

Create and place inference data into another `data` directory, e.g. `infer_data/`.

```bash
# Batch inference on directory
python src/infer.py data.data_dir=infer_data/ run_name=model_name

# Custom output location
python src/infer.py data.data_dir=images/ local_save_dir=predictions/ run_name=model_name
```

Results are saved to `results/inference/{run_name}/`:

- `predictions/`: Sample predictions with overlays/outlines

## Configuration

The project uses Hydra for hierarchical configuration management.

### Configuration Priority

1. `config/config.yaml` - Base defaults
2. `config/{group}/{option}.yaml` - Group-specific configs
3. Command-line overrides - Highest priority

### Key configuration groups:

- `config/data/` - Data configuration
- `config/loss/` - Loss configuration
- `config/model/` - Model configuration
- `config/trainer/` - Training settings
- `config/config.yaml` - Training/Testing entry file
- `config/infer.yaml` - Inference entry file

All configurations are saved to `outputs/YYYY-MM-DD/HH-MM-SS/` directory with time markers when a run is executed. Models are named using the `run_name` parameter configured in `config/config.yaml`. When configuring runs for models or backbones not defined in `src/models/smp_models.py`, add them first to the class's dictionaries.

## Checkpointing & Export

### Automatic Checkpointing

Checkpoints are saved automatically when validation loss improves:

```bash
models/checkpoints/{run_name}.pt
```

Each checkpoint contains:

- Model state dict
- Optimizer state
- Learning rate scheduler state
- Current epoch
- Best validation loss
- Training configuration

### Resume Training

```bash
# Resume from last checkpoint
python src/train.py resume=true

# Resume specific run
python src/train.py resume=true run_name=unet_resnet50_20251203_0016
```

## MLflow Tracking

All experiments are automatically tracked with MLflow.

### Logged Information

**Metrics** (per epoch):

- Training: loss, classification loss, segmentation loss, learning rate
- Validation: loss, accuracy, precision, recall, F1, Dice, IoU, AUROC
- Testing: All classification metrics + confusion matrix

**Parameters**:

- Model architecture and hyperparameters
- Training configuration
- Data augmentation settings
- Loss function details

**Artifacts**:

- Model checkpoints
- Training configuration (YAML)
- Test results (confusion matrix, ROC curves, classification report)
- Sample predictions

**View experiments**: 

Start MLflow UI:

```bash
# Open http://localhost:5000
mlflow ui --backend-store-uri mlruns
```

## Results

Average weighted metrics on the test set for various model configurations using the default `config/config.yaml` settings. 

<table>
  <thead>
    <tr>
      <th rowspan="2">Model/Backbone</th>
      <th colspan="4">Classification</th>
      <th colspan="2">Segmentation</th>
    </tr>
    <tr>
      <th>Accuracy</th> 
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>Dice</th>
      <th>IoU</th>
    </tr>
    <tr>
      <td>Unet/ResNet-34</td> 
      <td>0.83</td> 
      <td>0.86</td>
      <td>0.83</td>
      <td>0.84</td>
      <td>0.72</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>Unet/ResNet-50</td> 
      <td>0.81</td> 
      <td>0.85</td>
      <td>0.81</td>
      <td>0.82</td>
      <td>0.73</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>Unet/EfficientNet-B3</td> 
      <td>0.84</td> 
      <td>0.87</td>
      <td>0.84</td>
      <td>0.85</td>
      <td>0.73</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>DeepLabV3/ResNet-34</td> 
      <td>0.85</td> 
      <td>0.86</td>
      <td>0.85</td>
      <td>0.85</td>
      <td>0.70</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>DeepLabV3/ResNet-50</td> 
      <td>0.85</td> 
      <td>0.87</td>
      <td>0.85</td>
      <td>0.86</td>
      <td>0.72</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>Segformer/ResNet-34</td> 
      <td>0.82</td> 
      <td>0.85</td>
      <td>0.82</td>
      <td>0.83</td>
      <td>0.72</td>
      <td>0.85</td>
    </tr>
  </thead>
</table>

See the `results/test/` directory for per-class metrics and visualization.

## Project Structure

```markdown
.
├── data/                        # Dataset
|
├── config/                      # Hydra YAML configurations
│   ├── data/                    # Dataset configs
│   ├── loss/                    # Loss presets
│   ├── model/                   # Model architecture
│   ├── trainer/                 # Trainer settings
│   ├── config.yaml              # Main training config
│   └── infer.yaml               # Inference config
│
├── src/
│   ├── data/                    # Dataset and dataloaders
│   ├── losses/                  # Loss functions
│   ├── metrics/                 # Metrics manager
│   ├── models/                  # Model wrapper
│   ├── training/                # Training pipeline
│   ├── utils/                   # Helper utilities
│   └── factories.py             # Pipeline factory functions
│
├── .devcontainer/               # VS Code dev container
├── models/                      # Checkpoints
├── mlruns/                      # MLflow tracking
├── outputs/                     # Hydra run outputs
├── results/                     # Test/inference results
│
├── train.py                     # Training/Testing entry point
├── test.py                      # Testing entry point
├── infer.py                     # Inference entry point
|
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

## Development

### Code Quality

Install pre-commit hooks for automatic code formatting:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Citation

- Tschandl, P., Rinner, C., Apalla, Z. et al. Human–computer collaboration for skin cancer recognition. Nat Med (2020). https://doi.org/10.1038/s41591-020-0942-0

- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

## Acknowledgments

- **Dataset**: ISIC Archive and HAM10000 contributors
- **Models**: Segmentation Models PyTorch (SMP)
- **Framework**: PyTorch, Hydra, MLflow