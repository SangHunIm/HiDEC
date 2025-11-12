# HiDEC

Hierarchical Deep Extreme multi-label Classification

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies:
* torch >= 2.0.0
* pytorch-lightning >= 2.0.0
* transformers >= 4.30.0
* wandb >= 0.15.0
* hydra-core >= 1.3.0
* omegaconf >= 2.3.0
* anytree

## Configuration with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management, providing flexible and composable configs.

### Configuration Structure

```
config/
├── config.yaml                    # Base configuration
├── preprocessing.yaml             # Preprocessing configuration
├── model/                         # Model-specific configs
│   ├── bert.yaml                 # BERT encoder settings
│   └── gru.yaml                  # GRU encoder settings
└── experiment/                    # Full experiment configs
    ├── HiDEC_with_BERT.yaml      # BERT experiment
    └── HiDEC_with_GRU.yaml       # GRU experiment
```

## Data Preprocessing

1. Download EURLEX57K dataset from https://github.com/iliaschalkidis/lmtc-eurlex57k

2. Create necessary directories:
```bash
mkdir -p data/pregenerated data/datasets/EURLEX57K
```

3. Run preprocessing with Hydra:

```bash
python src/preprocessing.py \
    name=EURLEX57K \
    raw_dir=data/datasets/EURLEX57K/dataset \
    save_dir=data/pregenerated \
    hierarchy_file=data/datasets/EURLEX57K/EURLEX57K.json
```

## Training

### Basic Training

Train with BERT encoder:
```bash
python src/train.py experiment=HiDEC_with_BERT
```

Train with GRU encoder:
```bash
python src/train.py experiment=HiDEC_with_GRU
```

### Override Configuration

You can override any configuration parameter from the command line:

```bash
# Change learning rate
python src/train.py experiment=HiDEC_with_BERT model_hparams.learning_rate=0.0001

# Change batch size
python src/train.py experiment=HiDEC_with_BERT batch_size=64

# Change GPU devices
python src/train.py experiment=HiDEC_with_BERT trainer_hparams.devices=[0,1,2,3]

# Multiple overrides
python src/train.py \
    experiment=HiDEC_with_BERT \
    batch_size=64 \
    model_hparams.learning_rate=0.0001 \
    trainer_hparams.max_epochs=50
```

### View Configuration

Check the full configuration without running:
```bash
python src/train.py experiment=HiDEC_with_BERT --cfg job
```

## Testing/Evaluation

Run evaluation on a trained model:

```bash
python src/test.py \
    experiment=HiDEC_with_BERT \
    checkpoint_path=path/to/model.ckpt
```

With custom batch size:
```bash
python src/test.py \
    experiment=HiDEC_with_BERT \
    checkpoint_path=path/to/model.ckpt \
    test_batch_size=256
```

## Hydra Features

### Multirun - Hyperparameter Sweeps

Run experiments with different hyperparameters:

```bash
# Sweep over learning rates
python src/train.py -m \
    experiment=HiDEC_with_BERT \
    model_hparams.learning_rate=0.0001,0.00005,0.00001

# Sweep over multiple parameters
python src/train.py -m \
    experiment=HiDEC_with_BERT \
    batch_size=32,64,128 \
    model_hparams.learning_rate=0.0001,0.00005
```

### Working Directory

By default, Hydra doesn't change the working directory. Output logs are saved to:
```
outputs/YYYY-MM-DD/HH-MM-SS/
```

### Configuration Composition

Create custom experiment configs by composing base configs:

```yaml
# config/experiment/my_experiment.yaml
defaults:
  - override /model: bert
  - _self_

batch_size: 64
path:
  dataset: MY_DATASET
  save: ./my_ckpt

model_hparams:
  learning_rate: 0.0001
```

Then run:
```bash
python src/train.py experiment=my_experiment
```

## Migration Notes

### Old Usage (JSON configs)
```bash
python src/train.py --config ./config/HiDEC_with_BERT.cfg --devices 0 1
```

### New Usage (Hydra YAML configs)
```bash
python src/train.py experiment=HiDEC_with_BERT trainer_hparams.devices=[0,1]
```

The old JSON config files (.cfg) are still available in the `config/` directory but are now deprecated in favor of the new YAML-based Hydra configs.

## Documentation

For more information about Hydra configuration:
- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
