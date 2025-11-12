"""
Hydra configuration schema for HiDEC using dataclasses.
These structured configs provide type safety and better IDE support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from omegaconf import MISSING


@dataclass
class PathConfig:
    """Data and model paths configuration."""
    train: str = "data/pregenerated/train.jsonl"
    dev: str = "data/pregenerated/dev.jsonl"
    test: str = "data/pregenerated/test.jsonl"
    hierachy_node: str = "data/pregenerated/labels.txt"
    target_node: str = "data/pregeneratedtarget_labels.txt"
    hierarchy_relation: str = "data/pregenerated/parent_child_map.txt"
    vocab: str = "data/pregenerated/vocab.txt"
    embedding: str = "data/glove.6B.300d.txt"
    save: str = "./final_ckpt"
    dataset: str = "EURLEX57K"
    hparam_summary: str = MISSING


@dataclass
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    max_epochs: int = 100
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    accelerator: str = "gpu"
    strategy: str = "ddp"
    devices: List[int] = field(default_factory=lambda: [0, 1])
    num_sanity_val_steps: int = 0
    check_val_every_n_epoch: int = 1


@dataclass
class ModelConfig:
    """HiDEC model hyperparameters."""
    vocab_size: int = 60002
    text_embedding_size: int = 300
    label_embedding_size: int = 300
    encoder_type: str = MISSING  # "BERT" or "GRUOrig"
    encoder_n_layers: int = 1
    encoder_hidden_size: int = 300
    encoder_bidirectional: bool = True
    encoder_layer_norm: bool = True
    decoder_n_layers: int = 1
    decoder_hidden_size: int = 300
    decoder_ffn_size: int = 600
    decoder_residual: bool = False
    dropout: float = 0.1
    learning_rate: float = 0.0001
    weight_decay: float = 0.1
    warmup_proportion: float = 0.1
    adam_epsilon: float = 1e-08
    emb_scale: int = 1
    padding_idx: int = 0

    # These are added dynamically during training
    hierarchy_label2child: Optional[Dict[int, List[int]]] = None
    hierarchy_max_len: Optional[int] = None
    hierarchy_depth: Optional[int] = None
    hierarchy_size: Optional[int] = None
    hierarchy_name: Optional[str] = None
    hierarchy_metric_mask: Optional[Any] = None
    hierarchy_target_size: Optional[int] = None
    total_step: Optional[int] = None


@dataclass
class HiDECConfig:
    """Main configuration for HiDEC training."""
    path: PathConfig = field(default_factory=PathConfig)
    batch_size: int = 32
    end_token_for_coarse_grained_labels: bool = False
    trainer_hparams: TrainerConfig = field(default_factory=TrainerConfig)
    model_hparams: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    name: str = MISSING  # Dataset name (e.g., EURLEX57K)
    raw_dir: str = MISSING  # Raw data directory
    hierarchy_file: str = MISSING  # Hierarchy structure file
    save_dir: str = MISSING  # Output directory
