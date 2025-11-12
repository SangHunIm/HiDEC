"""
HiDEC Training Script with Hydra Configuration
Usage:
    python src/train.py experiment=HiDEC_with_BERT
    python src/train.py experiment=HiDEC_with_GRU
    python src/train.py experiment=HiDEC_with_BERT model_hparams.learning_rate=0.0001
    python src/train.py experiment=HiDEC_with_BERT trainer_hparams.devices=[0,1,2,3]
"""

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from helper.vocab import Vocab
from utils.tree import get_tree
from modeling.model import HiDEC
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from config_schema import HiDECConfig

import warnings
import torch
import pytorch_lightning as pl
import socket


# Register the config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=HiDECConfig)


def hierarchydict2flatdict(flat, hierarchy, prefix=""):
    """Convert nested hierarchy dict to flat dict with prefixed keys."""
    for k, v in hierarchy.items():
        if type(v) == dict:
            if prefix == "":
                new_prefix = f"{prefix}{k}"
            else:
                new_prefix = f"{prefix}-{k}"
            hierarchydict2flatdict(flat, v, new_prefix)
        else:
            flat[prefix + k] = v


warnings.filterwarnings(action='ignore')


def hierarchy_configuration(config: DictConfig):
    """Compute and add hierarchy-related configuration parameters."""
    tree, nodes = get_tree(config.path.hierachy_node, config.path.hierarchy_relation)
    label2child = {}
    max_len = 0
    max_depth = 0

    for k, v in tree.items():
        if not v.is_leaf:
            children = [3] + [nodes[i.name] for i in v.children]
            label2child[nodes[k]] = children
            if max_len < len(children):
                max_len = len(children)
        else:
            if v.depth > max_depth:
                max_depth = v.depth

    # Add computed hierarchy parameters to config
    config.model_hparams.hierarchy_label2child = label2child
    config.model_hparams.hierarchy_max_len = max_len
    config.model_hparams.hierarchy_depth = max_depth + 3
    config.model_hparams.hierarchy_size = len(nodes)
    config.model_hparams.hierarchy_name = config.path.dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function with Hydra configuration.

    Args:
        config: Hydra configuration object (DictConfig)
    """
    # Print configuration for debugging
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Configure hierarchy parameters
    hierarchy_configuration(config)

    # Build vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=10,
                         max_size=60000)

    # Select dataset and collate function based on encoder type
    if config.model_hparams.encoder_type == "BERT":
        from utils.dataset import HiDECBERTDataset as Dataset
        from utils.dataset import HiDECBERTCollateFn as collate_fn
    else:
        from utils.dataset import HiDECDataset as Dataset
        from utils.dataset import HiDECCollateFn as collate_fn

    # Create datasets
    train_dataset = Dataset(config, corpus_vocab, "train")
    dev_dataset = Dataset(config, corpus_vocab, "dev")
    test_dataset = Dataset(config, corpus_vocab, "test")

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, config.batch_size, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, config.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, config.batch_size, collate_fn=collate_fn)

    # Load target labels and create metric mask
    metric_mask = []
    label = corpus_vocab.v2i["label"]

    for node in Path(config.path.target_node).open().readlines():
        idx = label[node.strip()]
        metric_mask.append(int(idx))

    metric_mask = torch.tensor(metric_mask) - 5
    config.model_hparams.hierarchy_metric_mask = metric_mask
    config.model_hparams.hierarchy_target_size = metric_mask.size(0)
    config.model_hparams.pretrained_embedding = config.path.embedding
    config.model_hparams.token_map = corpus_vocab.v2i['token']

    # Calculate total training steps for warmup scheduling
    total_step = (len(train_dataset) * config.trainer_hparams.max_epochs) / \
                 (len(config.trainer_hparams.devices) *
                  config.trainer_hparams.accumulate_grad_batches *
                  config.batch_size)
    total_step = int(total_step)
    config.model_hparams.total_step = total_step

    # Create model
    # Convert DictConfig to dict for model initialization
    model_params = OmegaConf.to_container(config.model_hparams, resolve=True)
    model = HiDEC(**model_params)

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.path.save}/{config.path.dataset}-{config.path.hparam_summary}-{config.trainer_hparams.devices}-{socket.gethostname()}",
        filename="{epoch}-{micro_f1:.4f}",
        verbose=True,
        save_last=False,
        save_top_k=10,
        monitor='micro_f1',
        mode='max'
    )

    # Setup WandB logger
    wandb_jobtype = config.path.hparam_summary
    wandb_project = "HiDEC_PL"
    wandb_group = config.path.dataset
    wandb_logger = WandbLogger(project=wandb_project, group=wandb_group, job_type=wandb_jobtype)

    # Create trainer with all configurations
    trainer_params = OmegaConf.to_container(config.trainer_hparams, resolve=True)
    trainer_params["callbacks"] = [checkpoint_callback]
    trainer_params["logger"] = wandb_logger
    trainer_params["replace_sampler_ddp"] = True
    trainer_params["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(**trainer_params)

    # Train model
    trainer.fit(model, train_dataloader, dev_dataloader)

    # Test model
    trainer.test(model, test_dataloader, ckpt_path="best", verbose=True)


if __name__ == "__main__":
    main()
