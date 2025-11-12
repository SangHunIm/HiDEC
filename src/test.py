"""
HiDEC Testing/Evaluation Script with Hydra Configuration
Usage:
    python src/test.py experiment=HiDEC_with_BERT checkpoint_path=path/to/model.ckpt
    python src/test.py experiment=HiDEC_with_GRU checkpoint_path=path/to/model.ckpt batch_size=256
"""

from omegaconf import DictConfig, OmegaConf
import hydra

from utils.dataset import HiDECDataset, HiDECCollateFn, HiDECBERTDataset, HiDECBERTCollateFn
from helper.vocab import Vocab
from torch.utils.data import DataLoader
from modeling.model import HiDEC
from pytorch_lightning import Trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Main testing function with Hydra configuration.

    Args:
        config: Hydra configuration object (DictConfig)
    """
    # Check if checkpoint path is provided
    if not hasattr(config, 'checkpoint_path') or config.checkpoint_path is None:
        raise ValueError(
            "Checkpoint path is required for testing. "
            "Please provide it via: python src/test.py experiment=... checkpoint_path=path/to/model.ckpt"
        )

    print("=" * 80)
    print("Testing Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Build vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=10,
                         max_size=60000)

    tm = corpus_vocab.v2i['token']

    # Select dataset and collate function based on encoder type
    if config.model_hparams.encoder_type == "BERT":
        dataset = HiDECBERTDataset(config, corpus_vocab, "test")
        collate_fn = HiDECBERTCollateFn
    else:
        dataset = HiDECDataset(config, corpus_vocab, "test")
        collate_fn = HiDECCollateFn

    # Create dataloader with configurable batch size
    test_batch_size = config.get('test_batch_size', config.batch_size)
    dataloader = DataLoader(dataset, test_batch_size, collate_fn=collate_fn)

    print(f"Loading model from checkpoint: {config.checkpoint_path}")
    model = HiDEC.load_from_checkpoint(config.checkpoint_path, token_map=tm)

    # Create trainer for testing
    trainer_devices = config.trainer_hparams.get('devices', [0])
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp" if len(trainer_devices) > 1 else "auto",
        devices=trainer_devices
    )

    # Run testing
    print("=" * 80)
    print("Running evaluation...")
    print("=" * 80)
    trainer.test(model=model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
