from argparse import ArgumentParser
from pathlib import Path

from helper.configure import Configure
from helper.vocab import Vocab
from utils.tree import get_tree
from modeling.model import HiDEC
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import warnings
import torch
import pytorch_lightning as pl
import socket


def hierarchydict2flatdict(flat, hierarchy, prefix=""):
    for k,v in hierarchy.items():
        if type(v) == dict:
            if prefix == "":
                new_prefix = f"{prefix}{k}"
            else:
                new_prefix = f"{prefix}-{k}"
            hierarchydict2flatdict(flat, v, new_prefix)
        else:
            flat[prefix+k]=v

warnings.filterwarnings(action='ignore')

def hierarchy_configuration(config):
    tree, nodes = get_tree(config.path.hierachy_node, config.path.hierarchy_relation)
    label2child = {}
    max_len = 0
    max_depth = 0
    for k, v in tree.items():
        if not v.is_leaf:
            children = [3]+[nodes[i.name] for i in v.children]
            label2child[nodes[k]] = children
            if max_len < len(children):
                max_len = len(children)
        else:
            if v.depth > max_depth:
                max_depth = v.depth
    config.model_hparams.add("hierarchy_label2child",label2child)
    config.model_hparams.add("hierarchy_max_len",max_len)
    config.model_hparams.add("hierarchy_depth",max_depth+3)
    config.model_hparams.add("hierarchy_size",len(nodes))
    config.model_hparams.add("hierarchy_name",config.path.dataset)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, required=True,
                        help="training data dir. Should be pre-generated data.")
    parser.add_argument("--devices", nargs="+" ,type=int, default=None, required=False,
                        help="training data dir. Should be pre-generated data.")
    
    args = parser.parse_args()

    config = Configure(config_json_file=args.config)
    hierarchy_configuration(config)

    if args.devices != None:
        config.trainer_hparams.devices = args.devices

    corpus_vocab = Vocab(config,
                         min_freq=10,
                         max_size=60000)
    if config.model_hparams.encoder_type == "BERT":
        from utils.dataset import HiDECBERTDataset as Dataset
        from utils.dataset import HiDECBERTCollateFn as collate_fn
    else:
        from utils.dataset import HiDECDataset as Dataset
        from utils.dataset import HiDECCollateFn as collate_fn

    train_dataset = Dataset(config, corpus_vocab, "train")
    dev_dataset = Dataset(config, corpus_vocab, "dev")
    test_dataset = Dataset(config, corpus_vocab, "test")
    train_dataloader = DataLoader(train_dataset, config.batch_size, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, config.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, config.batch_size, collate_fn=collate_fn)

    metric_mask = []
    label = corpus_vocab.v2i["label"]
    
    for node in Path(config.path.target_node).open().readlines():
        idx = label[node.strip()]
        metric_mask.append(int(idx))

    metric_mask = torch.tensor(metric_mask)-5
    config.model_hparams.add("hierarchy_metric_mask",metric_mask)
    config.model_hparams.add("hierarchy_target_size",metric_mask.size(0))
    config.model_hparams.add("pretrained_embedding",config.path.embedding)
    config.model_hparams.add("token_map",corpus_vocab.v2i['token'])

    total_step = (len(train_dataset)*config.trainer_hparams.max_epochs)/(len(config.trainer_hparams.devices)*config.trainer_hparams.accumulate_grad_batches*config.batch_size)
    total_step = int(total_step)
    config.model_hparams.add("total_step", total_step)

    model = HiDEC(**config.model_hparams.__dict__)


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.path.save}/{config.path.dataset}-{config.path.hparam_summary}-{config.trainer_hparams.devices}-{socket.gethostname()}",
        filename="{epoch}-{micro_f1:.4f}",
        verbose=True,
        save_last=False,
        save_top_k=10,
        monitor='micro_f1',
        mode='max'
    )

    wandb_jobtype = config.path.hparam_summary
    wandb_project = "HiDEC_PL"
    wandb_group = config.path.dataset
    wandb_logger = WandbLogger(project=wandb_project, group=wandb_group, job_type=wandb_jobtype)

    config.trainer_hparams.add("callbacks", [checkpoint_callback])
    config.trainer_hparams.add("logger", wandb_logger)
    config.trainer_hparams.add("replace_sampler_ddp", True)
    config.trainer_hparams.add("strategy", DDPStrategy(find_unused_parameters=True))

    trainer = pl.Trainer(**config.trainer_hparams.__dict__)

    trainer.fit(model, train_dataloader, dev_dataloader)

    trainer.test(model, test_dataloader, ckpt_path="best", verbose=True)
    
if __name__ =="__main__":
    main()