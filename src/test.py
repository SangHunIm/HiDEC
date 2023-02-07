from utils.dataset import HiDECDataset, HiDECCollateFn
from helper.configure import Configure
from helper.vocab import Vocab

from torch.utils.data import DataLoader
from modeling.model import HiDEC
from pytorch_lightning import Trainer

config = Configure(config_json_file="/home/ish/workspace/python_workspace/HiDEC/config/GRUEncoder_EURLEX57K.cfg")

corpus_vocab = Vocab(config,
                        min_freq=10,
                        max_size=60000)

tm = corpus_vocab.v2i['token']

dataset = HiDECDataset(config, corpus_vocab, "test")
dataloader = DataLoader(dataset, 256, collate_fn=HiDECCollateFn)

ckpt = ".ckpt" #CKPT file path

model = HiDEC.load_from_checkpoint(ckpt, token_map=tm)

trainer = Trainer(accelerator="gpu", strategy="ddp", devices=[0])

trainer.test(model=model, dataloaders=dataloader)