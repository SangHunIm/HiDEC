## Requirements:

* anytree 2.8.0
* pytorch 1.11.0
* wandb 0.12.16
* transformers 4.18.0

### Data preprecossing

1. download EURLEX57K dataset from https://github.com/iliaschalkidis/lmtc-eurlex57k
2. run script 

```
python src/preprocessing.py --name=EURLEX57K --raw_dir=data/datasets/EURLEX57K/dataset --save_dir=data/pregenerated  --hierarchy_file=data/datasets/EURLEX57K/EURLEX57K.json
```

### Training
```
python src/train.py --config ./config/HiDEC_with_BERT.cfg
```

