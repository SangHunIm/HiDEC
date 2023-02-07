from pathlib import Path
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.dataset_rcv import HiDECDataset as Dataset
from utils.dataset_rcv import HiDECCollateFn as collate_fn
from utils.dataset_rcv import create_tgt_level_mask
from utils.metric import AccuracyThresh, AverageMeter, F1Score
from modeling import utils

def insert_list_in_list(s, t, p):
    return s[:p+1] + t + s[p+1:]
    
def trancate(data, max_len, pad):
    if len(data) >= max_len:
        return data[:max_len]
    else:
        data += [pad for x in range(max_len-len(data))]
        return data

def mask_trancate(data, max_len, pad):
    for i in range(max_len):
        try:
            data[i]+=[pad for x in range(max_len-len(data[i]))]
        except IndexError:
            data.append([pad for x in range(max_len)])
    return data

def delete_pad(data, pad=0):
    for i, b in enumerate(data):
        for j, d in enumerate(b):
            if d == pad:
                data[i] = data[i][:j]
    return data

def make_next_level_input(batch, preds, label2child, max_len, next_level, device):
    new_tgt_input_ids = delete_pad(batch["tgt_input_ids"].tolist())
    new_tgt_level_ids = delete_pad(batch["tgt_level_ids"].tolist())
    old_tgt_position = batch["tgt_position"].tolist()
    new_tgt_position = (batch["tgt_position"] & False).tolist()
    new_tgt_child = []
    new_tgt_child_num = []
    old_tgt_child = batch["tgt_child"].tolist()
    preds = preds.tolist()
    target_idx = -1

    for b_idx, tgt_position in enumerate(old_tgt_position):
        tmp = -1
        for i, position in enumerate(tgt_position):
            if position == False:
                continue
            target_idx += 1
            for j, pred in enumerate(preds[target_idx]):
                if pred == False:
                    continue
                child_idx = old_tgt_child[target_idx][j]
                if child_idx == 0:
                    break
                elif child_idx == 3:
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, False, False], i+3*tmp)
                elif child_idx not in label2child:
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 1, 3, 2, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level+1, next_level+1, next_level+1, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, False, False, False, False, False], i+3*tmp)
                    tmp+=1
                else:
                    new_childs = label2child[child_idx]
                    new_childs += [0 for _ in range(max_len-len(new_childs))]
                    new_tgt_child.append(new_childs)
                    new_tgt_child_num.append(len(new_childs))
                    tmp+=1
                    new_tgt_input_ids[b_idx] = insert_list_in_list(new_tgt_input_ids[b_idx], [1, child_idx, 2], i+3*tmp)
                    new_tgt_level_ids[b_idx] = insert_list_in_list(new_tgt_level_ids[b_idx], [next_level, next_level, next_level], i+3*tmp)
                    new_tgt_position[b_idx] = insert_list_in_list(new_tgt_position[b_idx], [False, True, False], i+3*tmp)

    max_seq = max([len(s) for s in new_tgt_input_ids])
    try:
        new_tgt_self_level_mask = [mask_trancate(create_tgt_level_mask(i, copy.copy(l)), max_seq, False) for i, l in zip(new_tgt_input_ids, new_tgt_level_ids)]
    except:
        print(new_tgt_level_ids)
        print(new_tgt_input_ids)

    for i in range(len(new_tgt_input_ids)):
        new_tgt_input_ids[i] = trancate(new_tgt_input_ids[i], max_seq, 0)
        new_tgt_level_ids[i] = trancate(new_tgt_level_ids[i], max_seq, 0)
        new_tgt_position[i] = trancate(new_tgt_position[i], max_seq, 0)

    batch["tgt_input_ids"] = torch.LongTensor(new_tgt_input_ids).cuda()
    batch["tgt_level_ids"] = torch.LongTensor(new_tgt_level_ids).cuda()
    batch["tgt_position"] = torch.BoolTensor(new_tgt_position).cuda()
    batch["tgt_child"] = torch.LongTensor(new_tgt_child).cuda()
    batch["tgt_child_num"] = torch.LongTensor(new_tgt_child_num).cuda()
    batch["tgt_self_level_mask"] = torch.BoolTensor(new_tgt_self_level_mask).cuda()
    
    if len(new_tgt_child) == 0:
        flag = False
    else:
        flag = True

    return flag, batch

def extract_leaf(tgt):
    import time
    result = []
    tgt = delete_pad(tgt.tolist())
    for instance in tgt:
        a = []
        stop = 0
        while 3 in instance:
            for i, ids in enumerate(instance):
                if instance[i] == 1 and instance[i+2]==1 and instance[i+3]==3 and instance[i+4]==2 and instance[i+5]==2:
                    a.append(instance[i+1])
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
                elif instance[i] == 1 and instance[i+2]==2 and instance[i+3]==1 and instance[i+4]==3 and instance[i+5]==2:
                    a.append(instance[i+1])
                    instance.pop(i)
                    instance.pop(i)
                    instance.pop(i)
            stop +=1
            if stop >10:
                print(instance)
                break
        result.append(a)
    return result

class Trainer():
    def __init__(self, cfg, model, optimizer, scheduler, local_rank) -> None:
        self.cfg = cfg
        self.model = model
        self.train_files = list(Path(cfg.path_config.train_dir).iterdir())
        self.dev_files = list(Path(cfg.path_config.train_dir).iterdir())
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_one_step(self, batch):
        self.model.train()

        for k, v in batch.items():
            batch[k] = v.cuda()

        loss, logits = self.model(**batch)
        
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.learning_config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

        return loss, logits

    def train_one_epoch(self, epoch):
        acc_metric = AccuracyThresh()
        f1_metric = F1Score()
        tr_acc = AverageMeter()
        tr_f1 = AverageMeter()
        tr_loss = AverageMeter()
        train_logs = {}
        stime = time()

        for file in self.train_files:
            dataset = Dataset(str(file). self.cfg.path_config.vocab_path)
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch)
            loader = DataLoader(dataset, self.cfg.learning_config.lr, collate_fn=collate_fn, num_workers=4, sampler=sampler)
            for batch in loader:
                loss, logits = self.train_one_step(batch)
                acc_metric(logits, batch["tgt_golden"])
                f1_metric(logits, batch["tgt_golden"])
                
                tr_acc.update(acc_metric.value(), n=batch["tgt_golden"].size(0))
                tr_f1.update(f1_metric.value(), n=batch["tgt_golden"].size(0))
                tr_loss.update(loss.item(), n=1)
        
        eta = time() - stime
        if eta > 3600:
            eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
        elif eta > 60:
            eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
            eta_format = '%ds' % eta

        train_logs['loss'] = tr_loss.avg
        train_logs['acc'] = tr_acc.avg
        train_logs['f1'] = tr_f1.avg

        show_info = f'[Training]:[{epoch}/{self.cfg.learning_config.epochs}] ' \
                    f'- ETA: {eta_format}' + "-".join(
            [f' {key}: {value:.4f} ' for key, value in train_logs.items()])
        logger.info(show_info)

    def evaluate(self):
            self.model.eval()
            with torch.no_grad():
                for file in self.dev_files:
                    dataset = Dataset(str(file). self.cfg.path_config.vocab_path)
                    sampler = DistributedSampler(dataset)
                    loader = DataLoader(dataset, self.cfg.learning_config.lr, collate_fn=collate_fn, num_workers=4, sampler=sampler)
                    for batch in loader:
                        for k, v in batch.items():
                            batch[k] = v.cuda()

                        batch_size = batch["tgt_input_ids"].size(0)

                        # prepare Encoder inputs
                        src_input = batch["src_input_ids"]
                        i_mask = utils.create_pad_mask(src_input, 0)

                        # Encode text
                        encoder_outputs = self.model.encode(src_input, i_mask)  # [B, S, E]
                        # prepare level-1 Decoder inputs 
                        i_mask[src_input.unsqueeze(1)==0] = True
                        for i in extract_leaf(batch["tgt_input_ids"]):
                            tmp = [0 for _ in range(103)]
                            for k in i:
                                # 79 is root`
                                if k < 79:
                                    tmp[k-4] = 1
                                elif k > 79:
                                    tmp[k-5] = 1
                                else:
                                    continue
                            golden.append(tmp)
                        batch["tgt_input_ids"] = torch.LongTensor([[1,79,2]]*batch_size).cuda()
                        batch["tgt_level_ids"] = torch.LongTensor([[1,1,1]]*batch_size).cuda()
                        batch["tgt_position"] = torch.BoolTensor([[False,True,False]]*batch_size).cuda()
                        batch["tgt_child"] = torch.LongTensor([label2child[79]+[0 for _ in range(max_len-len(label2child[79]))]]*batch_size).cuda()
                        batch["tgt_child_num"] = torch.LongTensor([max_len]*batch_size).cuda()
                        new_tgt_self_level_mask = [create_tgt_level_mask(batch["tgt_input_ids"].tolist()[0], batch["tgt_level_ids"].tolist()[0])]*batch_size
                        batch["tgt_self_level_mask"] = torch.BoolTensor(new_tgt_self_level_mask).cuda()
                        t_mask = utils.create_pad_mask(batch["tgt_input_ids"], 0)

                        flag = True
                        level = 1
                        while flag:
                            level += 1
                            # Decode hierarchy
                            logits, self_att, enc_dec_att = model.decode(
                                                                        batch["tgt_input_ids"], 
                                                                        batch["tgt_level_ids"], 
                                                                        encoder_outputs, 
                                                                        i_mask, 
                                                                        batch["tgt_self_level_mask"], 
                                                                        t_mask
                                                                        )

                            logits = logits[batch["tgt_position"]]
                            logits = torch.repeat_interleave(logits,batch["tgt_child_num"], dim=0) 
                            logits = logits.view(-1, max_len, embedding_size)
                            tgt_childs = model.dec_vocab_embedding(batch["tgt_child"]).view(-1, max_len, embedding_size)
                            logits = logits * tgt_childs
                            logits.mul_(model.emb_scale)
                            logits = logits.sum(dim=-1)
                            probs = logits.sigmoid()
                            preds = probs > 0.5

                            f, batch = make_next_level_input(batch, preds, label2child, max_len, level, device)
                            flag = f
                            t_mask = utils.create_pad_mask(batch["tgt_input_ids"], 0)
                            
                            """
                            for k, v in batch.items():
                                print(k, v.size(), v.device)"""
                            
                        
                        for i in extract_leaf(batch["tgt_input_ids"],label2child):
                            tmp = [0 for _ in range(103)]
                            for k in i:
                                # 79 is root
                                if k < 79:
                                    tmp[k-4] = 1
                                elif k > 79:
                                    tmp[k-5] = 1
                                else:
                                    continue
                            prediction.append(tmp)

                    print("sample : ",f1_score(golden, prediction, average="samples", zero_division=0))
                    print("micro:", f1_score(golden, prediction, average="micro", zero_division=0))
                    print("macro:", f1_score(golden, prediction, average="macro", zero_division=0))
                    print("weighted:", f1_score(golden, prediction, average="weighted", zero_division=0))


    def train(self):
        for epoch in range(self.cfg.learning_config.epochs):
            self.train_one_epoch(epoch)
            self.evaluate(epoch)