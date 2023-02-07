from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import torch
import numpy as np
import copy
import re
from utils.tree import nodes2vector, get_tree
from transformers import AutoTokenizer


english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]

def create_tgt_level_mask(seq, level):
        index = [x for x in range(len(level))]
        level_copy = copy.copy(level)
        tmp = []
        level_len = len(level)

        while len(level) > 0:
            t = [0 for x in range(level_len)]
            for i in range(len(level)):
                if level[i] == level[i+1] and level[i+1] == level[i+2]:
                    level.pop(i)
                    level.pop(i)
                    level.pop(i)
                    t[index.pop(i)] = 1
                    t[index.pop(i)] = 1
                    t[index.pop(i)] = 1
                    break
            tmp.append(t)

        tmp = sorted(tmp, key=lambda x: x.index(1))
        tmp = [np.array(t) for t in tmp]

        result = []
        level = level_copy

        for s, l in zip(seq, level):
            l = l-1
            prev = np.zeros(len(level), dtype=int)
            if s == 2:
                for t in tmp[:l+1]:
                    prev += t
                result.append(prev)
                tmp.pop(l)
            else:
                for t in tmp[:l+1]:
                    prev += t
                result.append(prev)
        tgt_self_mask = [x.tolist() for x in result]
        return tgt_self_mask

def cleaning(src):
    string = src
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string

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

def BasicTokenizer(src, vocab):
    tokens = [word.lower() for word in src.split() if word not in english_stopwords and len(word) > 1]
    ids = [vocab[to] for to in tokens]
    return tokens, ids

def transformersTokenizer(src, tokenizer):
    ids = tokenizer(src)["input_ids"]
    return None, ids


class HiDECDataset(Dataset):
    def __init__(self, config, vocab, task):
        lines = Path(config.path[task]).open().readlines()
        self.config = config
        self.instances = {i:None for i in range(len(lines))}
        self.lines = lines
        self.hierarchy, self.label2id = get_tree(config.path.hierachy_node, config.path.hierarchy_relation)
        self.vocab = vocab

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        if self.instances[i]==None:
            article = json.loads(self.lines[i])

            src_input_ids = [self.vocab.v2i["token"][t] if t in self.vocab.v2i["token"] else self.vocab.v2i["token"]["[UNK]"] for t in article["token"]]
            if self.config.end_token_for_coarse_grained_labels:
                target_labels = article["labels"]
            else:
                target_labels = article["labels"]
                for label in article["labels"]:
                    if self.hierarchy[label].parent.name in target_labels:
                        target_labels.pop(target_labels.index(self.hierarchy[label].parent.name))
            
            tgt_seq, tgt_input_ids, tgt_level_ids, tgt_position, target_children, target_golden = nodes2vector(target_labels, self.hierarchy, self.label2id)
            tgt_childs = [] 
            tgt_child_num = [] 
            tgt_goldens = [] 
            for child, golden in zip(target_children, target_golden):
                tgt_child_num.append(len(child))
                tmp_golden = [0 for _ in range(len(child))]
                for g in golden:
                    tmp_golden[child.index(g)] = 1
                tgt_childs.append(child)
                tgt_goldens.append(tmp_golden)
        
            tgt_child = sum(tgt_childs, [])
            tgt_golden = sum(tgt_goldens,[])

            tgt_self_level_mask = create_tgt_level_mask(tgt_input_ids, copy.copy(tgt_level_ids))

            instance = {
                    "src_input_ids" : src_input_ids,
                    "tgt_input_ids" : tgt_input_ids,
                    "tgt_level_ids" : tgt_level_ids,
                    "tgt_position" : tgt_position,
                    "tgt_child" : tgt_child,
                    "tgt_child_num" : tgt_child_num,
                    "tgt_golden" : tgt_golden,
                    "tgt_self_level_mask" : tgt_self_level_mask
                }
            self.instances[i]=instance
        else:
            instance = self.instances[i]
        return instance
            
def HiDECCollateFn(inputs):
    outputs = {
                "src_input_ids" : [],
                "tgt_input_ids" : [],
                "tgt_level_ids" : [],
                "tgt_position" : [],
                "tgt_child" : [],
                "tgt_child_num" : [],
                "tgt_golden" : [],
                "tgt_self_level_mask" : []
            }
    src_max_len = 0
    tgt_max_len = 0
    for i in inputs:
        src_max_len = max(src_max_len, len(i["src_input_ids"]))
        tgt_max_len = max(tgt_max_len, len(i["tgt_input_ids"]))
    src_max_len = min(src_max_len, 512)

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

    for i in inputs:
        for k, v in i.items():
            if k in ["src_input_ids"]:
                outputs[k].append(trancate(v,src_max_len, 0))
            elif k == "tgt_input_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_level_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_position":
                outputs[k].append(trancate(v,tgt_max_len, False))
            elif k == "tgt_self_level_mask":
                outputs[k].append(mask_trancate(v, tgt_max_len, 0))
            else:
                outputs[k] += v

    for k, v in outputs.items():
        if k in ["tgt_position", "tgt_self_level_mask"]:
            outputs[k] = torch.BoolTensor(v)
        else:
            outputs[k] = torch.LongTensor(v)

    return outputs


class HiDECBERTDataset(Dataset):
    def __init__(self, config, vocab, task):
        lines = Path(config.path[task]).open().readlines()
        self.instances = {i:None for i in range(len(lines))}
        self.lines = lines
        self.hierarchy, self.label2id = get_tree(config.path.hierachy_node, config.path.hierarchy_relation)
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        if self.instances[i]==None:
            article = json.loads(self.lines[i])

            src_input_ids = self.tokenizer(article["text"], truncation=True)
            _, tgt_input_ids, tgt_level_ids, tgt_position, target_children, target_golden = nodes2vector(article["labels"], self.hierarchy, self.label2id)
            tgt_childs = [] 
            tgt_child_num = [] 
            tgt_goldens = [] 
            for child, golden in zip(target_children, target_golden):
                tgt_child_num.append(len(child))
                tmp_golden = [0 for _ in range(len(child))]
                for g in golden:
                    tmp_golden[child.index(g)] = 1
                tgt_childs.append(child)
                tgt_goldens.append(tmp_golden)
        
            tgt_child = sum(tgt_childs, [])
            tgt_golden = sum(tgt_goldens,[])

            tgt_self_level_mask = create_tgt_level_mask(tgt_input_ids, copy.copy(tgt_level_ids))

            instance = {
                    "src_input_ids" : src_input_ids,
                    "tgt_input_ids" : tgt_input_ids,
                    "tgt_level_ids" : tgt_level_ids,
                    "tgt_position" : tgt_position,
                    "tgt_child" : tgt_child,
                    "tgt_child_num" : tgt_child_num,
                    "tgt_golden" : tgt_golden,
                    "tgt_self_level_mask" : tgt_self_level_mask
                }
            self.instances[i]=instance
        else:
            instance = self.instances[i]
        return instance
            
def HiDECBERTCollateFn(inputs):
    outputs = {
                "src_input_ids" : {"input_ids":[], "token_type_ids":[], "attention_mask":[]},
                "tgt_input_ids" : [],
                "tgt_level_ids" : [],
                "tgt_position" : [],
                "tgt_child" : [],
                "tgt_child_num" : [],
                "tgt_golden" : [],
                "tgt_self_level_mask" : []
            }
    src_max_len = 0
    tgt_max_len = 0
    for i in inputs:
        src_max_len = max(src_max_len, len(i["src_input_ids"]["input_ids"]))
        tgt_max_len = max(tgt_max_len, len(i["tgt_input_ids"]))
    src_max_len = min(src_max_len, 512)

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

    for i in inputs:
        for k, v in i.items():
            if k in ["src_input_ids"]:
                for kk, vv in v.items():
                    outputs[k][kk].append(trancate(vv,src_max_len, 0))
            elif k == "tgt_input_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_level_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_position":
                outputs[k].append(trancate(v,tgt_max_len, False))
            elif k == "tgt_self_level_mask":
                outputs[k].append(mask_trancate(v, tgt_max_len, 0))
            else:
                outputs[k] += v

    for k, v in outputs.items():
        if k in ["tgt_position", "tgt_self_level_mask"]:
            outputs[k] = torch.BoolTensor(v)
        elif k == "src_input_ids":
            for kk, vv in v.items():
                outputs[k][kk] = torch.LongTensor(vv)
        else:
            outputs[k] = torch.LongTensor(v)

    return outputs


class HiDECWOSDataset(Dataset):
    def __init__(self, config, vocab, task):
        lines = Path(config.path[task]).open().readlines()
        self.config = config
        self.instances = {i:None for i in range(len(lines))}
        self.lines = lines
        self.hierarchy, self.label2id = get_tree(config.path.hierachy_node, config.path.hierarchy_relation)
        self.vocab = vocab

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        if self.instances[i]==None:
            article = json.loads(self.lines[i])

            src_input_ids = [self.vocab.v2i["token"][t] if t in self.vocab.v2i["token"] else self.vocab.v2i["token"]["[UNK]"] for t in article["token"]]
            if self.config.end_token_for_coarse_grained_labels:
                target_labels = article["labels"]
            else:
                target_labels = article["labels"]
                for label in article["labels"]:
                    if self.hierarchy[label].parent.name in target_labels:
                        target_labels.pop(target_labels.index(self.hierarchy[label].parent.name))
            tgt_seq, tgt_input_ids, tgt_level_ids, tgt_position, target_children, target_golden = nodes2vector(article["labels"], self.hierarchy, self.label2id)
            tgt_childs = [] 
            tgt_child_num = [] 
            tgt_goldens = [] 
            for child, golden in zip(target_children, target_golden):
                tgt_child_num.append(len(child))
                tmp_golden = [0 for _ in range(len(child))]
                for g in golden:
                    tmp_golden[child.index(g)] = 1
                tgt_childs.append(child)
                tgt_goldens.append(tmp_golden)
        
            #tgt_child = sum(tgt_childs, [])
            #tgt_golden = sum(tgt_goldens,[])

            tgt_self_level_mask = create_tgt_level_mask(tgt_input_ids, copy.copy(tgt_level_ids))

            instance = {
                    "src_input_ids" : src_input_ids,
                    "tgt_input_ids" : tgt_input_ids,
                    "tgt_level_ids" : tgt_level_ids,
                    "tgt_position" : tgt_position,
                    "tgt_child" : tgt_childs,
                    "tgt_child_num" : tgt_child_num,
                    "tgt_golden" : tgt_goldens,
                    "tgt_self_level_mask" : tgt_self_level_mask
                }
            self.instances[i]=instance
        else:
            instance = self.instances[i]
        return instance
            
def HiDECWOSCollateFn(inputs):
    outputs = {
                "src_input_ids" : [],
                "tgt_input_ids" : [],
                "tgt_level_ids" : [],
                "tgt_position" : [],
                "tgt_child" : [],
                "tgt_child_num" : [],
                "tgt_golden" : [],
                "tgt_self_level_mask" : []
            }
    src_max_len = 0
    tgt_max_len = 0
    for i in inputs:
        src_max_len = max(src_max_len, len(i["src_input_ids"]))
        tgt_max_len = max(tgt_max_len, len(i["tgt_input_ids"]))
    src_max_len = min(src_max_len, 512)
    child_max = max()

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

    for i in inputs:
        for k, v in i.items():
            if k in ["src_input_ids"]:
                outputs[k].append(trancate(v,src_max_len, 0))
            elif k == "tgt_input_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_level_ids":
                outputs[k].append(trancate(v,tgt_max_len, 0))
            elif k == "tgt_position":
                outputs[k].append(trancate(v,tgt_max_len, False))
            elif k == "tgt_self_level_mask":
                outputs[k].append(mask_trancate(v, tgt_max_len, 0))
            else:
                outputs[k] += v

    for k, v in outputs.items():
        if k in ["tgt_position", "tgt_self_level_mask"]:
            outputs[k] = torch.BoolTensor(v)
        else:
            outputs[k] = torch.LongTensor(v)

    return outputs