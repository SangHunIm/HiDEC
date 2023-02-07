import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modeling.encoder_modules import *
from modeling.transformer_modules import *
from helper.train_helper import *
from utils.adamw import AdamW
from torchmetrics import F1Score, RetrievalNormalizedDCG, RetrievalRecall, Accuracy
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from modeling.embedding import EmbeddingLayer
from pathlib import Path

encoder_types = {
                "GRUOrig" : GRUOriginalEncoder,
                "BERT" : BERTEncoder,
                }

multi_parent_map = {}


def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask

def eurlexLabelFilter(labels):
    result = []
    for l in labels:
        if labels in multi_parent_map and multi_parent_map[labels] not in result:
            result.append(multi_parent_map[labels])
        else:
            result.append(l)
    return result

class HiDEC(pl.LightningModule):
    def __init__(self,
                vocab_size,
                text_embedding_size,
                label_embedding_size,
                pretrained_embedding,
                encoder_type,
                encoder_n_layers,
                encoder_hidden_size,
                encoder_bidirectional,
                encoder_layer_norm,
                decoder_n_layers,
                decoder_hidden_size,
                decoder_ffn_size,
                decoder_residual,
                dropout,
                hierarchy_name,     
                hierarchy_size,     
                hierarchy_depth,
                hierarchy_label2child,
                hierarchy_max_len,
                hierarchy_metric_mask,
                hierarchy_target_size,
                learning_rate,
                weight_decay,
                warmup_proportion,
                adam_epsilon,
                total_step,
                token_map=None,
                emb_scale=1,
                padding_idx=0):
        super().__init__()

        if encoder_type != "BERT":
            self.word_embedding = EmbeddingLayer(
                vocab_map=token_map,
                embedding_dim=text_embedding_size,
                vocab_name='pretrained embeddings',
                padding_index=padding_idx,
                pretrained_dir=pretrained_embedding,
                initial_type="kaiming_uniform"
            )
        
        self.hierarchy_embedding = nn.Embedding(hierarchy_size, label_embedding_size)
        self.hierarchy_level_embedding = nn.Embedding(hierarchy_depth+3, label_embedding_size, padding_idx=padding_idx)
        nn.init.normal_(self.hierarchy_embedding.weight, mean=0,
                        std=label_embedding_size**-0.5)
        nn.init.normal_(self.hierarchy_level_embedding.weight, mean=0,
                        std=label_embedding_size**-0.5)

        if encoder_type != "BERT":
            self.encoder = encoder_types[encoder_type](
                                                    encoder_n_layers,
                                                    text_embedding_size,
                                                    encoder_hidden_size,
                                                    encoder_bidirectional,
                                                    dropout,
                                                    encoder_layer_norm)
            self.encoder_embedding_dropout = nn.Dropout(dropout)
        else:
            self.encoder = encoder_types[encoder_type]()

        self.decoder = Decoder(decoder_hidden_size, decoder_ffn_size, dropout, n_layers=decoder_n_layers, residual=decoder_residual)
        self.decoder_embedding_dropout = nn.Dropout(dropout)

        if hierarchy_name == "WOS46985":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

        self.acc_metric = Accuracy(average="samples")
        self.f1_metric = F1Score(average="samples")
        
        self.micro_f1 = F1Score(num_classes=hierarchy_target_size, multiclass=False, average="micro")
        self.macro_f1 = F1Score(num_classes=hierarchy_target_size, multiclass=False, average="macro")
        if hierarchy_name in ["EURLEX57K"]:
            self.nDCG = RetrievalNormalizedDCG(k=5)
            self.R5 = RetrievalRecall(k=5)

        
        self.register_buffer("LONGTENSOR", torch.LongTensor([1]))
        self.register_buffer("FLOATENSOR", torch.FloatTensor([1]))
        self.register_buffer("BOOLTENSOR", torch.BoolTensor([True]))
        self.save_hyperparameters(ignore=["token_map"])

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        src_input_ids = batch["src_input_ids"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_level_ids = batch["tgt_level_ids"]
        tgt_self_level_mask = batch["tgt_self_level_mask"]
        tgt_child = batch["tgt_child"]
        tgt_position = batch["tgt_position"]
        tgt_child_num = batch["tgt_child_num"]
        tgt_golden = batch["tgt_golden"]
        if self.hparams.encoder_type == "BERT":
            i_mask = create_pad_mask(src_input_ids["input_ids"], 0)
        else:
            i_mask = create_pad_mask(src_input_ids, 0)
        enc_output = self.encode(src_input_ids, i_mask)

        if self.hparams.encoder_type == "BERT":
            i_mask[src_input_ids["input_ids"].unsqueeze(1)==0] = True
        else:
            i_mask[src_input_ids.unsqueeze(1)==0] = True
        t_mask = create_pad_mask(tgt_input_ids, 0).type_as(self.LONGTENSOR)
        logits, _, _ = self.decode(tgt_input_ids, tgt_level_ids, enc_output, i_mask, tgt_self_level_mask, t_mask)
        logits = self.train_sub_hierarchy_expansion(logits, tgt_child, tgt_position, tgt_child_num)
        if self.hparams.hierarchy_name == "WOS46985":
            idx = 0
            loss_list = []
            for child_num in tgt_child_num:
                logit = logits[idx:idx+child_num].view(1, -1)
                golden = tgt_golden[idx:idx+child_num].view(1, -1)
                loss_list.append(self.loss_fct(logit,golden.to(torch.float)))
                idx+=child_num
            loss = torch.stack(loss_list).mean()
        else:
            loss = self.loss_fct(logits, tgt_golden.to(torch.float))
        

        self.acc_metric(logits, tgt_golden)
        self.f1_metric(logits, tgt_golden)

        training_step_outputs = {"loss":loss, "acc":self.acc_metric, "f1":self.f1_metric}

        self.log_dict(training_step_outputs, sync_dist=True)

        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        targets = batch["tgt_input_ids"]
        preds = self.sub_hiearchy_expansion(batch=batch)

        return (preds, targets)
    
    def validation_epoch_end(self, outputs):
        preds = []
        targets = []
        for pred, target in outputs:
            if self.hparams.hierarchy_name in ["RCV1-v2", "WOS46985", "NYT"]:
                target = extract_labels(target)
                pred = extract_labels(pred)
            else:
                target = extract_fine_grained_labels(target)
                pred = extract_fine_grained_labels(pred)
            preds += pred
            targets += target

        preds = self.convert_to_multi_hot_vector(delete_pad(preds))
        preds = torch.LongTensor(preds)
        preds = preds[:,self.hparams.hierarchy_metric_mask]

        targets = self.convert_to_multi_hot_vector(delete_pad(targets))
        targets = torch.LongTensor(targets)
        targets = targets[:,self.hparams.hierarchy_metric_mask]

        micro_f1 = f1_score(targets, preds,average="micro", zero_division=0)
        macro_f1 = f1_score(targets, preds,average="macro", zero_division=0)
        validation_epoch_outputs = {"micro_f1":micro_f1, 
                                    "macro_f1":macro_f1}
        if self.hparams.hierarchy_name in ["EURLEX57K"]:
            r5 = mean_recall_k(targets.numpy(), preds.numpy(), 5)
            ndcg = mean_ndcg_score(targets.numpy(), preds.numpy(), 5)
            validation_epoch_outputs["nDCG@5"] = ndcg.item()
            validation_epoch_outputs["R@5"]=r5.item()
        self.log_dict(validation_epoch_outputs, sync_dist=True)


    def test_step(self, batch, batch_idx):
        targets = batch["tgt_input_ids"]
        preds = self.sub_hiearchy_expansion(batch=batch)
        return (preds, targets)

    def test_epoch_end(self, outputs):
        preds = []
        targets = []
        for pred, target in outputs:
            if self.hparams.hierarchy_name in ["RCV1-v2", "WOS46985", "NYT"]:
                target = extract_labels(target)
                pred = extract_labels(pred)
            else:
                target = extract_fine_grained_labels(target)
                pred = extract_fine_grained_labels(pred)
            preds += pred
            targets += target
        save = Path(self.hparams.hierarchy_name+"_outputs.txt").open("w")
        for t, p in zip(targets, preds):
            save.write(f"{t}\t{p}\n")
        save.close()

        preds = self.convert_to_multi_hot_vector(delete_pad(preds))
        preds = torch.LongTensor(preds)
        preds = preds[:,self.hparams.hierarchy_metric_mask]

        targets = self.convert_to_multi_hot_vector(delete_pad(targets))
        targets = torch.LongTensor(targets)
        targets = targets[:,self.hparams.hierarchy_metric_mask]

        micro_f1 = f1_score(targets, preds,average="micro", zero_division=0)
        macro_f1 = f1_score(targets, preds,average="macro", zero_division=0)
        test_epoch_outputs = {"micro_f1_test":micro_f1, 
                              "macro_f1_test":macro_f1}
        if self.hparams.hierarchy_name in ["EURLEX57K"]:
            r5 = mean_recall_k(targets.numpy(), preds.numpy(), 5)
            ndcg = mean_ndcg_score(targets.numpy(), preds.numpy(), 5)
            test_epoch_outputs["nDCG@5_test"] = ndcg.item()
            test_epoch_outputs["R@5_test"]=r5.item()
        self.log_dict(test_epoch_outputs, sync_dist=True)
    
    def predict_step(self, batch, batch_idx):        
        sub_hierarchy_sequences = self.sub_hiearchy_expansion(batch=batch)
        return sub_hierarchy_sequences

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        def lr_lambda(current_step):
            num_training_steps = self.hparams.total_step
            num_warmup_steps = self.hparams.total_step*self.hparams.warmup_proportion
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        schedular = LambdaLR(optimizer, lr_lambda, -1)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": schedular,
                    "interval":"step"
                },
            }

    def encode(self, inputs, i_mask):
        if self.hparams.encoder_type != "BERT":
            input_embedded = self.word_embedding(inputs)
            tmp = i_mask.squeeze(1).unsqueeze(-1).to(torch.long)
            input_embedded *= (1-tmp)
            input_embedded *= self.hparams.emb_scale
            input_embedded = self.encoder_embedding_dropout(input_embedded)
            seq_len = (inputs!=0).sum(dim=1)

            return self.encoder(input_embedded, seq_len.cpu())
        else:
            return self.encoder(inputs)

    def decode(self, targets, tgt_level_embedding, enc_output, i_mask, t_self_mask, t_mask,
               cache=None):
        target_embedded = self.hierarchy_embedding(targets)
        tmp = t_mask.squeeze(1).unsqueeze(-1).type_as(self.LONGTENSOR)
        target_embedded *= (1-tmp)
        target_embedded *= self.hparams.emb_scale
        target_embedded += self.hierarchy_level_embedding(tgt_level_embedding)
        target_embedded = self.decoder_embedding_dropout(target_embedded)

        t_self_mask = t_self_mask==False
        
        decoder_output, self_att, enc_dec_att = self.decoder(target_embedded, enc_output, i_mask,
                                    t_self_mask, cache)
        return decoder_output, self_att, enc_dec_att

    def train_sub_hierarchy_expansion(self, logits, tgt_child, tgt_position, tgt_child_num):
        logits = logits[tgt_position]
        logits = torch.repeat_interleave(logits,tgt_child_num, dim=0) 
        tgt_childs = self.hierarchy_embedding(tgt_child)
        logits = logits * tgt_childs
        logits.mul_(self.hparams.emb_scale)
        logits = logits.sum(dim=1)
        return logits

    def sub_hiearchy_expansion(self, batch):
        src_input = batch["src_input_ids"]
        if self.hparams.encoder_type == "BERT":
            i_mask = create_pad_mask(src_input["input_ids"], 0)
        else:
            i_mask = create_pad_mask(src_input, 0)
        encoder_output = self.encode(src_input, i_mask.type_as(self.BOOLTENSOR))

        if self.hparams.encoder_type == "BERT":
            i_mask[src_input["input_ids"].unsqueeze(1)==0] = True
            batch = self.create_init_input(src_input["input_ids"].size(0))
        else:
            i_mask[src_input.unsqueeze(1)==0] = True
            batch = self.create_init_input(src_input.size(0))
        batch["enc_output"] = encoder_output
        batch["i_mask"] = i_mask

        flag = True
        level = 1
        for _ in range(self.hparams.hierarchy_depth):
            level += 1
            for k, v in batch.items():
                if v.type() == "torch.FloatTensor":
                    batch[k] = v.type_as(self.FLOATENSOR)
                elif v.type() == "torch.LongTensor":
                    batch[k] = v.type_as(self.LONGTENSOR)
                elif v.type() == "torch.BoolTensor":
                    batch[k] = v.type_as(self.BOOLTENSOR)
            logits, _, _ = self.decode(batch["tgt_input_ids"],
                                        batch["tgt_level_ids"], 
                                        batch["enc_output"], 
                                        batch["i_mask"], 
                                        batch["tgt_self_level_mask"], 
                                        batch["t_mask"])
            logits = logits[batch["tgt_position"]]
            logits = torch.repeat_interleave(logits, batch["tgt_child_num"], dim=0) 
            logits = logits.view(-1, self.hparams.hierarchy_max_len, self.hparams.label_embedding_size)
            tgt_childs = self.hierarchy_embedding(batch["tgt_child"]).view(-1, self.hparams.hierarchy_max_len, self.hparams.label_embedding_size)
            logits = logits * tgt_childs
            logits.mul_(self.hparams.emb_scale)
            logits = logits.sum(dim=-1)

            logits[batch["tgt_child"]==0]=1e-9

            if self.hparams.hierarchy_name == "WOS46985":
                _ , top1 = logits.topk(1)
                preds = torch.zeros_like(logits)
                for p, t in zip(preds, top1):
                    p[t.item()]=1
                preds.type_as(self.BOOLTENSOR)
            else:
                probs = logits.sigmoid()
                preds = probs > 0.5

            flag, batch = make_next_level_input(batch, preds, self.hparams.hierarchy_label2child, self.hparams.hierarchy_max_len, level)
            if not flag:
                break

        return batch["tgt_input_ids"]

    def create_init_input(self, batch_size, root=4):
        batch = {}
        batch["tgt_input_ids"] = torch.LongTensor([[1,root,2]]*batch_size)
        batch["tgt_level_ids"] = torch.LongTensor([[1,1,1]]*batch_size)
        batch["tgt_position"] = torch.BoolTensor([[False,True,False]]*batch_size)
        batch["tgt_child"] = torch.LongTensor([self.hparams.hierarchy_label2child[root]+[0 for _ in range(self.hparams.hierarchy_max_len-len(self.hparams.hierarchy_label2child[root]))]]*batch_size)
        batch["tgt_child_num"] = torch.LongTensor([self.hparams.hierarchy_max_len]*batch_size)
        new_tgt_self_level_mask = [create_tgt_level_mask(batch["tgt_input_ids"].tolist()[0], batch["tgt_level_ids"].tolist()[0])]*batch_size
        batch["tgt_self_level_mask"] = torch.BoolTensor(new_tgt_self_level_mask)
        batch["t_mask"] = create_pad_mask(batch["tgt_input_ids"], 0)
        return batch
    
    def convert_to_multi_hot_vector(self, labels):
        result = []
        for instance in labels:
            instance = torch.LongTensor(instance).type_as(self.LONGTENSOR) - 5 
            instance = instance[instance > -1]
            instance = instance.unsqueeze(0)
            instance = torch.zeros(instance.size(0), self.hparams.hierarchy_size-5).type_as(self.LONGTENSOR).scatter_(1,instance,1)
            result+=instance.tolist()
        return result

    