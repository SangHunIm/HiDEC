import torch
import torch.nn as nn
import torch.nn.functional as F
from sru import SRU
from transformers import AutoConfig, AutoModel


class GRUOriginalEncoder(nn.Module):
    def __init__(self,
                 n_layers,
                 text_embedding_size,
                 encoder_hidden_size,
                 bidirectional,
                 dropout,
                 layer_norm
                 ):
        super(GRUOriginalEncoder, self).__init__()
        if bidirectional:
            hidden_size = int(encoder_hidden_size/2)
        else:
            hidden_size = encoder_hidden_size
        self.sru = nn.GRU(text_embedding_size, 
                            hidden_size, 
                            num_layers=n_layers, 
                            dropout=dropout, 
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, seq_len):
        encoder_output = inputs.transpose(0, 1)
        encoder_output, _ = self.sru(encoder_output)
        encoder_output = self.dropout(encoder_output)
        return encoder_output.transpose(0, 1)

class BERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, inputs):
        encoder_output = self.model(**inputs).last_hidden_state
        return encoder_output
