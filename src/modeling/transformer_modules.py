import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=arguments-differ

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=2):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        att = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if mask is not None:
            att += (mask.unsqueeze(1)*-1e9)
        att = torch.softmax(att, dim=3)
        x = self.att_dropout(att)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, (att,)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, residual_connection=False):
        super(EncoderLayer, self).__init__()

        self.residual_connection = residual_connection

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y, self_att = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        if self.residual_connection:
            x = x + y
        else:
            x = y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        if self.residual_connection:
            x = x + y
        else:
            x = y
            
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, residual_connection=False):
        super(DecoderLayer, self).__init__()
        self.residual_connection = residual_connection
    
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
        self.self_ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.self_ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        self_att = (None,)
        y = self.self_attention_norm(x)
        y, self_att = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        if self.residual_connection:
            x = x + y
        else:
            x = y
        
        y = self.self_ffn_norm(x)
        y = self.self_ffn(y)
        y = self.self_ffn_dropout(y)
        if self.residual_connection:
            x = x + y
        else:
            x = y

        enc_dec_att = (None,)

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y, enc_dec_att = self.enc_dec_attention(y, enc_output, enc_output, None,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            if self.residual_connection:
                x = x + y
            else:
                x = y
                
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        if self.residual_connection:
            x = x + y
        else:
            x = y

        return x, self_att, enc_dec_att

class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, residual):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate, residual_connection=residual)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        self_att = ()
        enc_dec_att = ()
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output, s_att, ed_att = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache)
            self_att += s_att
            enc_dec_att += ed_att

        return self.last_norm(decoder_output), self_att, enc_dec_att
        
if __name__ == "__main__":
    def func(n):
        r = 1
        for i in range(3, n+1):
            for j in range(2, i):
                if i % j == 0:
                    break
                else:
                    r += 1
        return r

    print(func(6))