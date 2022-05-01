'''
Basic transformer code is borrowed from https://github.com/harvardnlp/annotated-transformer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np
from utils.nn_distance import nn_distance

from lib.config import CONF
from torch.autograd import Variable

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, keep_value=False):
        "Take in number of heads and model size."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.keep_value = keep_value

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        if self.keep_value:
            self.value = value

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    "Implements word embedding."
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm (pre-LN).
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionalEncoding(nn.Module):
    "sinusoidal positional encodings"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, src_pos=None):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionalEncodingLearned(nn.Module):
    """
    Learnable positional encoding inspired by Ze Liu et al., "Group-Free 3D Object Detection via Transformers"
    """

    def __init__(self, input_channel, d_model=128):
        #input_channel: 3 for xyz 6 for xyzwhl
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=1))

    def forward(self, x, xyz):
        return x + self.position_embedding_head(xyz.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        x = self.norm(x)
        return x

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, obj_indicator=None):
        if obj_indicator is not None:
            x = torch.cat((obj_indicator, x), dim=1)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn (optional: late_guide), and feed forward."
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, early_guide=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.early_guide = early_guide
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        if not self.early_guide: # late_guide
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, early_guide=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.early_guide = early_guide

    def forward(self, src, tgt, src_mask, tgt_mask, obj_indicator=None, src_pos=None, obj_idx=None):
        if self.encoder is None:
            return self.decode(self.src_embed(src, src_pos) if src_pos is not None else src, src_mask,
                               tgt, tgt_mask,
                               obj_indicator=obj_indicator, obj_idx=obj_idx)
        else:
            return self.decode(self.encode(src, src_pos, src_mask), src_mask,
                               tgt, tgt_mask,
                               obj_indicator=obj_indicator, obj_idx=obj_idx)

    def encode(self, src, src_pos, src_mask):
        return self.encoder(self.src_embed(src, src_pos), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, obj_indicator=None, obj_idx=None):
        if memory.shape[0] != tgt.shape[0]: # inference phase
            assert memory.shape[0]*memory.shape[1] == tgt.shape[0]
            B,K,_ = memory.shape
            obj_indicator = obj_indicator + memory.view(B*K, -1).unsqueeze(1)
            if self.early_guide:
                memory = torch.repeat_interleave(memory, memory.shape[1], dim=0)


        if obj_idx is not None: # training phase
            obj_indicator = obj_indicator + torch.gather(memory, 1, obj_idx.repeat(1, memory.size(-1)).unsqueeze(1))

        if self.early_guide:
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, obj_indicator=obj_indicator)
        else:
            return self.decoder(self.tgt_embed(tgt), obj_indicator, None, tgt_mask, None)

class TransformerDecoderModel(nn.Module):

    def make_model(self, tgt_vocab, N=6, h=8, d_model=512, d_ff=2048, dropout=0.1, bn_momentum=0.1,
                   src_pos_type=None,  use_transformer_encoder=False, early_guide=True):
        #src_pos_type: xyz | center | loc
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        if src_pos_type is not None:
            src_position = PositionalEncodingLearned(input_channel=3 if (src_pos_type=='xyz' or src_pos_type=='center') else 6, d_model=d_model)
        else:
            src_position = c(position) # use sin cos encoding

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, MultiHeadedAttention(h, d_model, keep_value=self.check_relation), c(ff), dropout), N) if use_transformer_encoder else None, #encoder
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, early_guide), N), #decoder
            src_position if self.use_transformer_encoder else lambda x:x,  #src_embed
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),  #tgt_embed
            Generator(d_model, tgt_vocab), early_guide=early_guide)

        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                m.momentum = bn_momentum

        return model

    def __init__(self, vocabulary, N, h, d_model, d_ff, transformer_dropout, bn_momentum=0.1,
                 src_pos_type = None, use_transformer_encoder=False, early_guide=True,
                 check_relation=False,
                 ):
        # src_pos_type: xyz, center, ...
        # check_relation: check relation between two proposal using their score-portion
        super(TransformerDecoderModel, self).__init__()

        self.word_to_idx = vocabulary['word2idx']
        self.src_pos_type = src_pos_type
        self.vocabulary = vocabulary
        self.use_transformer_encoder = use_transformer_encoder
        self.check_relation = check_relation
        self.early_guide = early_guide

        self.model = self.make_model(len(vocabulary['word2idx']),
            N=N, h=h, d_model=d_model, d_ff=d_ff, dropout=transformer_dropout, bn_momentum=bn_momentum,
            src_pos_type=src_pos_type, use_transformer_encoder=use_transformer_encoder, early_guide=early_guide)

        if check_relation:
            self.relation_proposal = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 9),
            )


    def _prepare_feature(self, seq=None):
        if self.early_guide:
            # crop the last one
            seq = seq[:,:-1]
        else:
            # crop the last one and the first obj indicator
            seq = seq[:, 1:-1]
        seq_mask = (seq.data > 0)
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)


        return seq[:, 1:] if self.early_guide else seq, seq_mask #discard the first object indicator

    def forward(self, data_dict, is_eval=False):
        if not is_eval:
            data_dict = self.forward_train(data_dict)
        else:
            data_dict = self.forward_eval(data_dict)

        return data_dict

    def forward_train(self, endpoints):
        # src: get the input token to transformer encoder
        src = endpoints['aggregated_vote_features']  # B, #proposal, C

        # src_pos: get the position of the input token to transformer encoder
        if self.src_pos_type == 'xyz':
            src_pos = endpoints['aggregated_vote_xyz']
        elif self.src_pos_type == 'center':
            src_pos = endpoints['center']
        elif self.src_pos_type == 'loc':
            src_pos = torch.cat([endpoints['center'], endpoints['pred_size']], -1)
        else:
            src_pos = None

        # ref_obj_feature: get the proposal nearest to the ref object
        _, _, target_ious, idx = nn_distance(endpoints['aggregated_vote_xyz'], endpoints['ref_center_label'].unsqueeze(1))  # B,1
        endpoints['match_idx'] = idx.squeeze(1) # B
        ref_obj_feature = torch.gather(src, 1, idx.repeat(1, src.size(-1)).unsqueeze(1))  # B,1, C

        # get the input token and mask to transformer decoder (note we append object indicator in later step)
        seq, seq_mask = self._prepare_feature(endpoints['lang_label'])

        out = self.model(src=src, tgt=seq,
                         src_mask=endpoints['bbox_mask'].unsqueeze(1), tgt_mask=seq_mask,
                         obj_indicator=ref_obj_feature, src_pos=src_pos, obj_idx=idx if self.use_transformer_encoder else None)

        out = out[:, 1:, :] if self.early_guide else out[:,:,:]  # exclude the object indicator part

        outputs = self.model.generator(out)
        endpoints['lang_cap'] = outputs


        good_bbox_masks = (target_ious > -1).squeeze(1)
        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        # store
        endpoints["pred_ious"] = mean_target_ious
        endpoints["good_bbox_masks"] = good_bbox_masks

        # Compute relation
        if self.check_relation:
            attn_matrix = self.model.encoder.layers[-1].self_attn.attn.unsqueeze(-1).repeat(1,1,1,1,16)
            value_matrix = self.model.encoder.layers[-1].self_attn.value.unsqueeze(-3)
            nbatches, nheads, _, nproposals, nsubfeatures = value_matrix.shape
            relation_feature = (attn_matrix*value_matrix).transpose(1,2).transpose(2,3).contiguous().view(nbatches,nproposals,nproposals,nheads*nsubfeatures)
            relation_pred = self.relation_proposal(relation_feature)
            endpoints['relation_pred'] = relation_pred

        return endpoints

    def forward_eval(self, endpoints):
        obj_features = endpoints["aggregated_vote_features"] # B, #proposal, C

        B, K, _ = obj_features.shape
        if not self.use_transformer_encoder:
            src = torch.repeat_interleave(obj_features, K, dim=0)  # B*#proposal, #proposal, C
        else:
            src = endpoints["aggregated_vote_features"]

        if self.src_pos_type == 'xyz':
            if not self.use_transformer_encoder:
                src_pos = torch.repeat_interleave(endpoints['aggregated_vote_xyz'], K, dim=0)
            else:
                src_pos = endpoints['aggregated_vote_xyz']
        elif self.src_pos_type == 'center':
            if not self.use_transformer_encoder:
                src_pos = torch.repeat_interleave(endpoints['center'], K, dim=0)
            else:
                src_pos = endpoints['center']
        elif self.src_pos_type == 'loc':
            if not self.use_transformer_encoder:
                src_pos = torch.repeat_interleave(torch.cat([endpoints['center'], endpoints['pred_size']], -1), K, dim=0)
            else:
                src_pos = torch.cat([endpoints['center'], endpoints['pred_size']], -1)
        else:
            src_pos = None

        obj_features = obj_features.view(B*K, -1)

        max_len = CONF.TRAIN.MAX_DES_LEN
        start_symbol = self.word_to_idx['sos']
        ys = torch.ones(B*K, 1).fill_(start_symbol).type_as(endpoints['heading_class_label'].data)

        for i in range(max_len+1):
            out = self.model(src, Variable(ys), endpoints['bbox_mask'].unsqueeze(1),
                             Variable(subsequent_mask(ys.size(1)+1).type_as(src.data)) if self.early_guide else Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
                             obj_features.unsqueeze(1), src_pos=src_pos)

            out = out[:,-1,:]

            assert len(out.shape) == 2
            prob = self.model.generator(out)
            _, next_word = torch.max(prob, dim=-1)
            assert next_word.size(0) == B*K
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

        outputs = ys[:, 1:].view(B, K, -1)

        # store
        endpoints["lang_cap"] = outputs
        return endpoints

