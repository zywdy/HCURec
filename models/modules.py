import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int, cfg) -> None:
        super(SelfAttend, self).__init__()

        self.cfg = cfg
        self.emb_size = embedding_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.h1 = nn.Sequential(
            nn.Linear(self.emb_size, 200),   # 全连接层 [16, 300]--->[16, 200]
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(200, 1)   # [16, 200]--->[16, 1]

    def forward(self, seqs, seq_len, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        seqs = self.dropout(seqs)
        gates = self.gate_layer(self.h1(seqs.reshape(-1, self.emb_size))).reshape(-1, seq_len, 1).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=-2)
        return output


class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.word_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.word_dim, num_heads=cfg.word_head_num
        )
        self.cnn = nn.Conv1d(cfg.word_dim, cfg.word_dim, 3, padding='same')
        self.word_self_attend = SelfAttend(cfg.word_dim, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.word_layer_norm = nn.LayerNorm(cfg.word_dim)

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)    # [320, 30, 300]
        X = self.dropout(embs)    # [320, 30, 300]

        #  permute：将tensor的维度换位
        X1 = X.permute(1, 0, 2)  # [30, 320, 300]
        output1, _ = self.mh_self_attn(X1, X1, X1)
        output1 = output1.permute(1, 0, 2)
        output1 = self.dropout(output1)  # [320, 30, 300]
        # output = self.word_proj(output)
        X2 = X.permute(0, 2, 1)
        output2 = self.cnn(X2)
        output2 = output2.permute(0, 2, 1)
        output2 = self.dropout(output2)
        output = output1 + output2

        return self.word_layer_norm(output + X)

    def forward(self, seqs):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens, self.cfg.max_title_len)

        return self_attend


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.entity_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.entity_dim, num_heads=cfg.entity_head_num
        )
        self.word_self_attend = SelfAttend(cfg.entity_dim, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.word_layer_norm = nn.LayerNorm(cfg.entity_dim)

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)
        # output = self.word_proj(output)

        return self.word_layer_norm(output + X)

    def forward(self, seqs):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens, self.cfg.max_entity_len)

        return self_attend


class NewsEncoder(nn.Module):
    def __init__(self, cfg):
        super(NewsEncoder, self).__init__()
        self.cfg = cfg
        self.title_encoder = TitleEncoder(cfg)
        self.entity_encoder = EntityEncoder(cfg)

        self.trans = nn.Linear(cfg.word_dim + cfg.entity_dim, cfg.hidden_size)

    def forward(self, title_seq, entity_seq):

        title_seq = self.title_encoder(title_seq)
        entity_seq = self.entity_encoder(entity_seq)
        news_seq = torch.cat((title_seq, entity_seq), -1)

        news_seq = self.trans(news_seq)

        return news_seq


class PoEncoder(nn.Module):
    def __init__(self, cfg):
        super(PoEncoder, self).__init__()
        self.cfg = cfg
        self.entity_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.entity_emb), freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.word_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.entity_dim, num_heads=cfg.entity_head_num
        )
        self.entity_self_attend = SelfAttend(cfg.entity_dim*2, cfg)

        self.dropout = nn.Dropout(cfg.dropout)
        self.word_layer_norm = nn.LayerNorm(cfg.entity_dim)
        self.title_tran = nn.Linear(cfg.word_dim, cfg.entity_dim)
        self.po_trans = nn.Linear(cfg.entity_dim*2, cfg.hidden_size)

    def _extract_hidden_rep(self, entity_seq, title_seq):
        entity_embs = self.entity_embedding(entity_seq)
        X = self.dropout(entity_embs)

        X = X.permute(1, 0, 2)  # 5,8192,100
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)    # 8192,5,100
        output = self.dropout(output)
        X = X.permute(1, 0, 2)  # 8192,5,100
        result0 = self.word_layer_norm(output + X)
        result1 = result0.permute(1, 0, 2)  # 5,8192,100
        # output = self.word_proj(output)
        title_embs = self.word_embedding(title_seq)  # 8192,30,300
        X2 = self.dropout(title_embs)
        X2 = X2.permute(1, 0, 2)    # 30,8192,300
        X2 = self.title_tran(X2.reshape(-1, self.cfg.word_dim)).reshape(self.cfg.max_title_len, -1, self.cfg.entity_dim)    # 30,8192,100
        output2, _ = self.mh_self_attn(result1, X2, X2)   # 5,8192,100
        output2 = output2.permute(1, 0, 2)
        output2 = self.dropout(output2)

        result2 = self.word_layer_norm(output2 + result0)

        potential = torch.cat((result0, result2), -1)

        return potential

    def forward(self, entity_seq, title_seq):
        hiddens = self._extract_hidden_rep(entity_seq, title_seq)

        self_attend = self.entity_self_attend(hiddens, self.cfg.max_entity_len)
        entity_po = self.po_trans(self_attend)

        return entity_po


class Attend(nn.Module):
    def __init__(self, embedding_size: int, cfg) -> None:
        super(Attend, self).__init__()

        self.cfg = cfg
        self.emb_size = embedding_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.h1 = nn.Sequential(
            nn.Linear(self.emb_size, 200),   # 全连接层 [16, 300]--->[16, 200]
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(200, 1)   # [16, 200]--->[16, 1]

    def forward(self, seqs, seq_len, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        seqs = self.dropout(seqs)
        gates = self.gate_layer(self.h1(seqs.reshape(-1, self.emb_size))).reshape(-1, seq_len, 1).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs.reshape(-1, seq_len, self.cfg.hidden_size*2) * p_attn
        output = torch.sum(h, dim=-2)
        return output

