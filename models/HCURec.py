import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.modules import NewsEncoder, PoEncoder, Attend
import numpy as np


class HCURec(nn.Module):
    def __init__(self, cfg):
        super(HCURec, self).__init__()

        self.news_encoder = NewsEncoder(cfg)
        self.cfg = cfg
        self.category_emb = nn.Embedding(cfg.cate_num, cfg.hidden_size)
        self.subcategory_emb = nn.Embedding(cfg.subcate_num, cfg.hidden_size)
        self.news_title_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_title_emb), requires_grad=False)
        self.news_entity_indexes = nn.Parameter(torch.LongTensor(self.cfg.news_entity_emb), requires_grad=False)

        self.cate_num_emb = nn.Embedding(cfg.cate_num, cfg.hidden_size)
        self.sub_num_emb = nn.Embedding(cfg.subcate_num, cfg.hidden_size)
        self.po_encoder = PoEncoder(cfg)
        self.sub_attention = Attend(cfg.hidden_size*2, cfg)
        self.cate_attention = Attend(cfg.hidden_size*2, cfg)

        self.sub_tran = nn.Linear(self.cfg.hidden_size, 1)
        self.cate_tran = nn.Linear(self.cfg.hidden_size, 1)
        self.sub_d = nn.Linear(self.cfg.hidden_size*2, self.cfg.hidden_size)
        self.cate_d = nn.Linear(self.cfg.hidden_size * 2, self.cfg.hidden_size)

        self.sub_h1 = nn.Sequential(
            nn.Linear(cfg.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        self.cate_h1 = nn.Sequential(
            nn.Linear(cfg.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        self.global_h1 = nn.Sequential(
            nn.Linear(cfg.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        self.sub_match_reduce_layer = nn.Linear(cfg.hidden_size, 100)
        self.cate_match_reduce_layer = nn.Linear(cfg.hidden_size, 100)
        self.global_match_reduce_layer = nn.Linear(cfg.hidden_size, 100)

        self.sub_num_tran = nn.Linear(self.cfg.hidden_size, 1)
        self.cate_num_tran = nn.Linear(self.cfg.hidden_size, 1)

    def forward(self, data, start_point, user_sub_num, user_cate_num, test_mode=False):
        news_num = self.cfg.neg_count + 1
        if test_mode:
            news_num = 1

        user_emb = data[:, news_num * 5:]
        user_news_id = user_emb[:, self.cfg.ucatgeory_number + self.cfg.ucatgeory_number * self.cfg.usubcate_number:].reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news)
        user_news_title = self.news_title_indexes[user_news_id]
        user_news_entity = self.news_entity_indexes[user_news_id]
        user_news = self.news_encoder(user_news_title.reshape(-1, self.cfg.max_title_len), user_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news, self.cfg.hidden_size)
        user_categories1 = user_emb[:, :self.cfg.ucatgeory_number]
        user_categories = self.category_emb(user_categories1)
        user_subcategories1 = user_emb[:, self.cfg.ucatgeory_number: self.cfg.ucatgeory_number + self.cfg.ucatgeory_number * self.cfg.usubcate_number].reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number)
        user_subcategories = self.subcategory_emb(user_subcategories1)
        news_subcategory_index = data[:, news_num * 2: news_num * 3]
        news_subcategory_index = start_point * self.cfg.ucatgeory_number * self.cfg.usubcate_number + news_subcategory_index
        news_category_index = data[:, news_num: news_num * 2]
        news_category_index = start_point * self.cfg.ucatgeory_number + news_category_index

        news_id = data[:, :news_num]
        target_news_title = self.news_title_indexes[news_id]
        target_news_entity = self.news_entity_indexes[news_id]
        target_news = self.news_encoder(target_news_title.reshape(-1, self.cfg.max_title_len), target_news_entity.reshape(-1, self.cfg.max_entity_len)).reshape(-1, news_num, self.cfg.hidden_size)

        for i in range(news_num):
            user = user_news.reshape(-1, self.cfg.hidden_size)
            candi = target_news[:, i:i+1].reshape(-1, self.cfg.hidden_size)
            user_sub_att = self.sub_h1(user).reshape(-1, self.cfg.usubcate_news)
            candi_sub_att = self.sub_h1(candi).repeat(1, self.cfg.ucatgeory_number*self.cfg.usubcate_number*self.cfg.usubcate_news)
            candi_sub_att = candi_sub_att.reshape(-1, self.cfg.usubcate_news, 1)
            cross_user_sub_vecs = self.sub_match_reduce_layer(user).reshape(-1, self.cfg.usubcate_news, 100)
            cross_candi_sub_vecs = self.sub_match_reduce_layer(candi).repeat(1, self.cfg.ucatgeory_number*self.cfg.usubcate_number*self.cfg.usubcate_news)
            cross_candi_sub_vecs = cross_candi_sub_vecs.reshape(-1, self.cfg.usubcate_news, 100)
            cross_candi_sub_vecs = cross_candi_sub_vecs.transpose(1, 2)
            cross_sub_att = torch.bmm(cross_user_sub_vecs, cross_candi_sub_vecs)
            cross_user_sub_att = F.softmax(cross_sub_att, dim=-1)
            cross_user_sub_att = torch.bmm(cross_user_sub_att, candi_sub_att)
            user_sub_att = user_sub_att + cross_user_sub_att.squeeze(-1)*0.01
            user_sub_att = F.softmax(user_sub_att, dim=-1)
            user_subcate_rep0 = torch.sum(user_sub_att.unsqueeze(-1) * user_news.reshape(-1, self.cfg.usubcate_news, self.cfg.hidden_size), dim=-2)
            user_subcate_po = self.po_encoder(user_news_entity.reshape(-1, self.cfg.max_entity_len), user_news_title.reshape(-1, self.cfg.max_title_len))
            user_subcate_po = user_subcate_po.reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news, self.cfg.hidden_size)
            user_subcategories = user_subcategories.reshape(len(user_subcategories), -1)
            user_cate_po_att = torch.cat((user_subcate_po, user_subcategories.repeat(1, self.cfg.usubcate_news).view(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news, self.cfg.hidden_size)), dim=-1)
            user_subcate_po = self.sub_attention(user_cate_po_att, self.cfg.usubcate_news)
            user_subcate_po = self.sub_d(user_subcate_po)
            user_subcate_rep0 = user_subcate_rep0 + user_subcate_po
            if i == 0:
                user_subcate_rep = user_subcate_rep0[news_subcategory_index[:, i:i+1]]
            else:
                user_subcate_rep = torch.cat((user_subcate_rep, user_subcate_rep0[news_subcategory_index[:, i:i+1]]), dim=1)

            user_sub_num_emb = self.sub_num_emb(user_sub_num)
            user_sub_num_score = self.sub_num_tran(user_sub_num_emb).reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number)

            user_subcate = user_subcate_rep0
            candi_subcate = candi.repeat(1, self.cfg.ucatgeory_number*self.cfg.usubcate_number).reshape(-1, self.cfg.hidden_size)
            user_cate_att = self.cate_h1(user_subcate).reshape(-1, self.cfg.usubcate_number) + user_sub_num_score.reshape(-1, self.cfg.usubcate_number)
            candi_cate_att = self.cate_h1(candi_subcate).reshape(-1, self.cfg.usubcate_number, 1)
            cross_user_cate_vecs = self.cate_match_reduce_layer(user_subcate).reshape(-1, self.cfg.usubcate_number, 100)
            cross_candi_cate_vecs = self.cate_match_reduce_layer(candi_subcate)
            cross_candi_cate_vecs = cross_candi_cate_vecs.reshape(-1, self.cfg.usubcate_number, 100)
            cross_candi_cate_vecs = cross_candi_cate_vecs.transpose(1, 2)
            cross_cate_att = torch.bmm(cross_user_cate_vecs, cross_candi_cate_vecs)
            cross_user_cate_att = F.softmax(cross_cate_att, dim=-1)
            cross_user_cate_att = torch.bmm(cross_user_cate_att, candi_cate_att)
            user_cate_att = user_cate_att + cross_user_cate_att.squeeze(-1) * 0.01
            user_cate_att = F.softmax(user_cate_att, dim=-1)
            user_cate_rep0 = torch.sum(user_cate_att.unsqueeze(-1) * user_subcate.reshape(-1, self.cfg.usubcate_number, self.cfg.hidden_size), dim=-2)

            user_categories = user_categories.reshape(-1, self.cfg.hidden_size)
            user_subcate_po = user_subcate_po.reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.hidden_size)
            user_cate_po_att = torch.cat((user_subcate_po, user_categories.repeat(1, self.cfg.usubcate_number).view(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.hidden_size)), dim=-1)
            user_cate_po = self.cate_attention(user_cate_po_att, self.cfg.usubcate_number)
            user_cate_po = self.cate_d(user_cate_po).reshape(-1, self.cfg.ucatgeory_number, self.cfg.hidden_size)
            user_cate_rep0 = user_cate_rep0 + user_cate_po.reshape(-1, self.cfg.hidden_size)
            if i == 0:
                user_cate_rep = user_cate_rep0[news_category_index[:, i:i+1]]
            else:
                user_cate_rep = torch.cat((user_cate_rep, user_cate_rep0[news_category_index[:, i:i+1]]), dim=1)

            user_cate_num_emb = self.cate_num_emb(user_cate_num)
            user_cate_num_score = self.cate_num_tran(user_cate_num_emb).reshape(-1, self.cfg.ucatgeory_number)

            user_cate = user_cate_rep0
            candi_cate = candi.repeat(1, self.cfg.ucatgeory_number).reshape(-1, self.cfg.hidden_size)
            user_global_att = self.global_h1(user_cate).reshape(-1, self.cfg.ucatgeory_number) + user_cate_num_score.reshape(-1, self.cfg.ucatgeory_number)
            candi_global_att = self.global_h1(candi_cate).reshape(-1, self.cfg.ucatgeory_number, 1)
            cross_user_global_vecs = self.global_match_reduce_layer(user_cate).reshape(-1, self.cfg.ucatgeory_number, 100)
            cross_candi_global_vecs = self.global_match_reduce_layer(candi_cate)
            cross_candi_global_vecs = cross_candi_global_vecs.reshape(-1, self.cfg.ucatgeory_number, 100)
            cross_candi_global_vecs = cross_candi_global_vecs.transpose(1, 2)
            cross_global_att = torch.bmm(cross_user_global_vecs, cross_candi_global_vecs)
            cross_user_global_att = F.softmax(cross_global_att, dim=-1)
            cross_user_global_att = torch.bmm(cross_user_global_att, candi_global_att)
            user_global_att = user_global_att + cross_user_global_att.squeeze(-1) * 0.01
            user_global_att = F.softmax(user_global_att, dim=-1)
            user_global_rep0 = torch.sum(user_global_att.unsqueeze(-1) * user_cate.reshape(-1, self.cfg.ucatgeory_number, self.cfg.hidden_size), dim=-2)
            if i == 0:
                user_global_rep = user_global_rep0.unsqueeze(1)
            else:
                user_global_rep = torch.cat((user_global_rep, user_global_rep0.unsqueeze(1)), dim=1)

        return user_subcate_rep, user_cate_rep, user_global_rep, target_news


class create_model(nn.Module):
    def __init__(self, cfg):
        super(create_model, self).__init__()

        self.news_encoder = NewsEncoder(cfg)
        self.user_encoder = HCURec(cfg)
        self.cfg = cfg

    def forward(self, data, start_point, device, test_mode=False):
        news_num = self.cfg.neg_count + 1
        if test_mode:
            news_num = 1

        user_emb = data[:, news_num * 5:]
        user_news_id = user_emb[:, self.cfg.ucatgeory_number + self.cfg.ucatgeory_number * self.cfg.usubcate_number:].reshape(-1, self.cfg.ucatgeory_number, self.cfg.usubcate_number, self.cfg.usubcate_news)
        news_coefficient_category = data[:, news_num * 3: news_num * 4]
        news_coefficient_subcategory = data[:, news_num * 4: news_num * 5]
        news_subcategory_index = data[:, news_num * 2: news_num * 3]
        news_subcategory_index = start_point * self.cfg.ucatgeory_number * self.cfg.usubcate_number + news_subcategory_index
        news_category_index = data[:, news_num: news_num * 2]
        news_category_index = start_point * self.cfg.ucatgeory_number + news_category_index

        if test_mode:
            train_mask = torch.ones(len(user_news_id), 1).to(device)
        else:
            train_mask = torch.rand(len(user_news_id), 1) > self.cfg.mask_prob
            train_mask = train_mask.long().to(device)

        user_sub_num = self.cfg.usubcate_news - (user_news_id == 1).sum(axis=-1)
        rw_subcate = user_sub_num / (user_sub_num.reshape(len(user_news_id), -1).sum(axis=-1).reshape(-1, 1).repeat(1, self.cfg.ucatgeory_number).view(-1, self.cfg.ucatgeory_number, 1) + 10 ** (-8))
        rw_subcate = rw_subcate.reshape(-1, 1)[news_subcategory_index].reshape(-1, news_num)
        rw_subcate = rw_subcate * train_mask

        user_cate_num = user_sub_num.sum(axis=-1)
        rw_cate = user_cate_num / (user_sub_num.reshape(len(user_news_id), -1).sum(axis=-1).reshape(-1, 1) + 10 ** (-8))
        rw_cate = rw_cate.reshape(-1, 1)[news_category_index].reshape(-1, news_num)

        user_subcate_rep, user_cate_rep, user_global_rep, target_news = self.user_encoder(data, start_point, user_sub_num, user_cate_num, test_mode)

        subcate_score = torch.sum(user_subcate_rep * target_news, dim=-1)
        subcate_score = subcate_score * rw_subcate * self.cfg.lambda_s * news_coefficient_subcategory

        cate_score = torch.sum(user_cate_rep * target_news, dim=-1)
        cate_score = cate_score * rw_cate * self.cfg.lambda_t * news_coefficient_category

        user_score = torch.sum(user_global_rep * target_news, dim=-1) * self.cfg.lambda_g

        final_score = user_score + cate_score + subcate_score

        return final_score