# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class SADirectAU(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SADirectAU, self).__init__(config, dataset)

        # load the parametric of mine
        self.distance = config["distance"]
        self.is_change_old_new = config["is_change_old_new"]
        self.is_mean = config["is_mean"]
        self.alphaM = config["alphaM"]

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.encoder_name = config['encoder']

        # define layers and loss
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        elif self.encoder_name == 'LightGCN':
            self.n_layers = config['n_layers']
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.distance, self.is_change_old_new, self.is_mean, self.alphaM, self.n_layers, self.device)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)




    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align + uniform

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.encoder.user_embedding(user)
        item_e = self.encoder.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    # def save_params(self):
    #     user_embeddings, item_embeddings = self.encoder.get_all_embeddings()
    #     np.save('user-DirectAU.npy', user_embeddings.data.cpu().numpy())
    #     np.save('item-DirectAU.npy', item_embeddings.data.cpu().numpy())

    # def check(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #     user_e, item_e = self.forward(user, item)
    #
    #     user_e = user_e.detach()
    #     item_e = item_e.detach()
    #
    #     alignment_loss = self.alignment(user_e, item_e)
    #     uniform_loss = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
    #
    #     return alignment_loss, uniform_loss


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size, norm_adj, distance, is_change_old_new, is_mean, alphaM, n_layers=3, device = "cuda:0"):
        super(LGCNEncoder, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.device = device

        self.user_embedding = torch.nn.Embedding(user_num, emb_size)
        self.item_embedding = torch.nn.Embedding(item_num, emb_size)

        # load the parametric of mine
        self.distance = distance
        self.is_change_old_new = is_change_old_new
        self.is_mean = is_mean
        self.alphaM = alphaM

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()

        if self.is_mean:

            embeddings_list = [all_embeddings]

            for layer_idx in range(self.n_layers):
                # all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
                # embeddings_list.append(all_embeddings)
                # embeddings_list.append(all_embeddings)

                all_embeddingn = torch.sparse.mm(self.norm_adj, all_embeddings)
                score_old, score_new = self.GrowthScore(all_embeddings, all_embeddingn)
                if self.is_change_old_new:
                    all_embeddings = torch.mul(score_old, all_embeddingn) + torch.mul(score_new, all_embeddings)
                else:
                    all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
                # 感觉DirectAU需要这个mean会更好
                embeddings_list.append(all_embeddings)


            lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
            lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        else:
            for layer_idx in range(self.n_layers):
                # all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
                # embeddings_list.append(all_embeddings)
                # embeddings_list.append(all_embeddings)

                all_embeddingn = torch.sparse.mm(self.norm_adj, all_embeddings)
                score_old, score_new = self.GrowthScore(all_embeddings, all_embeddingn)
                if self.is_change_old_new:
                    all_embeddings = torch.mul(score_old, all_embeddingn) + torch.mul(score_new, all_embeddings)
                else:
                    all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
            lightgcn_all_embeddings = all_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def GrowthScore(self, old_embedding, new_embedding):
        # 相对欧式距离
        # 欧氏距离 相对距离
        if self.distance == "XiangDuiOuSi":
            XL = old_embedding.shape[0]
            osdist = torch.nn.PairwiseDistance(p=2)
            os_score = osdist(old_embedding, new_embedding)
            d_old = torch.ones(XL).to(self.device)
            d_new = self.alphaM * torch.log(1 + os_score)
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

        # 欧式距离 绝对距离
        elif self.distance == "JueDuiOuSi":
            ZL = old_embedding.shape[1]
            zero = torch.zeros(ZL).to(self.device)
            pdist = torch.nn.PairwiseDistance(p=2)
            d_old = pdist(old_embedding, zero)
            d_new = pdist(new_embedding, zero)
            # d_old = self.alphaM * torch.log(1 + d_old)
            # d_new = self.alphaM * torch.log(1 + d_new)
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

        elif self.distance == "YuXian":
            XL = old_embedding.shape[0]
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_score = cos(old_embedding, new_embedding)
            d_old = torch.ones(XL).to(self.device)
            d_new = self.alphaM * torch.log(1 + cos_score)
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

            # KL散度
        elif self.distance == "KLSanDu":
            XL = old_embedding.shape[0]
            # osdist = torch.nn.PairwiseDistance(p=2)
            log_old_embedding = F.log_softmax(old_embedding)
            softmax_new_embedding = F.softmax(new_embedding, dim=1)
            os_score = F.kl_div(log_old_embedding, softmax_new_embedding, reduction='none')
            os_score = 20 * torch.sum(os_score, dim=1)
            d_old = torch.ones(XL).to(self.device)
            d_new = self.alphaM * torch.log(1 + torch.abs(os_score))
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

        else:
            raise "there is no this distance"

        return torch.unsqueeze(score_old, 1), torch.unsqueeze(score_new, 1)

        # if self.is_change_old_new:
        #     return torch.unsqueeze(score_new, 1), torch.unsqueeze(score_old, 1)
        # else:
        #     return torch.unsqueeze(score_old, 1), torch.unsqueeze(score_new, 1)

        # 欧氏距离
        # ZL = old_embedding.shape[1]
        # zero = torch.zeros(ZL).to(self.device)
        # pdist = nn.PairwiseDistance(p=2)
        # d_old = pdist(old_embedding, zero)
        # d_new = pdist(new_embedding, zero)

        # 余弦相似度
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # cos_score = cos(old_embedding, new_embedding)
        # d_new = 2*(cos_score)*d_new
        # d_new = (1- cos_score) * d_new
        # d_new = 2*d_new

        # d_old = d_old
        # d_new = 1.2 * torch.log(1+d_new)
        # d_all = d_old + 2*torch.log(1+d_new)
        # d_all = d_old + d_new
        # score_old = d_old / d_all
        # score_new = d_new / d_all
        # return torch.unsqueeze(score_old, 1), torch.unsqueeze(score_new, 1)

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed
