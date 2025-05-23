# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_scatter import scatter_add, scatter_softmax, scatter_sum, scatter_min, scatter_max
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tqdm import tqdm


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class SAGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SAGCN, self).__init__(config, dataset)

        # load datasets info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        # load the parametric of mine
        self.distance = config["distance"]
        # self.is_change_old_new = config["is_change_old_new"]
        self.old_new_dir = config["old_new_dir"]
        self.is_mean = config["is_mean"]
        self.alphaM = config["alphaM"]
        self.betaM = config["betaM"]


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        # A._update(data_dict)

        for (row, col), value in data_dict.items():
            A[row, col] = value


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
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        
        SparseL = SparseTensor(row=i[0], col=i[1], value=data, sparse_sizes=(self.n_users+self.n_items, self.n_users+self.n_items)) # torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        # SparseL = gcn_norm(SparseL, num_nodes=self.n_users+self.n_items)
        
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def GrowthScore(self, old_embedding, new_embedding):
        # 欧氏距离 相对距离
        if self.distance == "XiangDuiOuSi":
            XL = old_embedding.shape[0]
            osdist = torch.nn.PairwiseDistance(p=2)
            os_score = osdist(old_embedding, new_embedding)*self.betaM
            if self.old_new_dir == "new":
                d_old = torch.ones(XL).to(self.device)
                d_new = self.alphaM * torch.log(1 + os_score)
            elif self.old_new_dir == "old":
                d_new = torch.ones(XL).to(self.device)
                d_old = self.alphaM * torch.log(1 + os_score)
            else:
                raise "there have a error!"
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

        # 欧式距离 绝对距离
        elif self.distance == "JueDuiOuSi":
            ZL = old_embedding.shape[1]
            zero = torch.zeros(ZL).to(self.device)
            pdist = torch.nn.PairwiseDistance(p=2)
            d_old = pdist(old_embedding, zero)+ 1e-7
            d_new = pdist(new_embedding, zero)+ 1e-7
            # d_old = self.alphaM * torch.log(1 + d_old)
            d_new = self.alphaM * torch.log(1 + d_new)
            d_all = d_old + d_new
            score_old = d_old / d_all
            score_new = d_new / d_all

        # 余弦距离
        elif self.distance == "YuXian":
            XL = old_embedding.shape[0]
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_score = self.betaM*(1.0001 - torch.abs(cos(old_embedding, new_embedding)))
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
            os_score = self.betaM*torch.sum(os_score, dim=1)
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


        # 余弦相似度
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # cos_score = cos(old_embedding, new_embedding)
        # d_new = 2*(cos_score)*d_new
        # d_new = (1- cos_score) * d_new
        # d_new = 2*d_new

        # 添加参数设置部分
        # d_all = d_old + d_new
        # d_all = torch.exp(1 + d_old) + torch.exp(1 + d_new)
        # d_all = torch.exp(d_old) + torch.exp(d_new)
        #
        # d_old = torch.log(1 + d_old)

        # 这个是之前的，用来调参的
        # d_new = 2.5 * torch.log(1 + d_new)
        #

        # d_old = torch.exp(d_old)
        # d_new = 1 * torch.exp(d_new)


    def forward(self):
        all_embeddings = self.get_ego_embeddings()

        if self.is_mean:
            embeddings_list = [all_embeddings]

            for layer_idx in range(self.n_layers):
                all_embeddingn = matmul(self.norm_adj_matrix, all_embeddings)
                score_old, score_new = self.GrowthScore(all_embeddings, all_embeddingn)
                all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
                # 下面这里是没有交换（新embedding * 新score）
                # if self.is_change_old_new:
                #     all_embeddings = torch.mul(score_old, all_embeddingn) + torch.mul(score_new, all_embeddings)
                # else:
                #     all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
                embeddings_list.append(all_embeddings)

            lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
            lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        else:
        # embeddings_list = [all_embeddings]

            for layer_idx in range(self.n_layers):
                all_embeddingn = matmul(self.norm_adj_matrix, all_embeddings)
                score_old, score_new = self.GrowthScore(all_embeddings, all_embeddingn)
                all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
                # 下面这里是没有交换（新embedding * 新score）
                # if self.is_change_old_new:
                #     all_embeddings = torch.mul(score_old, all_embeddingn) + torch.mul(score_new, all_embeddings)
                # else:
                #     all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)

                # all_embeddings = torch.mul(score_old, all_embeddings) + torch.mul(score_new, all_embeddingn)
                # embeddings_list.append(all_embeddings)

            # lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
            # lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

            # 尝试去掉上面这两个mean操作，然后直接使用训练出来的结果
            lightgcn_all_embeddings = all_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
       #  print(user)
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
