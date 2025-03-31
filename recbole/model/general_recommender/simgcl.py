import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import BPRLoss, EmbLoss
# import pdb
# from torch_sparse import SparseTensor, fill_diag, matmul, mul
# from torch_sparse import sum as sparsesum
#
# from torch_scatter import scatter_add, scatter_softmax, scatter_sum, scatter_min, scatter_max
# from torch_geometric.utils import add_remaining_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

# def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=False, flow="source_to_target", dtype=None):
#
#     fill_value = 2. if improved else 1.
#
#     if isinstance(edge_index, SparseTensor):
#         assert flow in ["source_to_target"]
#         adj_t = edge_index
#         if not adj_t.has_value():
#             adj_t = adj_t.fill_value(1., dtype=dtype)
#         if add_self_loops:
#             adj_t = fill_diag(adj_t, fill_value)
#         deg = sparsesum(adj_t, dim=1)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
#         adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
#         return adj_t
#
#     else:
#         assert flow in ["source_to_target", "target_to_source"]
#         num_nodes = maybe_num_nodes(edge_index, num_nodes)
#
#         if edge_weight is None:
#             edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                      device=edge_index.device)
#
#         if add_self_loops:
#             edge_index, tmp_edge_weight = add_remaining_self_loops(
#                 edge_index, edge_weight, fill_value, num_nodes)
#             assert tmp_edge_weight is not None
#             edge_weight = tmp_edge_weight
#
#         row, col = edge_index[0], edge_index[1]
#         idx = col if flow == "source_to_target" else row
#         deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class SimGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.eps = config['eps']

        # new add
        self.cl_rate = float(config['lambda'])
        self.temp = float(config['temp'])
        self.pool = 'mean'
        self.decay = 1e-4
        self.cl_weight = float(config['cl_weight'])
        self.reg_weight = float(config['reg_weight'])
        self.reg = float(config["lambda"])

        self.n_layers = config['n_layers']
        self.require_pow = config["require_pow"]

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        # self.encoder_name = config['encoder']

        self.distance = config["distance"]
        # self.is_change_old_new = config["is_change_old_new"]
        self.old_new_dir = config["old_new_dir"]
        self.is_mean = config["is_mean"]
        self.alphaM = config["alphaM"]
        self.betaM = config["betaM"]
        self.device = config["device"]


        # define layers and loss
        # if self.encoder_name == 'MF':
        #     self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        # elif self.encoder_name == 'LightGCN':

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        # self.encoder = SimGCL_Encoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers,
        #                             self.device)

        # define layers and loss


        self.encoder = SimGCL_Encoder(config,self.n_users, self.n_items, self.embedding_size, self.eps, self.norm_adj,
                                      self.n_layers, self.device)
        # else:
        #     raise ValueError('Non-implemented Encoder.')

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
        # i = torch.LongTensor([row, col])
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs, user, item, neg_item):
        # user_gcn_emb: [batch_size, n_hops+1, channel] = torch.Size([1024, 4, 64])
        # pos_gcn_embs: [batch_size, n_hops+1, channel] = torch.Size([1024, 4, 64])
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]  = torch.Size([1024, 1, 4, 64]) 这里变成4维了，和其他两个tensor维度对不上了
        #
        # batch_size = user_gcn_emb.shape[0]
        #
        # u_e = self.pooling(user_gcn_emb)  # u_e.shape = torch.Size([1024, 256])
        # pos_e = self.pooling(pos_gcn_embs)
        # # neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, 1,
        # #                                                                                                -1)
        # neg_e = self.pooling(neg_gcn_embs)
        # # 这里把 neg_gcn_embs([2048, 1, 4, 64]) view成为([2048, 4, 64])，再 pooling ，再恢复为([2048, 1, 256])
        #
        # pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)  # pos_scores.shape = torch.Size([1024])
        # neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]
        #
        # mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))
        # # cul regularizer
        # # regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
        # #               + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
        # #               + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        # emb_loss = self.decay * regularize / batch_size
        #
        # return mf_loss + emb_loss, mf_loss, emb_loss
        # calculate BPR Loss
        pos_scores = torch.mul(user_gcn_emb, pos_gcn_embs).sum(dim=1)
        neg_scores = torch.mul(user_gcn_emb, neg_gcn_embs).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.encoder.user_embedding(user)
        pos_ego_embeddings = self.encoder.item_embedding(item)
        neg_ego_embeddings = self.encoder.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def forward(self, perturbed=False):
        # user_e, item_e = self.encoder(user, item)
        rec_user_emb, rec_item_emb ,rec_user_emb_all_layer,rec_item_emb_all_layer= self.encoder()
        # return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
        return rec_user_emb, rec_item_emb ,rec_user_emb_all_layer,rec_item_emb_all_layer

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        rec_user_emb, rec_item_emb, rec_user_emb_all_layer, rec_item_emb_all_layer = self.forward()

        # calculate bpr loss
        user_emb_all_layer, pos_item_emb_all_layer = rec_user_emb_all_layer[user], rec_item_emb_all_layer[item]
        neg_gcn_embs = rec_item_emb_all_layer[neg_item]

        # pdb.set_trace()
        # rec_loss = self.create_bpr_loss(user_emb_all_layer, pos_item_emb_all_layer, neg_gcn_embs, user, item, neg_item)
        rec_loss = self.create_bpr_loss(user_emb_all_layer, pos_item_emb_all_layer, neg_gcn_embs, user, item, neg_item)

        # neg_item_emb = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3]))
        # user_emb = self.pooling(user_emb_all_layer)
        # pos_item_emb = self.pooling(pos_item_emb_all_layer)

        cl_loss = self.cl_rate * self.cal_cl_loss([user, item]) * self.cl_weight
        # batch_loss = rec_loss + self.l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_embs) + cl_loss
        batch_loss = rec_loss + self.l2_reg_loss(self.reg, user_emb_all_layer, pos_item_emb_all_layer, neg_gcn_embs) + cl_loss
        return batch_loss

        # neg_id = interaction[self.]

        # user_e, item_e = self.forward(user, item)
        # user_e, item_e, user_e_all_layer, item_e_all_layer = self.encoder(user, item)
        # align = self.alignment(user_e, item_e)
        # uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        # return align + uniform


    # def train(self, mode: bool = True):
    #     r"""Override train method of base class.The subgraph is reconstructed each time it is called."""
    #     T = super().train(mode=mode)
    #     if mode:
    #         self.graph_construction()
    #     return T
    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.tensor(idx[0], dtype=torch.long, device=self.device).clone().detach())
        i_idx = torch.unique(torch.tensor(idx[1], dtype=torch.long, device=self.device).clone().detach())


        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1, _, _ = self.encoder(perturbed=True)
        user_view_2, item_view_2, _, _ = self.encoder(perturbed=True)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        _, _, user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     if self.encoder_name == 'LightGCN':
    #         if self.restore_user_e is None or self.restore_item_e is None:
    #             self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
    #         user_e = self.restore_user_e[user]
    #         all_item_e = self.restore_item_e
    #     else:
    #         user_e = self.encoder.user_embedding(user)
    #         all_item_e = self.encoder.item_embedding.weight
    #     score = torch.matmul(user_e, all_item_e.transpose(0, 1))
    #     return score.view(-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
       #  print(user)
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores


class SimGCL_Encoder(nn.Module):
    def __init__(self, config, user_num, item_num, emb_size, eps, norm_adj, n_layers=3, device = "cuda:0"):
        super(SimGCL_Encoder, self).__init__()
        # self.data = data
        self.n_users = user_num
        self.n_items = item_num
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        # self.norm_adj = data.norm_adj
        self.norm_adj = norm_adj
        self.device = device

        self.distance = config["distance"]
        # self.is_change_old_new = config["is_change_old_new"]
        self.old_new_dir = config["old_new_dir"]
        self.is_mean = config["is_mean"]
        self.alphaM = config["alphaM"]
        self.betaM = config["betaM"]

        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.emb_size
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.emb_size
        )
        # self.embedding_dict = self._init_model()
        # self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    # def _init_model(self):
    #     initializer = nn.init.xavier_uniform_
    #     embedding_dict = nn.ParameterDict({
    #         'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
    #         'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size))),
    #     })
    #     return embedding_dict

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, perturbed=False):
        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings = self.get_ego_embeddings()
        embs=[] # 作者说的，要跳过最原始的E_0

        all_embeddings = []
        ## 这段代码替换，以采用SAGCN框架
        # for k in range(self.n_layers):
        #     # ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        #     ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
        #     if perturbed:
        #         random_noise = torch.rand_like(ego_embeddings).to(self.device)
        #         ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
        #     all_embeddings.append(ego_embeddings)
        #     embs.append(ego_embeddings)
        # all_embeddings = torch.stack(all_embeddings, dim=1)
        # all_embeddings = torch.mean(all_embeddings, dim=1)

        for k in range(self.n_layers):
            # ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddingn = torch.sparse.mm(self.norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddingn).to(self.device)
                ego_embeddingn += torch.sign(ego_embeddingn) * F.normalize(random_noise, dim=-1) * self.eps
            score_old, score_new = self.GrowthScore(ego_embeddings, ego_embeddingn)
            ego_embeddings = torch.mul(score_old, ego_embeddings) + torch.mul(score_new, ego_embeddingn)
            embs.append(ego_embeddings)
        all_embeddings = ego_embeddings

        # user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        embs = torch.stack(embs,dim=1)
        # user_all_embeddings_all_layer, item_all_embeddings_all_layer = torch.split(embs, [self.data.user_num, self.data.item_num])
        user_all_embeddings_all_layer, item_all_embeddings_all_layer = torch.split(embs, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings,user_all_embeddings_all_layer,item_all_embeddings_all_layer

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
