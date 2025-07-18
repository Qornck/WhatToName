import torch
from torch import nn
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from copy import deepcopy
import torch_sparse
import scipy.sparse as sp

def inner_product(a, b):
    return torch.sum(a * b, dim=-1)

class LightGCN(nn.Module):
    def __init__(self, 
                args,
                dataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_mashups  = self.dataset.n_mashups
        self.num_apis  = self.dataset.n_apis
        self.latent_dim = self.args.recdim
        self.n_layers = self.args.layer
        self.embedding_mashup = torch.nn.Embedding(
            num_embeddings=self.num_mashups, embedding_dim=self.latent_dim)
        self.embedding_api = torch.nn.Embedding(
            num_embeddings=self.num_apis, embedding_dim=self.latent_dim)
        self.embedding_mashup_mashup = torch.nn.Embedding(
            num_embeddings=self.num_mashups, embedding_dim=self.latent_dim)
        self.embedding_api_api = torch.nn.Embedding(
            num_embeddings=self.num_apis, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_mashup.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_api.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_mashup_mashup.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_api_api.weight, gain=1)
        print('use xavier initilizer')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.aaGraph, self.aaGraph1 = self.dataset.getSparseGraph_aa()
        self.mmGraph, self.mmGraph1 = self.dataset.getSparseGraph_mm()
        # self.augGraph1 = self.dataset.getAugSparseGraph()
        # self.augGraph2 = self.dataset.getAugSparseGraph()
        self.diffGraph1 = None
        self.diffGraph2 = None
        x = torch.cat([self.embedding_mashup.weight, self.embedding_api.weight])
        self.DenoisingNet = DenoisingNet(x.cuda(), self.args)
        self.opt = torch.optim.Adam(self.DenoisingNet.parameters(), lr=self.args.lr, weight_decay=0)

    def forward_gcn(self, graph):
        all_emb = torch.cat([self.embedding_mashup.weight, self.embedding_api.weight])
        all_embs = [all_emb]
        mashup_mashup_emb = self.embedding_mashup_mashup.weight
        mashup_mashup_embs = [mashup_mashup_emb]
        api_api_embs = [self.embedding_api_api.weight]
        api_api_emb = self.embedding_api_api.weight
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            mashup_emb = torch.sparse.mm(self.mmGraph, mashup_mashup_emb)
            mashup1_emb = torch.sparse.mm(self.mmGraph1, mashup_mashup_emb)
            api_emb = torch.sparse.mm(self.aaGraph, api_api_emb)
            api1_emb = torch.sparse.mm(self.aaGraph1, api_api_emb)
            mashup_emb_o, api_emb_o = torch.split(all_emb, [self.num_mashups, self.num_apis])
            weighted_mashup_emb = (mashup_emb + mashup1_emb + mashup_emb_o) / 3
            weighted_api_emb = (api_emb + api1_emb + api_emb_o) / 3
            all_emb = torch.cat([weighted_mashup_emb, weighted_api_emb])          
            all_embs.append(all_emb)
            mashup_mashup_embs.append(mashup_emb)
            api_api_embs.append(api_emb)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        return all_embs
    
    def denoise_forward_gcn(self, graph):
        all_emb = torch.cat([self.embedding_mashup.weight, self.embedding_api.weight])
        all_embs = [all_emb]
        mashup_mashup_emb = self.embedding_mashup_mashup.weight
        mashup_mashup_embs = [mashup_mashup_emb]
        api_api_embs = [self.embedding_api_api.weight]
        api_api_emb = self.embedding_api_api.weight
        for layer in range(self.n_layers):
            with torch.no_grad():
                graph = self.DenoisingNet.generate(all_embs[-1], layer)
            all_emb = torch.sparse.mm(graph, all_emb)
            mashup_emb = torch.sparse.mm(self.mmGraph, mashup_mashup_emb)
            api_emb = torch.sparse.mm(self.aaGraph, api_api_emb)
            mashup_emb_o, api_emb_o = torch.split(all_emb, [self.num_mashups, self.num_apis])
            weighted_mashup_emb = (mashup_emb + mashup_emb_o) / 2
            weighted_api_emb = (api_emb + api_emb_o) / 2
            all_emb = torch.cat([weighted_mashup_emb, weighted_api_emb])          
            all_embs.append(all_emb)
            mashup_mashup_embs.append(mashup_emb)
            api_api_embs.append(api_emb)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        mashup_embs = torch.stack(mashup_mashup_embs, dim=1).mean(dim=1)
        api_embs = torch.stack(api_api_embs, dim=1).mean(dim=1)
        return all_embs, mashup_embs, api_embs
    
    def forward_gcn_(self, graph):
        all_emb = torch.cat([self.embedding_mashup.weight, self.embedding_api.weight])
        all_embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            isNan = torch.isnan(all_emb).any()
            all_embs.append(all_emb)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        return all_embs

    def getUsersRating(self, mashups):
        all_embs = self.forward_gcn(self.Graph)
        mashup_embs, api_embs = torch.split(all_embs, [self.num_mashups, self.num_apis])
        mashup_embs = F.embedding(mashups, mashup_embs)
        ratings = torch.matmul(mashup_embs, api_embs.T)
        return ratings
    
    def ssl_loss(self, data1, data2, index):
        index=torch.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        pos_score  = torch.sum(torch.mul(embeddings1, embeddings2), dim = 1)
        all_score  = torch.mm(embeddings1, embeddings2.T)
        pos_score  = torch.exp(pos_score / self.args.ssl_temp)
        all_score  = torch.sum(torch.exp(all_score / self.args.ssl_temp), dim = 1)
        ssl_loss  = (-torch.sum(torch.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss
    
    def forward(self, mashups, pos_apis, neg_apis, diffGraph, diffGraph1):
        all_embs = self.forward_gcn(self.Graph)
        diffGraph = self.dataset.getDiffSparseGraph(diffGraph)
        all_embs1 = self.forward_gcn(diffGraph)
        diffGraph1 = self.dataset.getDiffSparseGraph(diffGraph1)
        all_embs2 = self.forward_gcn(diffGraph1)
        # self.DenoisingNet.set_fea_adj(self.num_mashups + self.num_apis, diffGraph1)
        # with torch.no_grad():
        #     diffGraph1 = self.DenoisingNet.generate(all_embs, layer=0)
        # all_embs2,_,_ = self.denoise_forward_gcn(diffGraph1)

        # self.opt.zero_grad()
        # denoise_loss = self.DenoisingNet(mashups, pos_apis, neg_apis, self.args.ssl_temp, self.num_mashups, self.num_apis)
        # denoise_loss.backward()
        # self.opt.step()

        mashup_embs, api_embs = torch.split(all_embs, [self.num_mashups, self.num_apis])
        mashup_embs1, api_embs1 = torch.split(all_embs1, [self.num_mashups, self.num_apis])
        mashup_embs2, api_embs2 = torch.split(all_embs2, [self.num_mashups, self.num_apis])

        mashup_embeddings = F.embedding(mashups, mashup_embs)
        api_embeddings = F.embedding(pos_apis, api_embs)
        neg_embeddings = F.embedding(neg_apis, api_embs)

        mashup_embeddings1 = F.embedding(mashups, mashup_embs1)
        api_embeddings1 = F.embedding(pos_apis, api_embs1)
        neg_embeddings1 = F.embedding(neg_apis, api_embs1)

        mashup_embeddings2 = F.embedding(mashups, mashup_embs2)
        api_embeddings2 = F.embedding(pos_apis, api_embs2)
        neg_embeddings2 = F.embedding(neg_apis, api_embs2)

        sup_pos_ratings = inner_product(mashup_embeddings, api_embeddings)
        sup_neg_ratings = inner_product(mashup_embeddings, neg_embeddings)
        loss = torch.mean(torch.nn.functional.softplus(sup_neg_ratings - sup_pos_ratings))

        sup_pos_ratings1 = inner_product(mashup_embeddings1, api_embeddings1)
        sup_neg_ratings1 = inner_product(mashup_embeddings1, neg_embeddings1)
        loss1 = torch.mean(torch.nn.functional.softplus(sup_neg_ratings1 - sup_pos_ratings1))

        sup_pos_ratings2 = inner_product(mashup_embeddings2, api_embeddings2)
        sup_neg_ratings2 = inner_product(mashup_embeddings2, neg_embeddings2)
        loss2 = torch.mean(torch.nn.functional.softplus(sup_neg_ratings2 - sup_pos_ratings2))

        loss = loss + loss1 + loss2

        # bpr loss
        reg_loss = (1/2)*(mashup_embeddings.norm(2).pow(2) + 
                         api_embeddings.norm(2).pow(2)  +
                         neg_embeddings.norm(2).pow(2))/float(len(mashups))
        
        reg_loss1 = (1/2)*(mashup_embeddings1.norm(2).pow(2) +
                            api_embeddings1.norm(2).pow(2)  +
                            neg_embeddings1.norm(2).pow(2))/float(len(mashups))
        
        reg_loss2 = (1/2)*(mashup_embeddings2.norm(2).pow(2) +
                            api_embeddings2.norm(2).pow(2)  +
                            neg_embeddings2.norm(2).pow(2))/float(len(mashups))
        
        reg_loss = reg_loss + reg_loss1 + reg_loss2
        
        mashup_embs = F.normalize(mashup_embs, p = 2, dim=1)
        api_embs = F.normalize(api_embs, p = 2, dim=1)
        mashup_embs1 = F.normalize(mashup_embs1, p = 2, dim=1)
        api_embs1 = F.normalize(api_embs1, p = 2, dim=1)
        mashup_embs2 = F.normalize(mashup_embs2, p = 2, dim=1)
        api_embs2 = F.normalize(api_embs2, p = 2, dim=1)
        
        ssl_mashups = self.ssl_loss(mashup_embs1, mashup_embs2, mashups)
        ssl_apis = self.ssl_loss(api_embs1, api_embs2, pos_apis)

        return loss, reg_loss, ssl_mashups, ssl_apis

class DenoisingNet(nn.Module):
    def __init__(self, features, args):
        super(DenoisingNet, self).__init__()

        self.features = features
        self.args = args

        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        hidden = self.args.recdim

        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.attentions_0 = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.attentions_1 = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.attentions_2 = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )


    def get_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1
        if layer == 2:
            nb_layer = self.nblayers_2
            selflayer = self.selflayers_2

        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)
        if layer == 2:
            weight10 = self.attentions_2(input10)
        
        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = self.args.gamma
        zeta = self.args.zeta

        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + torch.tensor(np_random)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta-gamma) +gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def generate(self, x, layer=0):
        f1_features = x[self.row, :]
        f2_features = x[self.col, :]

        weight = self.get_attention(f1_features, f2_features, layer)

        mask = self.hard_concrete_sample(weight, training=False)

        mask = torch.squeeze(mask)
        adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

        ind = deepcopy(adj._indices())
        row = ind[0, :]
        col = ind[1, :]

        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = torch.mul(adj._values(), row_inv_sqrt)
        values = torch.mul(values, col_inv_sqrt)

        support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

        return support
        
    def l0_norm(self, log_alpha, beta):
        gamma = self.args.gamma
        zeta = self.args.zeta
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma/zeta))

        return torch.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        self.node_size = nodes
        self.adj_mat = adj

        ind = deepcopy(adj._indices())

        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.maskes = []

        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]

        for layer in range(self.args.layer):
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]

            weight = self.get_attention(f1_features, f2_features, layer=layer)
            mask = self.hard_concrete_sample(weight, temperature, training)

            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = torch.squeeze(mask)

            adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
            ind = deepcopy(adj._indices())
            row = ind[0, :]
            col = ind[1, :]

            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()

            nextx = torch.sparse.mm(support, x)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)
    
    def lossl0(self, temperature):
        l0_loss = torch.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def forward(self, users, items, neg_items, temperature, n_mashups, n_apis):
        x = self.call(temperature, True)
        x_user, x_item = torch.split(x, [n_mashups, n_apis], dim=0)
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        sup_pos_ratings = inner_product(ancEmbeds, posEmbeds)
        sup_neg_ratings = inner_product(ancEmbeds, negEmbeds)
        scoreDiff = sup_pos_ratings - sup_neg_ratings
        bprLoss = - (scoreDiff).sigmoid().log().sum() / self.args.bpr_batch
        regLoss = (1/2)*(ancEmbeds.norm(2).pow(2) + 
                        posEmbeds.norm(2).pow(2)  +
                        negEmbeds.norm(2).pow(2))/float(len(users))

        lossl0 = self.lossl0(temperature) * self.args.lambda0
        return bprLoss + regLoss + lossl0

class GraphMAE(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(GraphMAE, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        self.encoder = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self):
        encoded_users = self.encoder(self.user_emb.weight)  # [num_users, dim]
        encoded_items = self.encoder(self.item_emb.weight)  # [num_items, dim]
        return encoded_users, encoded_items

    def random_edge_dropout(self, adj, dropout_ratio=0.3):
        adj = adj.coalesce()
        values = adj._values()
        indices = adj._indices()

        num_edges = values.size(0)
        mask = torch.rand(num_edges, device=values.device) > dropout_ratio
        kept_indices = indices[:, mask]
        kept_values = values[mask]

        return torch.sparse.FloatTensor(kept_indices, kept_values, adj.shape), indices[:, ~mask]

    def reconstruction_loss(self, z_user, z_item, masked_edges):
        u_idx, i_idx = masked_edges
        scores = (z_user[u_idx] * z_item[i_idx]).sum(dim=1)
        labels = torch.ones_like(scores)
        return F.binary_cross_entropy_with_logits(scores, labels)

    @torch.no_grad()
    def generate_denoised_adj(self, z_user, z_item, threshold=0.5):
        score_matrix = torch.sigmoid(torch.matmul(z_user, z_item.T))  # [n_user, n_item]
        keep_mask = score_matrix > threshold
        row, col = keep_mask.nonzero(as_tuple=True)
        values = score_matrix[row, col]
        indices = torch.stack([row, col])
        shape = (z_user.size(0), z_item.size(0))
        return torch.sparse.FloatTensor(indices, values, torch.Size(shape))