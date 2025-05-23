import torch
from torch import nn
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

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
        self.aaGraph = self.dataset.getSparseGraph_aa()
        self.mmGraph = self.dataset.getSparseGraph_mm()
        self.augGraph1 = self.dataset.getAugSparseGraph()
        self.augGraph2 = self.dataset.getAugSparseGraph()

    def forward_gcn(self, graph):
        all_emb = torch.cat([self.embedding_mashup.weight, self.embedding_api.weight])
        all_embs = [all_emb]
        mashup_mashup_emb = self.embedding_mashup_mashup.weight
        api_api_emb = self.embedding_api_api.weight
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            # mashupmashup_embs = torch.sparse.mm(self.mmGraph, mashup_mashup_emb)
            # apiapi_embs = torch.sparse.mm(self.aaGraph, api_api_emb)
            # mashup_embs, api_embs = torch.split(all_emb, [self.num_mashups, self.num_apis])
            # mashup_embs = (mashup_embs + mashupmashup_embs) / 2.0
            # api_embs = (api_embs + apiapi_embs) / 2.0
            # all_emb = torch.cat([mashup_embs, api_embs])
            all_embs.append(all_emb)
        all_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        return all_emb

    def getUsersRating(self, mashups):
        all_embs = self.forward_gcn(self.Graph)
        mashup_embs, api_embs = torch.split(all_embs, [self.num_mashups, self.num_apis])
        mashup_embs = F.embedding(mashups, mashup_embs)
        ratings = torch.matmul(mashup_embs, api_embs.T)
        return ratings
    
    def ssl_loss(self, data1, data2,   index):
        index=torch.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = torch.exp(pos_score / self.args.ssl_temp)
        all_score  = torch.sum(torch.exp(all_score / self.args.ssl_temp), dim = 1)
        ssl_loss  = (-torch.sum(torch.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss
    
    def forward(self, mashups, pos_apis, neg_apis, aug_graph1, aug_graph2, recompute):
        all_embs = self.forward_gcn(self.Graph)
        diff_graph1 = self.dataset.getDiffSparseGraph(aug_graph1, recompute)
        diff_graph2 = self.dataset.getDiffSparseGraph(aug_graph2, recompute)
        all_aug_embs1 = self.forward_gcn(diff_graph1)
        all_aug_embs2 = self.forward_gcn(diff_graph2)

        mashup_embs, api_embs = torch.split(all_embs, [self.num_mashups, self.num_apis])
        mashup_embs1, api_embs1 = torch.split(all_aug_embs1, [self.num_mashups, self.num_apis])
        mashup_embs2, api_embs2 = torch.split(all_aug_embs2, [self.num_mashups, self.num_apis])

        mashup_embeddings = F.embedding(mashups, mashup_embs)
        api_embeddings = F.embedding(pos_apis, api_embs)
        neg_embeddings = F.embedding(neg_apis, api_embs)

        sup_pos_ratings = inner_product(mashup_embeddings, api_embeddings)
        sup_neg_ratings = inner_product(mashup_embeddings, neg_embeddings)
        loss = torch.mean(torch.nn.functional.softplus(sup_neg_ratings - sup_pos_ratings))

        # bpr loss
        reg_loss = (1/2)*(mashup_embeddings.norm(2).pow(2) + 
                         api_embeddings.norm(2).pow(2)  +
                         neg_embeddings.norm(2).pow(2))/float(len(mashups))
        
        mashup_embs = F.normalize(mashup_embs, p = 2, dim=1)
        api_embs = F.normalize(api_embs, p = 2, dim=1)
        mashup_embs1 = F.normalize(mashup_embs1, p = 2, dim=1)
        api_embs1 = F.normalize(api_embs1, p = 2, dim=1)
        mashup_embs2 = F.normalize(mashup_embs2, p = 2, dim=1)
        api_embs2 = F.normalize(api_embs2, p = 2, dim=1)
        
        ssl_mashups = self.ssl_loss(mashup_embs1, mashup_embs2, mashups)
        ssl_apis = self.ssl_loss(api_embs1, api_embs2, pos_apis)

        return loss, reg_loss, ssl_mashups, ssl_apis