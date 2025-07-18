import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from scipy.sparse import csr_matrix, save_npz


def randint_choice(n, size=None, replace=True):
    return np.random.choice(n, size=size, replace=replace)

def UniformSample_original_python(dataset):
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_mashups, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.n_apis)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

class BPRTrainSampler(Dataset):
    def __init__(self, dataset):
        self.data = UniformSample_original_python(dataset)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        user, pos, neg = self.data[index]
        return user, pos, neg
    
class BPRTestSampler(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class Loader(Dataset):
    def __init__(self):
        # train or test
        self.n_mashups = 0
        self.n_apis = 0
        self.path = "./data/3mashups"
        train_file = self.path + "/train.txt"
        test_file = self.path + "/test.txt"
        co_aafile = self.path + "/api_co_category.txt"
        co_aafile1 = self.path + "/co_mashup_api_matrix.npy"
        co_mmfile = self.path + "/mashup_co_category.txt"
        co_mmfile1 = self.path + "/mashup_co_api.txt"
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        coApi1, coApi2, apiValue = [], [], []
        coMashup1, coMashup2, mashupValue = [], [], []
        coApi_Mashup1, coApi_Mashup2, coApi_MashupValue = [], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.n_apis = max(self.n_apis, max(items))
                    self.n_mashups = max(self.n_mashups, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except:
                        print(l[0])
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.n_apis = max(self.n_apis, max(items))
                    self.n_mashups = max(self.n_mashups, uid)
                    self.testDataSize += len(items)
        self.n_apis += 1
        self.n_mashups += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        with open(co_aafile) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ', 2)
                    coApi1.append(int(l[0]))
                    coApi2.append(int(l[1]))
                    apiValue.append(int(l[2]))

        self.coApi1 = np.array(coApi1)
        self.coApi2 = np.array(coApi2)
        self.apiValue = np.array(apiValue)

        coMashup_Api_Matrix = np.load(co_aafile1, allow_pickle=True)
        self.coMashupApi = csr_matrix(coMashup_Api_Matrix)

        with open(co_mmfile) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ', 2)
                    coMashup1.append(int(l[0])) 
                    coMashup2.append(int(l[1]))
                    mashupValue.append(int(l[2]))

        self.coMashup1 = np.array(coMashup1)
        self.coMashup2 = np.array(coMashup2)
        self.mashupValue = np.array(mashupValue)
        
        with open(co_mmfile1) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ', 2)
                    coApi_Mashup1.append(int(l[0]))
                    coApi_Mashup2.append(int(l[1]))
                    coApi_MashupValue.append(int(l[2]))
        self.coApi_Mashup1 = np.array(coApi_Mashup1)
        self.coApi_Mashup2 = np.array(coApi_Mashup2)
        self.coApi_MashupValue = np.array(coApi_MashupValue)

        self.Graph = None
        self.aaGraph = None
        self.aaGraph1 = None
        self.mmGraph = None
        self.mmGraph1 = None
        self.DiffGraph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_mashups / self.n_apis}")

        # (users,items), bipartite graph
        self.MashupApiNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_mashups, self.n_apis))
        
        self.coCate_ApiNet = csr_matrix((self.apiValue, (self.coApi1, self.coApi2)),
                                    shape=(self.n_apis, self.n_apis))
        
        self.coMashup_ApiNet = csr_matrix(coMashup_Api_Matrix)

        
        self.coCate_MashupNet = csr_matrix((self.mashupValue, (self.coMashup1, self.coMashup2)),
                                      shape=(self.n_mashups, self.n_mashups))
        
        self.coApi_MashupNet = csr_matrix((self.coApi_MashupValue, (self.coApi_Mashup1, self.coApi_Mashup2)),
                                            shape=(self.n_mashups, self.n_mashups))
        # self.coCate_ApiNet = csr_matrix((np.ones(len(self.coApi1)), (self.coApi1, self.coApi2)),shape=(self.n_apis, self.n_apis))
        # self.coCate_MashupNet = csr_matrix((np.ones(len(self.coMashup1)), (self.coMashup1, self.coMashup2)),shape=(self.n_mashups, self.n_mashups))
        self.users_D = np.array(self.MashupApiNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.MashupApiNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_mashups)))
        self.__testDict = self.__build_test()

    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.n_mashups + self.n_apis, self.n_mashups + self.n_apis), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.MashupApiNet.tolil()
        mmR = self.coCate_MashupNet.tolil()
        aaR = self.coCate_ApiNet.tolil()
        adj_mat[:self.n_mashups, self.n_mashups:] = R
        adj_mat[self.n_mashups:, :self.n_mashups] = R.T
        # adj_mat[:self.n_mashups, :self.n_mashups] = mmR
        # adj_mat[self.n_mashups:, self.n_mashups:] = aaR
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        # end = time()
        # print(f"costing {end-s}s, saved norm_mat...")
        # sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to("cuda")
        print("don't split the matrix")
        return self.Graph
    
    def getAugSparseGraph(self):
        keep_idx = randint_choice(len(self.trainUser), int(len(self.trainUser) * 0.9), replace=False)
        train_user = self.trainUser[keep_idx]
        train_item = self.trainItem[keep_idx]
        temp_adj = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                              shape=(self.n_mashups, self.n_apis))
        
        adj_mat = sp.dok_matrix((self.n_mashups + self.n_apis, self.n_mashups + self.n_apis), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = temp_adj.tolil()
        mmR = self.coCate_MashupNet.tolil()
        aaR = self.coCate_ApiNet.tolil()
        adj_mat[:self.n_mashups, self.n_mashups:] = R
        adj_mat[self.n_mashups:, :self.n_mashups] = R.T
        adj_mat[:self.n_mashups, :self.n_mashups] = mmR
        adj_mat[self.n_mashups:, self.n_mashups:] = aaR
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        aug_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        aug_graph = aug_graph.coalesce().to("cuda")
        return aug_graph
    
    def getDiffSparseGraph(self, aug_graph):
        adj_mat = sp.dok_matrix((self.n_mashups + self.n_apis, self.n_mashups + self.n_apis), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = aug_graph.tolil()
        mmR = self.coCate_MashupNet.tolil()
        aaR = self.coCate_ApiNet.tolil()
        adj_mat[:self.n_mashups, self.n_mashups:] = R
        adj_mat[self.n_mashups:, :self.n_mashups] = R.T
        # adj_mat[:self.n_mashups, :self.n_mashups] = mmR
        # adj_mat[self.n_mashups:, self.n_mashups:] = aaR
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        self.DiffGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.DiffGraph = self.DiffGraph.coalesce().to("cuda")
        return self.DiffGraph

    def getSparseGraph_aa(self):
        if self.aaGraph is None:
            # try:
            #     pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_aa.npz')
            #     print("successfully loaded aa graph...")
            #     norm_adj = pre_adj_mat
            # except:
            print("generating api homogenous adjacency matrix")
            adj_mat = sp.dok_matrix((self.n_apis, self.n_apis), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.coCate_ApiNet.tolil()
            adj_mat[:self.n_apis, :self.n_apis] = R
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # sp.save_npz(self.path + '/s_pre_adj_mat_aa.npz', norm_adj)
            self.aaGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.aaGraph = self.aaGraph.coalesce().to("cuda")

            adj_mat = sp.dok_matrix((self.n_apis, self.n_apis), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.coMashup_ApiNet.tolil()
            adj_mat[:self.n_apis, :self.n_apis] = R
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # sp.save_npz(self.path + '/s_pre_adj_mat_aa.npz', norm_adj)
            self.aaGraph1 = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.aaGraph1 = self.aaGraph1.coalesce().to("cuda")
        return self.aaGraph, self.aaGraph1
    
    def getSparseGraph_mm(self):
        if self.mmGraph is None:
            # try:
            #     pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_mm.npz')
            #     print("successfully loaded mm graph...")
            #     norm_adj = pre_adj_mat
            # except:
            print("generating mashup homogenous graph matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_mashups, self.n_mashups), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.coCate_MashupNet.tolil()
            adj_mat[:self.n_mashups, :self.n_mashups] = R
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # sp.save_npz(self.path + '/s_pre_adj_mat_mm.npz', norm_adj)      
            self.mmGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.mmGraph = self.mmGraph.coalesce().to("cuda")

            adj_mat = sp.dok_matrix((self.n_mashups, self.n_mashups), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.coApi_MashupNet.tolil()
            adj_mat[:self.n_mashups, :self.n_mashups] = R
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # sp.save_npz(self.path + '/s_pre_adj_mat_mm.npz', norm_adj)      
            self.mmGraph1 = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.mmGraph1 = self.mmGraph.coalesce().to("cuda")
        return self.mmGraph, self.mmGraph1          

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.MashupApiNet[user].nonzero()[1])
        return posItems
    

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)