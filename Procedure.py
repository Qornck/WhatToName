import numpy as np
import torch
import utils
from dataloader import Loader, BPRTrainSampler
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from parse import parse_args
import torch.utils.data as data
from gaussian_diffusion import GaussianDiffusion as gd

args = parse_args()

def BPR_train_original(dataset, recommend_model, loss_class, aug_graph1, aug_graph2, epoch, neg_k=1):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    Sampler = BPRTrainSampler(dataset)
    dataloader = data.DataLoader(Sampler, batch_size=args.bpr_batch, shuffle=True)
    aver_loss = 0.
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(dataloader):
        batch_users = batch_users.to("cuda")
        batch_pos = batch_pos.to("cuda")
        batch_neg = batch_neg.to("cuda")
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, aug_graph1, aug_graph2, batch_i == 0)
        aver_loss += cri
    aver_loss = aver_loss / len(dataloader)
    return f"loss{aver_loss:.3f}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    topks = eval(args.topks)
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch):
    u_batch_size = args.testbatch
    testDict = dataset.testDict
    Recmodel = Recmodel.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        testloader = data.DataLoader(users, batch_size=u_batch_size, shuffle=False)
        for batch_users in testloader:
            batch_users = batch_users.to("cuda")
            allPos = dataset.getUserPosItems(batch_users.cpu().numpy())
            groundTrue = [testDict[u] for u in batch_users.cpu().numpy()]
            rating = Recmodel.getUsersRating(batch_users)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        if epoch == 1000:
            with open(f'{args.dataset}_result.txt', "w+") as f:
                for i in range(len(users_list)):
                    for j in range(len(users_list[i])):
                        f.write(f"{users_list[i][j]}\t")
                        f.write(f"{rating_list[i][j].numpy().tolist()}\t")
                        f.write(f"{groundTrue_list[i][j]}\n")
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
        return results
