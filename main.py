import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
# ==============================
from parse import parse_args
from model import LightGCN, DenoisingNet
from dataloader import Loader
import gaussian_diffusion as gd
from DNN import DNN
from dataloader import DataDiffusion
from scipy.sparse import csr_matrix

args = parse_args()
print(">>SEED:", args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

dataset = Loader()
Recmodel = LightGCN(args=args, dataset=dataset).cuda()
bpr = utils.BPRLoss(Recmodel, args)

original_graph = torch.FloatTensor(dataset.MashupApiNet.toarray()).to(device)
# print(original_graph.shape)
# print(type(original_graph))

def csr_equal_strict(mat1, mat2):
    return (
        (mat1.shape == mat2.shape) and
        (np.array_equal(mat1.data, mat2.data)) and
        (np.array_equal(mat1.indices, mat2.indices)) and
        (np.array_equal(mat1.indptr, mat2.indptr))
    )

if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

out_dims = eval(args.dims) + [dataset.n_apis]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.recdim, time_type="cat", norm=args.norm).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.diff_lr, weight_decay=0.0)
model1 = DNN(in_dims, out_dims, args.recdim, time_type="cat", norm=args.norm).to(device)
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.diff_lr, weight_decay=0.0)

best_results = {'precision': [0, 0, 0, 0],
               'recall': [0, 0, 0, 0],
               'ndcg': [0, 0, 0, 0]}

try:
    for epoch in range(args.epochs):
        start = time.time()
        if epoch % 1 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch)
            if results['recall'][0] > best_results['recall'][0]:
                best_results['recall'][0] = results['recall'][0]
                best_results['precision'][0] = results['precision'][0]
                best_results['ndcg'][0] = results['ndcg'][0]
            if results['recall'][1] > best_results['recall'][1]:
                best_results['recall'][1] = results['recall'][1]
                best_results['precision'][1] = results['precision'][1]
                best_results['ndcg'][1] = results['ndcg'][1]
            if results['recall'][2] > best_results['recall'][2]:
                best_results['recall'][2] = results['recall'][2]
                best_results['precision'][2] = results['precision'][2]
                best_results['ndcg'][2] = results['ndcg'][2]
            if results['recall'][3] > best_results['recall'][3]:
                best_results['recall'][3] = results['recall'][3]
                best_results['precision'][3] = results['precision'][3]
                best_results['ndcg'][3] = results['ndcg'][3]
        diff_loss = diffusion.training_losses(model, original_graph, device, args.reweight)
        diff_loss1 = diffusion.training_losses(model1, original_graph, device, args.reweight)
        with torch.no_grad():
            diffGraph = diffusion.p_sample(model, original_graph, args.sampling_steps, args.sampling_noise)
            top_values, top_indices = torch.topk(diffGraph, k=args.reserve_nodes1, dim=1)
            diffGraph = torch.zeros_like(diffGraph).scatter_(1, top_indices, 1)
            diffGraph = torch.max(diffGraph, original_graph)
            diffGraph = csr_matrix(diffGraph.cpu().numpy())

            diffGraph1 = diffusion.p_sample(model1, original_graph, args.sampling_steps, args.sampling_noise)
            top_values1, top_indices1 = torch.topk(diffGraph1, k=args.reserve_nodes2, dim=1)
            diffGraph1 = torch.zeros_like(diffGraph1).scatter_(1, top_indices1, 1)
            diffGraph1 = torch.max(diffGraph1, original_graph)
            diffGraph1 = csr_matrix(diffGraph1.cpu().numpy())
        
        output_information = Procedure.BPR_train_original(dataset, Recmodel, diffGraph, diffGraph1, bpr)
        optimizer.zero_grad()
        optimizer1.zero_grad()
        diff_loss = diff_loss['loss'].mean() + diff_loss1['loss'].mean()
        diff_loss.backward()
        optimizer.step()
        optimizer1.step()
        print(f'EPOCH[{epoch+1}/{args.epochs}] {output_information} diff_loss {diff_loss}')
finally:
    recall_20 = best_results['recall'][2]
    ndcg_20 = best_results['ndcg'][2]
    recall_40 = best_results['recall'][3]
    ndcg_40 = best_results['ndcg'][3]
    torch.save(Recmodel.state_dict(), f"./save_model/Recmodel_{recall_20:.4f}_{ndcg_20:.4f}_{recall_40:.4f}_{ndcg_40:.4f}_{args.bpr_batch}_{args.recdim}_{args.layer}_{args.lr}_{args.ssl_temp}_{args.ssl_weight}_{args.mean_type}_{args.reserve_nodes1}_{args.reserve_nodes2}_{args.seed}.pth")
    cprint(best_results)