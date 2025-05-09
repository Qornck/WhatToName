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
from model import LightGCN
from dataloader import Loader
import gaussian_diffusion as gd
from DNN import DNN
from dataloader import DataDiffusion
from scipy.sparse import csr_matrix

args = parse_args()
print(">>SEED:", args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Loader()
Recmodel = LightGCN(args=args, dataset=dataset).cuda()
bpr = utils.BPRLoss(Recmodel, args)

original_graph = torch.FloatTensor(dataset.UserItemNet.toarray()).to(device)

if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

out_dims = eval(args.dims) + [dataset.n_apis]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.recdim, time_type="cat", norm=args.norm).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        with torch.no_grad():
            diffGraph = diffusion.p_sample(model, original_graph, args.sampling_steps, args.sampling_noise)
            top_values, top_indices = torch.topk(diffGraph, k=10, dim=1)
            diffGraph = torch.zeros_like(diffGraph).scatter_(1, top_indices, 1)
            diffGraph = csr_matrix(diffGraph.cpu().numpy())
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        optimizer.zero_grad()
        diff_loss = diff_loss['loss'].mean()
        diff_loss.backward()
        optimizer.step()
        print(f'EPOCH[{epoch+1}/{args.epochs}] {output_information}')
        # torch.save(Recmodel.state_dict(), weight_file)
finally:
    cprint(best_results)