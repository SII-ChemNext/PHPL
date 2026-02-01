import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import math 
import sys
import random
from itertools import cycle,islice
from model import GNN_graphpred
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

criterion = nn.L1Loss()

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()
    
def sync_loss(loss):
    # 假设 loss 是一个标量张量
    dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        loss /= dist.get_world_size()  # 求平均
    return loss

def sample_dataset_by_shuffle(dataset, sample_size):
    """
    对数据集进行随机打乱，并选取前 sample_size 个样本。
    参数：
      dataset: 支持索引和 len() 的数据集
      sample_size: 目标采样数目
    返回：
      一个包含选定样本的列表
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # 每个 epoch 随机打乱索引（你也可以在每个 epoch 外部不设置 seed 以确保每次不同）
    selected_indices = indices[:sample_size]
    sampled_dataset = [dataset[i] for i in selected_indices]
    return sampled_dataset

def cyclic_ic_50_loader(dataset, epoch, sample_size=80000, batch_size=256, shuffle=True):
    """顺序采样 ic_50 数据集，每个 epoch 选择不同的 10W 个样本"""
    total_samples = len(dataset)  # ic_50 数据总量
    num_chunks = math.ceil(total_samples / sample_size)  # 计算能分成多少块

    # 计算当前 epoch 对应的样本起始索引
    start_idx = (epoch % num_chunks) * sample_size  # 当 epoch 达到 num_chunks 时，从头开始采样
    
    # 如果当前块超出样本总量，取剩余部分
    end_idx = min(start_idx + sample_size, total_samples)  # 结束索引不能超过总样本数
    
    indices = list(range(start_idx, end_idx))  # 选取连续的样本
    sampled_subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(sampled_subset, batch_size=batch_size, shuffle=shuffle)

def train(args, model_list, device, loader_list,optimizer, epoch): # para_contrastive_loss,
    torch.autograd.set_detect_anomaly(True)
    cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader= loader_list
    max_len = max(len(cl_train_loader), len(vdss_train_loader), len(t1_2_train_loader))
    print(max_len)

    model_cl,model_vdss,model_t1_2 = model_list
    
    cl_iter = cycle(iter(cl_train_loader))
    vdss_iter = cycle(iter(vdss_train_loader))
    t1_2_iter = cycle(iter(t1_2_train_loader))
    ic_50_iter = cycle(iter(ic_50_train_loader))
    
    model_cl.train()
    model_vdss.train()
    model_t1_2.train()
    total_loss = 0 
    count = 0 
    
    # for step, (cl_batch, vdss_batch, t1_2_batch, ic_50_batch) in enumerate(
    #         tqdm(islice(
    #             zip(
    #                 cycle(cl_train_loader),
    #                 cycle(vdss_train_loader),
    #                 cycle(t1_2_train_loader),
    #                 cycle(ic_50_train_loader)
    #                 ),max_len),
    #              total=max_len,
    #              desc="Training Iteration", file=sys.stdout, mininterval=5)):
    for step in tqdm(range(max_len), desc="Training", mininterval=5):
        # Get next batches
        cl_batch = next(cl_iter).to(device)
        vdss_batch = next(vdss_iter).to(device)
        t1_2_batch = next(t1_2_iter).to(device)
        ic_50_batch = next(ic_50_iter).to(device)

        # cl_batch = cl_batch.to(device)
        # vdss_batch = vdss_batch.to(device)
        # t1_2_batch = t1_2_batch.to(device)
        # ic_50_batch = ic_50_batch.to(device)
        count += 1 
        # print(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True))
        
        cl_pred_log = model_cl(cl_batch.x, cl_batch.edge_index, cl_batch.edge_attr, cl_batch.batch, True)
        vdss_pred_log = model_vdss(vdss_batch.x, vdss_batch.edge_index, vdss_batch.edge_attr, vdss_batch.batch, True)
        t1_2_pred_log = model_t1_2(t1_2_batch.x, t1_2_batch.edge_index, t1_2_batch.edge_attr, t1_2_batch.batch, True)
        cl_pred_log_ic_50 = model_cl(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)
        vdss_pred_log_ic_50 = model_vdss(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)
        t1_2_pred_log_ic_50 = model_t1_2(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)

        cl_y = cl_batch.y.reshape(cl_batch.y.size(0), 1)
        vdss_y = vdss_batch.y.reshape(vdss_batch.y.size(0), 1)
        t1_2_y = t1_2_batch.y.reshape(t1_2_batch.y.size(0), 1)
       
        loss_cl = criterion(cl_pred_log, cl_y)
        loss_vdss = criterion(vdss_pred_log, vdss_y)
        loss_t1_2 = criterion(t1_2_pred_log, t1_2_y)
        loss_ic_50 = criterion(
        cl_pred_log_ic_50 + t1_2_pred_log_ic_50,
        vdss_pred_log_ic_50 + torch.log(torch.tensor(17.71, device=device))
        )
        loss = loss_cl + loss_vdss + loss_t1_2 + args.beta * loss_ic_50
        # loss = sync_loss(loss)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #     train_auc_pred_all.append(pred_log)
    #     train_auc_label_all.append(y)  
    
    # train_auc_pred_all = torch.cat(train_auc_pred_all, dim=0).detach().cpu().numpy()
    # train_auc_label_all = torch.cat(train_auc_label_all, dim=0).detach().cpu().numpy()
   
    return (total_loss)/count
   
def eval(args, model_list, device, loader_list, epoch): # para_contrastive_loss,
    cl_valid_loader,vdss_valid_loader,t1_2_valid_loader= loader_list

    model_cl,model_vdss,model_t1_2 = model_list
    count = 0
    model_cl.eval()
    model_vdss.eval()
    model_t1_2.eval()
    total_loss = 0
    
    y_scores_cl = []
    y_scores_vdss = []
    y_scores_t1_2 = []

    
    y_cl = []
    y_vdss = []
    y_t1_2 = []


    for step, (cl_batch, vdss_batch, t1_2_batch) in enumerate(
            tqdm(zip(cl_valid_loader, vdss_valid_loader, t1_2_valid_loader),
                 desc="validate Iteration", file=sys.stdout, mininterval=5)):
        
        cl_batch = cl_batch.to(device)
        vdss_batch = vdss_batch.to(device)
        t1_2_batch = t1_2_batch.to(device)
        ic_50_batch = ic_50_batch.to(device)
        count += 1 
        # print(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True))
        
        cl_pred_log = model_cl(cl_batch.x, cl_batch.edge_index, cl_batch.edge_attr, cl_batch.batch, True)
        vdss_pred_log = model_vdss(vdss_batch.x, vdss_batch.edge_index, vdss_batch.edge_attr, vdss_batch.batch, True)
        t1_2_pred_log = model_t1_2(t1_2_batch.x, t1_2_batch.edge_index, t1_2_batch.edge_attr, t1_2_batch.batch, True)
        # cl_pred_log_ic_50 = model_cl(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)
        # vdss_pred_log_ic_50 = model_vdss(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)
        # t1_2_pred_log_ic_50 = model_t1_2(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, True)

        cl_y = cl_batch.y.reshape(cl_batch.y.size(0), 1)
        vdss_y = vdss_batch.y.reshape(vdss_batch.y.size(0), 1)
        t1_2_y = t1_2_batch.y.reshape(t1_2_batch.y.size(0), 1)
       
        # y_scores_auc.append(auc_pred_log)   
        y_scores_cl.append(cl_pred_log)
        y_scores_vdss.append(vdss_pred_log)
        y_scores_t1_2.append(t1_2_pred_log)
        
        # y_auc.append(auc_y)
        y_cl.append(cl_y)
        y_vdss.append(vdss_y)
        y_t1_2.append(t1_2_y)
            
    # y_scores_auc = torch.cat(y_scores_auc, dim=0)
    y_scores_cl = torch.cat(y_scores_cl, dim=0)
    y_scores_vdss = torch.cat(y_scores_vdss, dim=0)
    y_scores_t1_2 = torch.cat(y_scores_t1_2, dim=0)
    
    # y_auc = torch.cat(y_auc, dim=0)
    y_cl = torch.cat(y_cl, dim=0)
    y_vdss = torch.cat(y_vdss, dim=0)
    y_t1_2 = torch.cat(y_t1_2, dim=0)
    
    # loss_auc = criterion(y_scores_auc, y_auc).cpu().detach().item()
    loss_cl = criterion(y_scores_cl, y_cl).detach()
    loss_vdss = criterion(y_scores_vdss, y_vdss).detach()
    loss_t1_2 = criterion(y_scores_t1_2, y_t1_2).detach()
    # loss_ic_50 = criterion(
    #     cl_pred_log_ic_50 + t1_2_pred_log_ic_50,
    #     vdss_pred_log_ic_50 + torch.log(torch.tensor(17.71, device=device))
    #     ).cpu().detach().item()

    # 计算总 loss（所有 loss 之和）
    total_loss_mean = loss_cl+ loss_vdss + loss_t1_2
    # total_loss_mean = sync_loss(total_loss_mean).item()

    print(f"Mean Loss CL: {loss_cl:.6f}, "
        f"Mean Loss VDSS: {loss_vdss:.6f}, "
        f"Mean Loss T1_2: {loss_t1_2:.6f}, "
        # f"Mean Loss IC_50: {loss_ic_50:.6f}, "
        f"Total Mean Loss: {total_loss:.6f}")
    return total_loss_mean

def eval_task(args, model, device, loader): # para_contrastive_loss,
    model.eval()
  
    y_scores = []
    y_list = []
  
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():

            pred_log = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)[0]

            y = batch.y.reshape(batch.y.size(0), 1)
            y_scores.append(pred_log)
            y_list.append(y)

    # print(type(y_scores))  # 检查 y_scores 的类型
    # print(y_scores)

            
    y_scores = torch.cat(y_scores, dim=0)
    y_list = torch.cat(y_list, dim=0)
    loss = criterion(y_scores, y_list).cpu().detach().item()

    return loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--ic_50_scale',type=int, default=10, help='rate of batch_size')
    parser.add_argument('--epochs', type=int, default=60, # 1500
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, # 0.0001
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-9,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = '', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--beta', type=float, default="0", help='the weight of formula')    
    
    ## loading the pretrained model 
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=True)
    parser.add_argument('--experiment_name', type=str, default="PEMAL")
    
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    ## add the multi-task alpha 
    parser.add_argument('--t1_2_alpha', type=float, default=1.)
    parser.add_argument('--k3_alphla', type=float, default=1.)
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()


    # args.use_early_stopping = args.dataset in ("muv", "hiv")
    # args.scheduler = args.dataset in ("bace")
    # setup()
    # rank = dist.get_rank()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # device = torch.device(f'cuda:{local_rank}')
    
    best_epoch = 0
    torch.manual_seed(args.runseed)
    # np.random.seed(args.runseed)
    # args.seed = args.runseed 
    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bs'+'_'+str(args.batch_size)+'_'+'dropout'+'_'+str(args.dropout_ratio)+'_'+'beta'+'_'+str(args.beta)
    writer = SummaryWriter(f'runs/0426/finetune/{args.experiment_name}')  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    os.makedirs(args.save + f"cl/{args.experiment_name}", exist_ok=True)
    os.makedirs(args.save + f"vdss/{args.experiment_name}", exist_ok=True)
    os.makedirs(args.save + f"t1_2/{args.experiment_name}", exist_ok=True)
    os.makedirs(f'runs/0410/finetune/{args.experiment_name}', exist_ok=True)
 
    num_tasks=1
    #set up model
    model_cl = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    model_vdss = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    model_t1_2 = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        model_cl.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
        model_vdss.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
        model_t1_2.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
    
    model_cl.to(device)
    model_vdss.to(device)
    model_t1_2.to(device)
    
    # model_cl = DDP(model_cl, device_ids=[rank])
    # model_vdss = DDP(model_vdss, device_ids=[rank])
    # model_t1_2 = DDP(model_t1_2, device_ids=[rank])
    
    model_list = [model_cl,model_vdss,model_t1_2]
    
        #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model_cl.parameters()})
    model_param_group.append({"params": model_vdss.parameters()})
    model_param_group.append({"params": model_t1_2.parameters(),"lr":args.lr*0.1})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        scheduler = None

    epoch_list = np.arange(0, args.epochs, 1)
    print('epoch_list_len',len(epoch_list))

    best_val_mae_loss=float('inf')
    wait=0
    patience = 21
    
    motif_list_path = 'dataset_reg/motif_list.pkl'
    print(f"文件 {motif_list_path} 存在，从文件中加载 motif_list...")
    with open(motif_list_path, 'rb') as f:
        motif_list = pickle.load(f)
    # if os.path.exists(motif_list_path):
    #     print(f"文件 {motif_list_path} 存在，从文件中加载 motif_list...")
    #     with open(motif_list_path, 'rb') as f:
    #         motif_list = pickle.load(f)
    # else:
    #     print(f"文件 {motif_list_path} 不存在，运行 get_motif_list...")
    #     motif_list = get_motif_list_parallel(smiles_list)
    #     # 保存到指定路径
    #     os.makedirs(os.path.dirname(motif_list_path), exist_ok=True)  # 确保输出文件夹存在
    #     with open(motif_list_path, 'wb') as f:
    #         pickle.dump(motif_list, f)
    #     print(f"motif_list 已保存到 {motif_list_path}")    
        
    ## set up pk dataset 
    cl_train_dataset = MoleculeDataset("dataset_reg/cl_train", dataset="cl_train", motif_list=motif_list)
    cl_valid_dataset = MoleculeDataset("dataset_reg/cl_valid", dataset="cl_valid", motif_list=motif_list)

    vdss_train_dataset = MoleculeDataset("dataset_reg/vdss_train", dataset="vdss_train", motif_list=motif_list)
    vdss_valid_dataset = MoleculeDataset("dataset_reg/vdss_valid", dataset="vdss_valid", motif_list=motif_list)

    t1_2_train_dataset = MoleculeDataset("dataset_reg/t1_2_train", dataset="t1_2_train", motif_list=motif_list)
    t1_2_valid_dataset = MoleculeDataset("dataset_reg/t1_2_valid", dataset="t1_2_valid", motif_list=motif_list)

    ic_50_dataset = MoleculeDataset("dataset_reg/ic_50", dataset="ic_50", data_type="data_motif",motif_list=motif_list)
    # train_indices, test_indices = train_test_split(range(len(ic_50_dataset)), test_size=0.2, random_state=42)
    # ic_50_train_dataset = torch.utils.data.Subset(ic_50_dataset, train_indices)
    # ic_50_valid_dataset = torch.utils.data.Subset(ic_50_dataset, test_indices)
    
    ic_50_batchsize = int(args.batch_size / args.beta)
    cl_train_loader = DataLoader(cl_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vdss_train_loader = DataLoader(vdss_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    t1_2_train_loader = DataLoader(t1_2_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    ic_50_train_loader = DataLoader(ic_50_dataset, batch_size=ic_50_batchsize, shuffle=True, num_workers=args.num_workers)
    cl_val_loader = DataLoader(cl_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    vdss_val_loader = DataLoader(vdss_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    t1_2_val_loader = DataLoader(t1_2_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # ic_50_val_loader = DataLoader(ic_50_valid_dataset, batch_size=ic_50_batchsize, shuffle=False, num_workers=args.num_workers)
    
    train_loader_list= [cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader]
    val_loader_list= [cl_val_loader,vdss_val_loader,t1_2_val_loader]

# 计算每个数据集的样本数
    # n_cl    = len(cl_train_dataset)
    # n_vdss  = len(vdss_train_dataset)
    # n_t1_2  = len(t1_2_train_dataset)

    # # 找到最小的数据集大小作为采样样本数
    # min_count = min(n_cl, n_vdss, n_t1_2)
    # min_val = min(len(cl_valid_dataset),len(vdss_valid_dataset),len(t1_2_valid_dataset))
    # print(f"数据集样本数：cl: {n_cl}, vdss: {n_vdss}, t1_2: {n_t1_2}")
    # print(f"每个 epoch 的采样样本数取：{min_count}")
    
    # cl_train_loader = DataLoader(cl_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # cl_val_loader = DataLoader(cl_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # vdss_train_loader = DataLoader(vdss_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # vdss_val_loader = DataLoader(vdss_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # t1_2_train_loader = DataLoader(t1_2_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # t1_2_val_loader = DataLoader(t1_2_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ic_50_train_loader = DataLoader(ic_50_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # ic_50_val_loader = DataLoader(ic_50_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        # cl_train_sample = sample_dataset_by_shuffle(cl_train_dataset, min_count)
        # vdss_train_sample = sample_dataset_by_shuffle(vdss_train_dataset, min_count)
        # t1_2_train_sample = sample_dataset_by_shuffle(t1_2_train_dataset, min_count)
        # ic_50_train_sample=sample_dataset_by_shuffle(ic_50_train_dataset, 80000)
        # cl_train_sampler = DistributedSampler(cl_train_sample, shuffle=True)
        # vdss_train_sampler = DistributedSampler(vdss_train_sample, shuffle=True)
        # t1_2_train_sampler = DistributedSampler(t1_2_train_sample, shuffle=True)
        # ic_50_train_sampler = DistributedSampler(ic_50_train_sample, shuffle=True)
        
        # cl_valid_sample = sample_dataset_by_shuffle(cl_valid_dataset, min_val)
        # vdss_valid_sample = sample_dataset_by_shuffle(vdss_valid_dataset, min_val)
        # t1_2_valid_sample = sample_dataset_by_shuffle(t1_2_valid_dataset, min_val)
        # ic_50_valid_sample=sample_dataset_by_shuffle(ic_50_valid_dataset, 10000)
        # cl_valid_sampler = DistributedSampler(cl_valid_sample, shuffle=False)
        # vdss_valid_sampler = DistributedSampler(vdss_valid_sample, shuffle=False)
        # t1_2_valid_sampler = DistributedSampler(t1_2_valid_sample, shuffle=False)
        # ic_50_valid_sampler = DistributedSampler(ic_50_valid_sample, shuffle=False)

        # cl_train_loader = DataLoader(cl_train_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=cl_train_sampler)
        # vdss_train_loader = DataLoader(vdss_train_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=vdss_train_sampler)
        # t1_2_train_loader = DataLoader(t1_2_train_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=t1_2_train_sampler)
        # ic_50_train_loader = DataLoader(ic_50_train_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=ic_50_train_sampler)
        # cl_val_loader = DataLoader(cl_valid_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=cl_valid_sampler)
        # vdss_val_loader = DataLoader(vdss_valid_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=vdss_valid_sampler)
        # t1_2_val_loader = DataLoader(t1_2_valid_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=t1_2_valid_sampler)
        # ic_50_val_loader = DataLoader(ic_50_valid_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,sampler=ic_50_valid_sampler)
        
        # train_loader_list= [cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader]
        # val_loader_list= [cl_val_loader,vdss_val_loader,t1_2_val_loader,ic_50_val_loader]
        train_mae_loss = train(args, model_list, device, train_loader_list, optimizer, epoch) # para_contrastive_loss,
        # val_mae_loss = eval(args, model_list, device, val_loader_list, epoch) # para_contrastive_loss,
        cl_val_loss = eval_task(args, model_cl, device, cl_val_loader)
        vdss_val_loss = eval_task(args, model_vdss, device, vdss_val_loader)
        t1_2_val_loss = eval_task(args, model_t1_2, device, t1_2_val_loader)
        val_mae_loss = cl_val_loss + vdss_val_loss + t1_2_val_loss
        # if dist.get_rank() == 0:
        print(f"Mean Loss CL: {cl_val_loss:.6f}, "
        f"Mean Loss VDSS: {vdss_val_loss:.6f}, "
        f"Mean Loss T1_2: {t1_2_val_loss:.6f}, "
        # f"Mean Loss IC_50: {loss_ic_50:.6f}, "
        f"Total Mean Loss: {val_mae_loss:.6f}")
        
        writer.add_scalar('Loss/train', train_mae_loss, epoch)
        writer.add_scalar('CL_Loss/valid', cl_val_loss, epoch)
        writer.add_scalar('VDSS_Loss/valid', vdss_val_loss, epoch)
        writer.add_scalar('T1_2_Loss/valid', t1_2_val_loss, epoch)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == 2:
            torch.save({'model_cl': model_cl.state_dict()}, args.save + f"cl/{args.experiment_name}/{epoch}.pth")
            torch.save({'model_vdss': model_vdss.state_dict()}, args.save + f"vdss/{args.experiment_name}/{epoch}.pth")
            torch.save({'model_t1_2': model_t1_2.state_dict()}, args.save + f"t1_2/{args.experiment_name}/{epoch}.pth")
        if val_mae_loss < best_val_mae_loss:
            best_val_mae_loss = val_mae_loss
            best_epoch = epoch 
            wait = 0
            torch.save({'model_cl': model_cl.state_dict()}, args.save + f"cl/{args.experiment_name}/best_model.pth")
            torch.save({'model_vdss': model_vdss.state_dict()}, args.save + f"vdss/{args.experiment_name}/best_model.pth")
            torch.save({'model_t1_2': model_t1_2.state_dict()}, args.save + f"t1_2/{args.experiment_name}/best_model.pth")
        else:
            wait +=1
            
        if wait > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if scheduler is not None:
            scheduler.step(metrics=val_mae_loss)
            

        print('best epoch:', best_epoch)
        print('best_val_mae_loss', best_val_mae_loss)
        print("train: %f val: %f " %(train_mae_loss, val_mae_loss))
    
        # dataframe = pd.DataFrame({'best_val_mae_loss': [best_val_mae_loss],
        #                             'best epoch': [best_epoch]})
        # dataframe.to_csv(args.save+args.experiment_name+"_"+"result.csv", index=False)
    save_path = args.save+"result.csv"

    writer.close()
    # 检查文件是否存在，存在则读取并追加新数据，否则新建
    new_data = pd.DataFrame({
        'best_val_mae_loss': [best_val_mae_loss],
        'best_epoch': [best_epoch]
    }, index=[args.experiment_name])

    # 检查文件是否存在，存在则读取并追加新数据，否则新建
    if os.path.exists(save_path):
        existing_data = pd.read_csv(save_path, index_col='experiment_name')
        # 使用 pd.concat 合并数据
        updated_data = pd.concat([existing_data, new_data])
    else:
        updated_data = new_data

    # 保存到CSV，保留索引（experiment_name作为行标签）
    updated_data.to_csv(save_path, index_label='experiment_name')
    # cleanup()
    
if __name__ == "__main__":
    main()
