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
from model import GNN_graphpred,GNN_together,MoE_expert
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

def check_gradients(model, clip_value=1e4):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            if grad_norm > clip_value:
                print(f"⚠️ Gradient norm too large on {name}: {grad_norm:.2f}")
    total_norm = total_norm ** 0.5
    return total_norm

def train(args, model, device, loader_list,optimizer, epoch): # para_contrastive_loss,
    # torch.autograd.set_detect_anomaly(True)
    cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader= loader_list
    max_len = max(len(cl_train_loader), len(vdss_train_loader), len(t1_2_train_loader))
    print(max_len)
    model.train()
    
    cl_iter = cycle(iter(cl_train_loader))
    vdss_iter = cycle(iter(vdss_train_loader))
    t1_2_iter = cycle(iter(t1_2_train_loader))
    ic_50_iter = cycle(iter(ic_50_train_loader))
    total_loss = 0 
    count = 0 
    warmup_epochs = 3
    warmup_start_lr = args.lr *0.001
    total_warmup_iterations = warmup_epochs * max_len
    epoch_grad_norms = []
    

    # for step, (cl_batch, vdss_batch, t1_2_batch, ic_50_batch) in enumerate(
    #         tqdm(zip(cl_train_loader, vdss_train_loader, t1_2_train_loader, ic_50_train_loader),
    #              desc="Training Iteration", file=sys.stdout, mininterval=5)):
    for step in tqdm(range(max_len), desc="Training", mininterval=5):
        if epoch < warmup_epochs: # 检查是否处于warm-up epoch
            if total_warmup_iterations > 0:
                # 1. 计算当前迭代的全局计数 current_global_iteration
                current_global_iteration = epoch * max_len + step

                # 2. 根据公式计算出当前step应该使用的学习率 lr
                lr = warmup_start_lr + (args.lr - warmup_start_lr) * (current_global_iteration / total_warmup_iterations)

                # 3. 将计算出的 lr 应用到优化器
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        # Get next batches
        cl_batch = next(cl_iter).to(device)
        vdss_batch = next(vdss_iter).to(device)
        t1_2_batch = next(t1_2_iter).to(device)
        ic_50_batch = next(ic_50_iter).to(device)
        cl_batch = cl_batch.to(device)
        vdss_batch = vdss_batch.to(device)
        t1_2_batch = t1_2_batch.to(device)
        ic_50_batch = ic_50_batch.to(device)
        count += 1 
        # print(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True))
        
        cl_pred_log = model(cl_batch.x, cl_batch.edge_index, cl_batch.edge_attr, cl_batch.batch, 0, True)
        vdss_pred_log = model(vdss_batch.x, vdss_batch.edge_index, vdss_batch.edge_attr, vdss_batch.batch, 1, True)
        t1_2_pred_log = model(t1_2_batch.x, t1_2_batch.edge_index, t1_2_batch.edge_attr, t1_2_batch.batch, 2, True)
        cl_pred_log_ic_50 = model(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, 0,True)
        vdss_pred_log_ic_50 = model(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch,  1, True)
        t1_2_pred_log_ic_50 = model(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, 2, True )

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

        importance1 = model.gates[0].weights
        importance2 = model.gates[1].weights
        importance3 = model.gates[2].weights
        importance = torch.stack([importance1,importance2,importance3],dim = 0)
        importance = importance.mean(dim=0)
        load_balance_loss = torch.var(importance)
        loss = loss_cl + loss_vdss + loss_t1_2 + args.beta * loss_ic_50 + args.lambda_balance * load_balance_loss
        # loss = sync_loss(loss)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        if step % 20 == 0:
            total_norm = check_gradients(model)
            epoch_grad_norms.append(total_norm)
            
        optimizer.step()
        avg_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        
    #     train_auc_pred_all.append(pred_log)
    #     train_auc_label_all.append(y)  
    
    # train_auc_pred_all = torch.cat(train_auc_pred_all, dim=0).detach().cpu().numpy()
    # train_auc_label_all = torch.cat(train_auc_label_all, dim=0).detach().cpu().numpy()
   
    return (total_loss)/count, avg_norm

   
def eval_task(args, model, device, loader, head_idx): # para_contrastive_loss,
    model.eval()
  
    y_scores = []
    y_list = []
  
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():

            pred_log = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, head_idx, False)[0]

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
    parser.add_argument('--batch_size', type=int, default=64, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
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
    parser.add_argument('--num_expert', type=int, default=6,
                        help='number of experts (default: 6).')
    parser.add_argument('--top_k', type=int, default=2,
                        help='number of activated experts (default: 2).')
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
    parser.add_argument('--beta', type=float, default=1.0, help='the weight of formula')    
    parser.add_argument('--lambda_balance', type=float, default=0.01, help='the weight of expert_load_balance')
    
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
    
    ### DDP
    # setup()
    # rank = dist.get_rank()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # device = torch.device(f'cuda:{local_rank}')
    
    best_epoch = 0
    torch.manual_seed(args.runseed)
    # np.random.seed(args.runseed)
    # args.seed = args.runseed 
    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bs'+'_'+str(args.batch_size)+'_'+'dropout'+'_'+str(args.dropout_ratio)+'_'+'beta'+'_'+str(args.beta)
    writer = SummaryWriter(f'{args.save}/{args.experiment_name}/')  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    os.makedirs(args.save + f"{args.experiment_name}", exist_ok=True)

    #set up model
    model = MoE_expert(args.num_layer, args.emb_dim, args.num_expert, args.top_k, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)

    if not args.input_model_file == "":
        checkpoint = torch.load(args.input_model_file, map_location=device)["model_atom_state_dict"]
        gnn_dict = {k: v for k, v in checkpoint.items() if not "graph_pred_linear" in k }
        print(gnn_dict)
        model.load_state_dict(gnn_dict, strict=False)    
    # if not args.input_model_file == "":
    #     print("load pretrained model from:", args.input_model_file)
    #     model_cl.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
    #     model_vdss.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
    #     model_t1_2.load_state_dict(torch.load(args.input_model_file, map_location=device)[f'model_atom_state_dict'])
    
    model.to(device)

    optimizer = optim.Adam([
    {"params": model.gnns.parameters(), "lr": args.lr},
    {"params": model.node_layerNorm.parameters(), "lr": args.lr},
    {"params": model.experts.parameters(), "lr": args.lr},
    {"params": model.gates.parameters(), "lr": args.lr},
    {"params": model.output_heads[0].parameters(), "lr": args.lr},
    {"params": model.output_heads[1].parameters(), "lr": args.lr*0.1},
    {"params": model.output_heads[2].parameters(), "lr": args.lr*0.1}], weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
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
    train_indices, test_indices = train_test_split(range(len(ic_50_dataset)), test_size=0.2, random_state=42)
    ic_50_train_dataset = torch.utils.data.Subset(ic_50_dataset, train_indices)
    ic_50_valid_dataset = torch.utils.data.Subset(ic_50_dataset, test_indices)

    cl_train_loader = DataLoader(cl_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    cl_val_loader = DataLoader(cl_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vdss_train_loader = DataLoader(vdss_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vdss_val_loader = DataLoader(vdss_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    t1_2_train_loader = DataLoader(t1_2_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    t1_2_val_loader = DataLoader(t1_2_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    ic_50_batchsize = int(args.batch_size / args.beta)
    ic_50_train_loader = DataLoader(ic_50_train_dataset, batch_size=ic_50_batchsize, shuffle=True, num_workers=args.num_workers)
    ic_50_val_loader = DataLoader(ic_50_valid_dataset, batch_size=ic_50_batchsize, shuffle=False, num_workers=args.num_workers)
    train_loader_list= [cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader]
    val_loader_list= [cl_val_loader,vdss_val_loader,t1_2_val_loader]
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
            
        train_mae_loss, grad= train(args, model, device, train_loader_list, optimizer, epoch) # para_contrastive_loss,
        cl_val_loss = eval_task(args, model, device, cl_val_loader,0)
        vdss_val_loss = eval_task(args, model, device, vdss_val_loader,1)
        t1_2_val_loss = eval_task(args, model, device, t1_2_val_loader,2)
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
        writer.add_scalar('grad/train', grad, epoch)
        
        if epoch % 10 == 0 or epoch == 1 or epoch == 2:
            torch.save({'model': model.state_dict()}, args.save + f"{args.experiment_name}/{epoch}.pth")
        if val_mae_loss < best_val_mae_loss:
            best_val_mae_loss = val_mae_loss
            best_epoch = epoch 
            wait = 0
            torch.save({'model': model.state_dict()}, args.save + f"{args.experiment_name}/best_model.pth")
        else:
            wait +=1
            
        if wait > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if args.scheduler:
            scheduler.step(metrics=val_mae_loss)

        print('best epoch:', best_epoch)
        print('best_val_mae_loss', best_val_mae_loss)
        print("train: %f val: %f " %(train_mae_loss, val_mae_loss))

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
