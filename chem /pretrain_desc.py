import argparse

from loader_desc import MoleculeDataset
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
import wandb

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

def check_gradients(model, clip_value=1):
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
    torch.autograd.set_detect_anomaly(True)
    cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader= loader_list
 
    max_len = max(len(cl_train_loader), len(vdss_train_loader), len(t1_2_train_loader))
    print(max_len)
    
    cl_iter = cycle(iter(cl_train_loader))
    vdss_iter = cycle(iter(vdss_train_loader))
    t1_2_iter = cycle(iter(t1_2_train_loader))
    ic_50_iter = cycle(iter(ic_50_train_loader))
    
    model.train()
    total_loss = 0 
    count = 0 
    warmup_epochs = 3
    warmup_start_lr = args.lr *0.001
    total_warmup_iterations = warmup_epochs * max_len

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
        
        count += 1 
        cl_pred_log = model(cl_batch.x, cl_batch.edge_index, cl_batch.edge_attr, cl_batch.batch,cl_batch.desc)[:,[0]]
        vdss_pred_log = model(vdss_batch.x, vdss_batch.edge_index, vdss_batch.edge_attr, vdss_batch.batch, vdss_batch.desc)[:,[1]]
        t1_2_pred_log = model(t1_2_batch.x, t1_2_batch.edge_index, t1_2_batch.edge_attr, t1_2_batch.batch, t1_2_batch.desc)[:,[2]]
        pred_log_ic_50 = model(ic_50_batch.x, ic_50_batch.edge_index, ic_50_batch.edge_attr, ic_50_batch.batch, ic_50_batch.desc)
        cl_pred_log_ic_50 = pred_log_ic_50[:,[0]]
        vdss_pred_log_ic_50 = pred_log_ic_50[:,[1]]
        t1_2_pred_log_ic_50 = pred_log_ic_50[:,[2]]

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
        # loss = loss_cl + loss_vdss + loss_t1_2
        # loss = sync_loss(loss)
        total_loss += loss.item()
        
        optimizer.zero_grad()  
        loss.backward()    
        optimizer.step()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2) # 计算梯度的L2范数
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print(f"Loss: {loss.item()}, Total Gradient Norm: {total_norm}")
    return (total_loss)/count
   
def eval_task(args, model, device, loader_list): # para_contrastive_loss,
    model.eval()
    loss_list =[]
    for i,loader in enumerate(loader_list):
        y_scores = []
        y_list = []
  
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            with torch.no_grad():
                pred_log = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.desc)[:,[i]]
                y = batch.y.reshape(batch.y.size(0), 1)
                y_scores.append(pred_log)
                y_list.append(y)

        y_scores = torch.cat(y_scores, dim=0)
        y_list = torch.cat(y_list, dim=0)
        loss = criterion(y_scores, y_list).cpu().detach().item()
        loss_list.append(loss)
    return loss_list


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--ic_50_scale',type=int, default=10, help='rate of batch_size')
    parser.add_argument('--epochs', type=int, default=100, # 1500
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
    parser.add_argument('--beta', type=float, default=1.0, help='the weight of formula')    
    
    ## loading the pretrained model 
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--input_model_vdss', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_model_t1_2', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_model_cl', type=str, default = '', help='filename to read the model (if there is any)')
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
    
    parser.add_argument('--cl_model_scale', type=float, default=1.)
    parser.add_argument('--vdss_model_scale', type=float, default=0.1)
    parser.add_argument('--t1_2_model_scale', type=float, default=0.1)
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    
    best_epoch = 0
    torch.manual_seed(args.runseed)

    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bs'+'_'+str(args.batch_size)+'_'+'dropout'+'_'+str(args.dropout_ratio)+'_'+'beta'+'_'+str(args.beta)
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project= 'PEMAL',name= args.experiment_name,config={"lr": args.lr,
                                            "bs": args.batch_size,
                                            "dropout": args.dropout_ratio,
                                            "decay": args.decay,
                                            "beta": args.beta
                                            })
    
    writer = SummaryWriter(f'{args.save}{args.experiment_name}/')  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    os.makedirs(f'{args.save}/{args.experiment_name}', exist_ok=True)
 
    num_tasks=3
    desc_emb = 216
    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type,desc_emb=desc_emb)
    
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        checkpoint = torch.load(args.input_model_file, map_location=device)['model_atom_state_dict']
        gnn_dict = {k: v for k, v in checkpoint.items() if not "graph_pred_linear" in k }
        model.load_state_dict(gnn_dict, strict=False)

    # elif not (args.input_model_vdss == "" or args.input_model_t1_2 == "" or args.input_model_cl == ""):
    #     model_vdss.load_state_dict(torch.load(args.input_model_vdss, map_location=device)['model_vdss'])

    #     model_t1_2.load_state_dict(torch.load(args.input_model_t1_2, map_location=device)['model_t1_2'])

    #     model_cl.load_state_dict(torch.load(args.input_model_cl, map_location=device)['model_cl'])
        
    model.to(device)

    optimizer= optim.Adam(
        [{"params": model.parameters()}],
        lr=args.lr,
        weight_decay=args.decay
    )

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    else:
        scheduler = None

    epoch_list = np.arange(0, args.epochs, 1)
    print('epoch_list_len',len(epoch_list))

    best_val_mae_loss=float('inf')
    wait=0
    patience = 15
    
    motif_list_path = 'dataset_reg/motif_list.pkl'
    print(f"文件 {motif_list_path} 存在，从文件中加载 motif_list...")
    with open(motif_list_path, 'rb') as f:
        motif_list = pickle.load(f)

    ## set up pk dataset 
    cl_train_dataset = MoleculeDataset("dataset_new_desc/cl_train", dataset="cl_train", motif_list=motif_list)
    cl_valid_dataset = MoleculeDataset("dataset_new_desc/cl_valid", dataset="cl_valid", motif_list=motif_list)

    vdss_train_dataset = MoleculeDataset("dataset_new_desc/vdss_train", dataset="vdss_train", motif_list=motif_list)
    vdss_valid_dataset = MoleculeDataset("dataset_new_desc/vdss_valid", dataset="vdss_valid", motif_list=motif_list)

    t1_2_train_dataset = MoleculeDataset("dataset_new_desc/t1_2_train", dataset="t1_2_train", motif_list=motif_list)
    t1_2_valid_dataset = MoleculeDataset("dataset_new_desc/t1_2_valid", dataset="t1_2_valid", motif_list=motif_list)

    ic_50_dataset = MoleculeDataset("dataset_new_desc/ic_50", dataset="ic_50", data_type="data_motif",motif_list=motif_list)

# ###9.13新增改动，添加generator，看看能否保证一致
#     g = torch.Generator()
#     g.manual_seed(7)
    ic_50_batchsize = int(args.batch_size / args.beta) if not args.beta == 0 else 2
    cl_train_loader = DataLoader(cl_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    vdss_train_loader = DataLoader(vdss_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    t1_2_train_loader = DataLoader(t1_2_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    ic_50_train_loader = DataLoader(ic_50_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    cl_val_loader = DataLoader(cl_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    vdss_val_loader = DataLoader(vdss_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    t1_2_val_loader = DataLoader(t1_2_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    # ic_50_val_loader = DataLoader(ic_50_valid_dataset, batch_size=ic_50_batchsize, shuffle=False, num_workers=args.num_workers)
    
    train_loader_list= [cl_train_loader,vdss_train_loader,t1_2_train_loader,ic_50_train_loader]
    val_loader_list = [cl_val_loader,vdss_val_loader,t1_2_val_loader]

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train_mae_loss = train(args, model, device, train_loader_list, optimizer, epoch) # para_contrastive_loss,
        # val_mae_loss = eval(args, model_list, device, val_loader_list, epoch) # para_contrastive_loss,
        cl_val_loss,vdss_val_loss,t1_2_val_loss = eval_task(args, model, device, val_loader_list)
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
        wandb.log({
        "train_loss": train_mae_loss,
        "cl_val": cl_val_loss,
        "vdss_val": vdss_val_loss,
        "t1_2_val": t1_2_val_loss,
        },step=epoch)
        # if epoch % 10 == 0 or epoch == 1 or epoch == 2:
        #     torch.save({'model_cl': model_cl.state_dict()}, args.save + f"cl/{args.experiment_name}/{epoch}.pth")
        #     torch.save({'model_vdss': model_vdss.state_dict()}, args.save + f"vdss/{args.experiment_name}/{epoch}.pth")
        #     torch.save({'model_t1_2': model_t1_2.state_dict()}, args.save + f"t1_2/{args.experiment_name}/{epoch}.pth")
        if val_mae_loss < best_val_mae_loss:
            best_val_mae_loss = val_mae_loss
            best_epoch = epoch 
            wait = 0
            torch.save(model.state_dict(), args.save + f"{args.experiment_name}/best_model.pth")

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
    new_data = pd.DataFrame({
        'best_val_mae_loss': [best_val_mae_loss],
        'best_epoch': [best_epoch]
    }, index=[args.experiment_name])
    wandb.log({'best_val_mae_loss': best_val_mae_loss,
        'best_epoch': best_epoch})
    wandb.finish()
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
