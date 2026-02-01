import argparse
import copy
import pickle

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tqdm import tqdm
import numpy as np

from model import GNN_graphpred, EMA
from sklearn.metrics import roc_auc_score, accuracy_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
import sys

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas 
import math 

import matplotlib.pyplot as plt 

from model import MLPregression 
from sklearn.manifold import TSNE 
from torch.utils.tensorboard import SummaryWriter
import wandb

criterion = nn.L1Loss()

K_1 = 16846
K_2 = 17.71 # 18

def remove_module_prefix(state_dict):
    """去掉多卡训练保存的state_dict中的'module.'前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉'module.'前缀
        else:
            new_state_dict[k] = v
    return new_state_dict

def r2_score(y_true, y_pred):
    """
    y_true: torch.Tensor, shape (N,)
    y_pred: torch.Tensor, shape (N,)
    """
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def train(args, vdss_model, t1_2_model, cl_model,  para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, optimizer, epoch): # para_contrastive_loss,
    vdss_model.train()
    t1_2_model.train()
    cl_model.train()
   
    total_loss = 0 
    count = 0 
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        
        k_1 = para_function_auc_cl * K_1
        k_3 = para_function_cl_vdss_t1_2 * K_2
        
        auc_pred_log = torch.log(k_1)-cl_pred_log
        
        auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
        cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
        vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
        t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
        
        loss_auc = criterion(auc_pred_log, auc_y)
        loss_cl = criterion(cl_pred_log, cl_y)
        loss_vdss = criterion(vdss_pred_log, vdss_y)
        loss_t1_2 = criterion(t1_2_pred_log, t1_2_y)
        
        loss_k3 = criterion(cl_pred_log+t1_2_pred_log, vdss_pred_log+torch.log(k_3))
        loss =  loss_auc + loss_cl + loss_vdss + loss_t1_2 + args.temp * loss_k3
       
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cl_model.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(vdss_model.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(t1_2_model.parameters(), max_norm=0.5)
        optimizer.step()

    return (total_loss)/count
   

def eval(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, epoch): # para_contrastive_loss,
    vdss_model.eval()
    t1_2_model.eval()
    cl_model.eval()
    
    y_scores_cl = []
    y_scores_vdss = []
    y_scores_t1_2 = []
    y_scores_auc = []
    
    y_cl = []
    y_vdss = []
    y_t1_2 = []
    y_auc = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", file=sys.stdout)):
        batch = batch.to(device)

        with torch.no_grad():
            vdss_pred_log, vdss_representation = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
            t1_2_pred_log, t1_2_representation = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
            cl_pred_log , cl_representation = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
         
            k_1 = para_function_auc_cl * K_1
            
            auc_pred_log = torch.log(k_1)-cl_pred_log
            
            auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
            cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
            vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
            t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
            
            y_scores_auc.append(auc_pred_log)
            y_scores_cl.append(cl_pred_log)
            y_scores_vdss.append(vdss_pred_log)
            y_scores_t1_2.append(t1_2_pred_log)
            
            y_auc.append(auc_y)
            y_cl.append(cl_y)
            y_vdss.append(vdss_y)
            y_t1_2.append(t1_2_y)
            
    y_scores_auc = torch.cat(y_scores_auc, dim=0)
    y_scores_cl = torch.cat(y_scores_cl, dim=0)
    y_scores_vdss = torch.cat(y_scores_vdss, dim=0)
    y_scores_t1_2 = torch.cat(y_scores_t1_2, dim=0)
    
    y_auc = torch.cat(y_auc, dim=0)
    y_cl = torch.cat(y_cl, dim=0)
    y_vdss = torch.cat(y_vdss, dim=0)
    y_t1_2 = torch.cat(y_t1_2, dim=0)
    
    loss_auc = criterion(y_scores_auc, y_auc).cpu().detach().item()
    loss_cl = criterion(y_scores_cl, y_cl).cpu().detach().item()
    loss_vdss = criterion(y_scores_vdss, y_vdss).cpu().detach().item()
    loss_t1_2 = criterion(y_scores_t1_2, y_t1_2).cpu().detach().item()
    
    r2_auc = r2_score(y_auc,y_scores_auc)
    r2_cl = r2_score(y_cl,y_scores_cl)
    r2_vdss = r2_score(y_vdss,y_scores_vdss)
    r2_t1_2 = r2_score(y_t1_2,y_scores_t1_2)
    
    ## mfce 
    # auc_mfce = torch.exp(torch.median(torch.abs(y_scores_auc - y_auc)))
    # cl_mfce = torch.exp(torch.median(torch.abs(y_scores_cl - y_cl)))
    # vdss_mfce = torch.exp(torch.median(torch.abs(y_scores_vdss - y_vdss)))
    # t1_2_mfce = torch.exp(torch.median(torch.abs(y_scores_t1_2 - y_t1_2)))
    
        
    loss_all = (loss_auc + loss_cl + loss_vdss + loss_t1_2 )
   
    return loss_auc, loss_cl, loss_vdss, loss_t1_2, loss_all, r2_auc, r2_cl, r2_vdss,r2_t1_2
           
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, # 60 # 1500
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, # 0.0001
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-5,
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
    parser.add_argument('--dataset', type=str, default = '4', help='root directory of dataset. For now, only classification.')
    
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_cl_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'checkpoints/gin_100.pth', help='output filename')
    
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=True)
    parser.add_argument('--experiment_name', type=str, default="graphmae")
    
    
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    
    ## add the multi-task alpha 
    parser.add_argument('--t1_2_alpha', type=float, default=1.)
    parser.add_argument('--k3_alphla', type=float, default=1.)
    
    
    ## add the each model scale 
    parser.add_argument('--cl_model_scale', type=float, default=0.1)
    parser.add_argument('--vdss_model_scale', type=float, default=1)
    parser.add_argument('--t1_2_model_scale', type=float, default=0.1)
    parser.add_argument('--temp', type=int, default=1 ,help="ratio of formula")
    
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()
    
    os.environ["WANDB_MODE"] = "offline"
    
    global motif_list
    motif_list_path = 'dataset_reg/motif_list.pkl'
        # 读取数据
    input_df = pd.read_csv('dataset_reg/ic_50/raw/ic_50.csv')
    smiles_list = list(input_df.smiles)
    
    if os.path.exists(motif_list_path):
        print(f"文件 {motif_list_path} 存在，从文件中加载 motif_list...")
        with open(motif_list_path, 'rb') as f:
            motif_list = pickle.load(f)
    else:
        print(f"文件 {motif_list_path} 不存在，运行 get_motif_list...")
        motif_list = get_motif_list_parallel(smiles_list)
        # 保存到指定路径
        os.makedirs(os.path.dirname(motif_list_path), exist_ok=True)  # 确保输出文件夹存在
        with open(motif_list_path, 'wb') as f:
            pickle.dump(motif_list, f)
        print(f"motif_list 已保存到 {motif_list_path}")
    
    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device), requires_grad=True)
    para_function_cl_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    
    para_loss.data.fill_(0.5)
    para_function_cl_vdss_t1_2.data.fill_(1.)
    para_function_auc_cl.data.fill_(1.)
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    # args.seed = args.runseed 
    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bs'+'_'+str(args.batch_size)+'_'+'drop'+'_'+str(args.dropout_ratio)+'_'+'temp'+'_'+str(args.temp)
    os.makedirs(args.save+args.experiment_name,exist_ok=True)
    writer = SummaryWriter(f'{args.save}{args.experiment_name}/')  
    wandb.init(project= 'PEMAL_final_0611ema_newdataset',name= args.experiment_name,dir= "./wandb/0610/final_finetune_mask0.5_reludrop/",
            config={
                "lr": args.lr,
                "bs": args.batch_size,
                "dropout": args.dropout_ratio,
                "decay": args.decay,
                "cl_model_scale":args.cl_model_scale,
                "vdss_model_scale":args.vdss_model_scale,
                "t1_2_model_scale":args.t1_2_model_scale,
                })
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset == "4":
        num_tasks = 1
        train_dataset_name = "4_train"
        valid_dataset_name = "4_valid"
    else:
        raise ValueError("Invalid dataset name.")

    ## set up pk dataset 
    train_dataset = MoleculeDataset("dataset_reg/"+train_dataset_name, dataset=train_dataset_name, motif_list=motif_list)
    valid_dataset = MoleculeDataset("dataset_reg/"+valid_dataset_name, dataset=valid_dataset_name, motif_list=motif_list)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)   
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    vdss_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    t1_2_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    cl_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)

    
    if not args.input_vdss_model_file == "":
        print("load pretrained model from:", args.input_vdss_model_file)
        vdss_model.load_state_dict(remove_module_prefix(torch.load(args.input_vdss_model_file, map_location=device)['model_vdss']))
    
    if not args.input_t1_2_model_file == "":
        print("load pretrained model from:", args.input_t1_2_model_file)
        t1_2_model.load_state_dict(remove_module_prefix(torch.load(args.input_t1_2_model_file, map_location=device)['model_t1_2']))
        
    if not args.input_cl_model_file == "":
        print("load pretrained model from:", args.input_cl_model_file)
        cl_model.load_state_dict(remove_module_prefix(torch.load(args.input_cl_model_file, map_location=device)['model_cl']))
    
    cl_model_ema = AveragedModel(
    cl_model,
    device=device,
    multi_avg_fn=get_ema_multi_avg_fn(decay=0.99) 
    vdss_model_ema = AveragedModel(
    vdss_model,
    device=device,
    multi_avg_fn=get_ema_multi_avg_fn(decay=0.99) 
)
    t1_2_model_ema = AveragedModel(
    t1_2_model,
    device=device,
    multi_avg_fn=get_ema_multi_avg_fn(decay=0.99) 
)
    
    vdss_model.to(device)
    t1_2_model.to(device)
    cl_model.to(device)
    
    model_list = [cl_model, vdss_model, t1_2_model, cl_model_ema, vdss_model_ema, t1_2_model_ema]
    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    # model_param_group.append({"params": vdss_model.gnn.parameters()})
    # model_param_group.append({'params': cl_model.gnn.parameters()})
    # model_param_group.append({'params': t1_2_model.gnn.parameters()})

    model_param_group.append({"params": para_loss})
    model_param_group.append({"params": para_function_cl_vdss_t1_2})
    model_param_group.append({'params': para_function_auc_cl})

    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": vdss_model.pool.parameters(), "lr":args.lr*args.vdss_model_scale})
    #     model_param_group.append({'params': cl_model.pool.parameters(), 'lr':args.lr*args.lr_scale})
    #     model_param_group.append({'params': t1_2_model.pool.parameters(), "lr":args.lr * args.t1_2_model_scale})
    model_param_group.append({"params": vdss_model.parameters(), "lr":args.lr*args.vdss_model_scale})
    model_param_group.append({'params': cl_model.parameters(), 'lr':args.lr*args.cl_model_scale })
    model_param_group.append({'params': t1_2_model.parameters(), "lr":args.lr * args.t1_2_model_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = None
        
    epoch_list = np.arange(0, args.epochs, 1)
    print('epoch_list_len',len(epoch_list))
    train_mse_loss_list = []
    val_mse_loss_list = []
    
    best_val_mae_loss=float('inf')
    best_true_val_mse_loss = float('inf')
    
    wait = 0
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_mae_loss = train(args, model_list, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
        train_mse_loss_list.append(train_mae_loss)
        print(para_function_cl_vdss_t1_2, para_function_auc_cl,)
        auc_val_loss, cl_val_loss, vdss_val_loss, t1_2_val_loss, all_loss, r2_auc, r2_cl, r2_vdss,r2_t1_2 = eval(args, vdss_model_ema, t1_2_model_ema, cl_model_ema, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, val_loader, epoch) # para_contrastive_loss,
        
        writer.add_scalar('Loss/train', train_mae_loss, epoch)
        writer.add_scalar('AUC_Loss/valid', auc_val_loss, epoch)
        writer.add_scalar('CL_Loss/valid', cl_val_loss, epoch)
        writer.add_scalar('VDSS_Loss/valid', vdss_val_loss, epoch)
        writer.add_scalar('T1_2_Loss/valid', t1_2_val_loss, epoch)
        
        if epoch == 1 :
            torch.save({'cl_model_state_dict':cl_model.state_dict(),
                        'vdss_model_state_dict':vdss_model.state_dict(),
                        't1_2_model_state_dict':t1_2_model.state_dict()
                        }, args.save + f"{args.experiment_name}/best_model.pth")
            best_auc_loss = auc_val_loss
            best_cl_loss = cl_val_loss 
            best_vdss_loss = vdss_val_loss
            best_t1_2_loss = t1_2_val_loss
            best_r2_cl = r2_cl
            best_r2_vdss = r2_vdss
            best_r2_auc = r2_auc
            best_r2_t1_2 = r2_t1_2
                        
        if scheduler is not None:
            scheduler.step(metrics=all_loss)
            
        # checkpoint = torch.load(args.save + f"{args.experiment_name}/best_model.pth")
        # if best_cl_loss>cl_val_loss:
        #     best_auc_loss = auc_val_loss
        #     best_cl_loss = cl_val_loss
        #     checkpoint['cl_model_state_dict'] = cl_model.state_dict()
        #     best_r2_cl = r2_cl
        #     best_r2_auc = r2_auc
        # if best_vdss_loss>vdss_val_loss:
        #     best_vdss_loss = vdss_val_loss 
        #     checkpoint['vdss_model_state_dict'] = vdss_model.state_dict()
        #     best_r2_vdss = r2_vdss
        # if best_t1_2_loss>t1_2_val_loss:
        #     best_t1_2_loss = t1_2_val_loss 
        #     checkpoint['t1_2_model_state_dict'] = t1_2_model.state_dict()
        #     best_r2_t1_2 = r2_t1_2
        # torch.save(checkpoint, args.save + f"{args.experiment_name}/best_model.pth")
        if best_val_mae_loss > all_loss:
            best_val_mae_loss = all_loss
            best_epoch = epoch 
            best_auc_loss = auc_val_loss
            best_cl_loss = cl_val_loss 
            best_vdss_loss = vdss_val_loss
            best_t1_2_loss = t1_2_val_loss
            best_r2_cl = r2_cl
            best_r2_vdss = r2_vdss
            best_r2_auc = r2_auc
            best_r2_t1_2 = r2_t1_2
            para_k2= para_function_cl_vdss_t1_2.item()
            para_k1= para_function_auc_cl.item()
            torch.save({'cl_model_state_dict':cl_model.state_dict(),
            'vdss_model_state_dict':vdss_model.state_dict(),
            't1_2_model_state_dict':t1_2_model.state_dict()
                }, args.save + f"{args.experiment_name}/best_model.pth")
            wait = 0
        else:
            wait +=1
        
        if wait > 15:
            break
        best_val_mae_loss =best_auc_loss + best_cl_loss + best_vdss_loss + best_t1_2_loss
        print('best_val_mae_loss', best_val_mae_loss)
        print('best_auc_loss:', best_auc_loss)
        print('best_cl_loss:', best_cl_loss)
        print('best_vdss_loss:', best_vdss_loss)
        print('best_t1_2_loss:', best_t1_2_loss)
        print(f'R² Scores -> CL: {best_r2_cl:.4f}, VDSS: {best_r2_vdss:.4f}, AUC: {best_r2_auc:.4f}, T1/2: {best_r2_t1_2:.4f}')
        print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
        print("train: %f val: %f " %(train_mae_loss, best_val_mae_loss))
        val_mse_loss_list.append(all_loss)
        wandb.log({
        "train_loss": train_mae_loss,
        "cl_val": cl_val_loss,
        "vdss_val": vdss_val_loss,
        "t1_2_val": t1_2_val_loss,
        },step=epoch)
        
    # cl_model.load_state_dict(torch.load(args.save + f"{args.experiment_name}/best_model.pth",map_location=device)['cl_model_state_dict'])
    # vdss_model.load_state_dict(torch.load(args.save + f"{args.experiment_name}/best_model.pth",map_location=device)['vdss_model_state_dict'])
    # t1_2_model.load_state_dict(torch.load(args.save + f"{args.experiment_name}/best_model.pth",map_location=device)['t1_2_model_state_dict'])
            
    new_data = pd.DataFrame({
                            'best_val_mae_loss':[best_val_mae_loss], 
                            'best_auc_loss':[best_auc_loss], 
                            'best epoch':[best_epoch],
                            'best_cl_loss':[best_cl_loss] , 
                            'best_vdss_loss':[best_vdss_loss], 
                            'best_t1_2_loss':[best_t1_2_loss],
                            "best_r2_cl": [r2_cl],
                            "best_r2_vdss": [r2_vdss],
                            "best_r2_auc": [r2_auc],
                            "best_r2_t1_2": [r2_t1_2]
                            }, index=[args.experiment_name])
        
    save_path = args.save+"result.csv"

    # 检查文件是否存在，存在则读取并追加新数据，否则新建
    if os.path.exists(save_path):
        existing_data = pd.read_csv(save_path, index_col='experiment_name')
        # 使用 pd.concat 合并数据
        updated_data = pd.concat([existing_data, new_data])
    else:
        updated_data = new_data

    # 保存到CSV，保留索引（experiment_name作为行标签）
    updated_data.to_csv(save_path, index_label='experiment_name')
    wandb.log({'best_val_mae_loss': best_val_mae_loss})
    wandb.finish()
if __name__ == "__main__":
    main()