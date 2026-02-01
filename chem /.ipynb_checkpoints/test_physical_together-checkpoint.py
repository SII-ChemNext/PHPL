import argparse
import copy
import pickle

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_Bayes_together
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

criterion = nn.L1Loss()

K_1 = 16846
K_2 = 17.71 # 18

#高斯似然函数（重构函数为mse）
def nll_gaussian(loss,std):
    nll = 0.5 * torch.log(2 * torch.pi * std**2) + \
        0.5 * (loss**2) / (std**2 + 1e-8)
    return(nll)

#拉普拉斯似然函数（重构函数为mae）
def nll_laplace(loss, b):
    return torch.log(2 * b) + torch.abs(loss) / (b + 1e-8)

def train(args, model,  para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, optimizer, epoch,temperature=1.0,num_samples = 5,beta=1.0): # para_contrastive_loss,
    model.train()

    temperature=args.temperature
    total_loss = 0 
    count = 0 
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        cl_mean,cl_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 0, False)
        vdss_mean,vdss_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 1, False)
        t1_2_mean,t1_2_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 2 ,False)

        k_1 = para_function_auc_cl * K_1
        k_3 = para_function_cl_vdss_t1_2 * K_2
        
        auc_pred_log = torch.log(k_1)-cl_mean
        
        auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
        cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
        vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
        t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
        
        loss_auc = criterion(auc_pred_log, auc_y)
        #使用mse和高斯似然
        loss_cl = F.mse_loss(cl_mean, cl_y, reduction='none')
        loss_vdss = F.mse_loss(vdss_mean, vdss_y, reduction='none')
        loss_t1_2 = F.mse_loss(t1_2_mean, t1_2_y, reduction='none')
        
        nll_cl =nll_gaussian(loss_cl,cl_std)
        nll_vdss =nll_gaussian(loss_vdss,vdss_std)
        nll_t1_2 =nll_gaussian(loss_t1_2,t1_2_std)  
        
        # #使用mae和拉普拉斯似然
        # loss_cl = F.l1_loss(cl_mean, cl_y, reduction='none')
        # loss_vdss = F.l1_loss(vdss_mean, vdss_y, reduction='none')
        # loss_t1_2 = F.l1_loss(t1_2_mean, t1_2_y, reduction='none')
        
        # nll_cl =nll_laplace(loss_cl,cl_std)
        # nll_vdss =nll_laplace(loss_vdss,vdss_std)
        # nll_t1_2 =nll_laplace(loss_t1_2,t1_2_std)  
        
        # #elbo方法   计算模型kl     
        # kl_loss_cl = model.bayes[0].kl_loss()
        # kl_loss_vdss = model.bayes[1].kl_loss()
        # kl_loss_t1_2 = model.bayes[2].kl_loss()
    
        # cl_loss = (nll_cl+kl_loss_cl*beta).mean()
        # vdss_loss = (nll_vdss+kl_loss_vdss*beta).mean()
        # t1_2_loss = (nll_t1_2+kl_loss_t1_2*beta).mean()        
        
        #nll加权方法
        weights_cl = torch.softmax(nll_cl.detach() / temperature, dim=0) * nll_cl.size(0)
        weights_vdss = torch.softmax(nll_vdss.detach() / temperature, dim=0) * nll_vdss.size(0)
        weights_t1_2 = torch.softmax(nll_t1_2.detach() / temperature, dim=0) * nll_t1_2.size(0)     
        
        cl_loss = (weights_cl * loss_cl).mean()
        vdss_loss = (weights_vdss * loss_vdss).mean()
        t1_2_loss = (weights_t1_2 * loss_t1_2).mean()
        
        loss_k3 = criterion(cl_mean+t1_2_mean, vdss_mean+torch.log(k_3))
        loss =  loss_auc + cl_loss + vdss_loss + t1_2_loss + loss_k3

        total_loss += loss.item()

        # loss = (loss_auc / loss_auc.detach()
        #         + cl_loss / cl_loss.detach()
        #         + vdss_loss / vdss_loss.detach()
        #         + t1_2_loss / t1_2_loss.detach()
        #         + loss_k3 / loss_k3.detach()
        #         )
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (total_loss)/count
   

def eval(args, model, para_function_auc_cl,  device, loader, epoch,num_samples=50): # para_contrastive_loss,
    model.eval()
     
    y_scores_cl = []
    y_scores_vdss = []
    y_scores_t1_2 = []
    y_scores_auc = []
    
    y_cl = []
    y_vdss = []
    y_t1_2 = []
    y_auc = []
    
    std_cl = []
    std_vdss = []
    std_t1_2 = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", file=sys.stdout)):
        batch = batch.to(device)

        with torch.no_grad():
            
            cl_mean,cl_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 0 ,False)
            vdss_mean,vdss_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 1 ,False)
            t1_2_mean,t1_2_std= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, num_samples, 2 ,False)

            k_1 = para_function_auc_cl * K_1
            
            auc_pred_log = torch.log(k_1)-cl_mean
            
            auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
            cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
            vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
            t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
            
            y_scores_auc.append(auc_pred_log)
            y_scores_cl.append(cl_mean)
            y_scores_vdss.append(vdss_mean)
            y_scores_t1_2.append(t1_2_mean)
            
            y_auc.append(auc_y)
            y_cl.append(cl_y)
            y_vdss.append(vdss_y)
            y_t1_2.append(t1_2_y)
            
            # 添加 std
            std_cl.append(cl_std)
            std_vdss.append(vdss_std)
            std_t1_2.append(t1_2_std)


    std_cl = torch.cat(std_cl, dim=0)
    std_vdss = torch.cat(std_vdss, dim=0)
    std_t1_2 = torch.cat(std_t1_2, dim=0)
            
    y_scores_auc = torch.cat(y_scores_auc, dim=0)
    y_scores_cl = torch.cat(y_scores_cl, dim=0)
    y_scores_vdss = torch.cat(y_scores_vdss, dim=0)
    y_scores_t1_2 = torch.cat(y_scores_t1_2, dim=0)
    
    y_auc = torch.cat(y_auc, dim=0)
    y_cl = torch.cat(y_cl, dim=0)
    y_vdss = torch.cat(y_vdss, dim=0)
    y_t1_2 = torch.cat(y_t1_2, dim=0)
    
    loss_auc = criterion(y_scores_auc, y_auc).cpu().detach().item()
    
    # mae+拉普拉斯
    # loss_cl = F.l1_loss(y_scores_cl, y_cl,reduction='none').detach()
    # loss_vdss = F.l1_loss(y_scores_vdss, y_vdss,reduction='none').detach()
    # loss_t1_2 = F.l1_loss(y_scores_t1_2, y_t1_2,reduction='none').detach()
    
    # nll_cl =nll_laplace(loss_cl,std_cl)
    # nll_vdss =nll_laplace(loss_vdss,std_vdss)
    # nll_t1_2 =nll_laplace(loss_t1_2,std_t1_2)    
    
    # loss_cl= loss_cl.mean().cpu().item()
    # loss_vdss= loss_vdss.mean().cpu().item()
    # loss_t1_2= loss_t1_2.mean().cpu().item()
    
    #mse+高斯
    loss_cl = F.mse_loss(cl_mean, cl_y, reduction='none')
    loss_vdss = F.mse_loss(vdss_mean, vdss_y, reduction='none')
    loss_t1_2 = F.mse_loss(t1_2_mean, t1_2_y, reduction='none')
    
    nll_cl =nll_gaussian(loss_cl,cl_std)
    nll_vdss =nll_gaussian(loss_vdss,vdss_std)
    nll_t1_2 =nll_gaussian(loss_t1_2,t1_2_std) 
     
    total_nll_cl = nll_cl.sum().item()
    total_nll_vdss = nll_vdss.sum().item()
    total_nll_t1_2 = nll_t1_2.sum().item()
    
    #mae评估
    loss_cl = criterion(cl_mean, cl_y).cpu().detach().item()
    loss_vdss = criterion(vdss_mean, vdss_y).cpu().detach().item()
    loss_t1_2 = criterion(t1_2_mean, t1_2_y).cpu().detach().item()
    loss_all = (loss_auc + loss_cl + loss_vdss + loss_t1_2 )
   
    return loss_auc, loss_cl, loss_vdss, loss_t1_2, loss_all, total_nll_cl, total_nll_vdss, total_nll_t1_2 
           
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=60, # 60 # 1500
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
    parser.add_argument('--temperature', type=float, default=1.)
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--input_cl_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='input filename')
    
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
    parser.add_argument('--cl_model_scale', type=float, default=1.)
    parser.add_argument('--vdss_model_scale', type=float, default=1.)
    parser.add_argument('--t1_2_model_scale', type=float, default=1.)
    
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()
    
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
    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bz'+'_'+str(args.batch_size)+'_'+'drop'+'_'+str(args.dropout_ratio)+'_'+'temp'+'_'+str(args.temperature)
    os.makedirs(args.save+args.experiment_name,exist_ok=True)

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
    
    model = GNN_Bayes_together(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)

    if not args.filename == "":
        print("load pretrained model from:", args.filename)
        checkpoint = torch.load(args.filename, map_location=device)['model_atom_state_dict']
        filtered_state_dict = {k: v for k, v in checkpoint.items() if not "graph_pred_linear" in k}
        model.load_state_dict(filtered_state_dict, strict=False)
               
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    model_param_group.append({"params": para_function_cl_vdss_t1_2,"lr":args.lr*0.1})
    model_param_group.append({'params': para_function_auc_cl,"lr":args.lr*0.05})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        scheduler = None
        
    epoch_list = np.arange(0, args.epochs, 1)
    print('epoch_list_len',len(epoch_list))
    train_mse_loss_list = []
    val_mse_loss_list = []
    
    best_val_mse_loss=float('inf')
    best_true_val_mse_loss = float('inf')
    
    wait = 0
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_mae_loss = train(args, model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
        train_mse_loss_list.append(train_mae_loss)
        print(para_function_cl_vdss_t1_2, para_function_auc_cl,)
        auc_val_loss, cl_val_loss, vdss_val_loss, t1_2_val_loss, all_loss, total_nll_cl, total_nll_vdss, total_nll_t1_2 =eval(args, model, para_function_auc_cl, device, val_loader, epoch) # para_contrastive_loss,
        
        if epoch % 10 == 0 or epoch == 1 or epoch == 2:
            torch.save({'model_state_dict':model.state_dict(),
                        }, args.save + f"{args.experiment_name}/{epoch}.pth")
            
        if scheduler is not None:
            scheduler.step(metrics=all_loss)
            
        if best_val_mse_loss > all_loss:
            best_val_mse_loss = all_loss
            best_epoch = epoch 
            best_auc_loss = auc_val_loss
            best_cl_loss = cl_val_loss 
            best_vdss_loss = vdss_val_loss
            best_t1_2_loss = t1_2_val_loss
            para_k2= para_function_cl_vdss_t1_2.item()
            para_k1= para_function_auc_cl.item()
            torch.save({'model_state_dict':model.state_dict(),
                }, args.save + f"{args.experiment_name}/best_model.pth")
            wait = 0
        else:
            wait +=1
        
        if wait > 21:
            break
            
        print('best epoch:', best_epoch)
        print('best_val_mse_loss', best_val_mse_loss)
        print('best_auc_loss:', best_auc_loss)
        print('best_cl_loss:', best_cl_loss)
        print('best_vdss_loss:', best_vdss_loss)
        print('best_t1_2_loss:', best_t1_2_loss)
        print(f"Total NLL CL: {total_nll_cl:.6f}, \
                Total NLL VDSS: {total_nll_vdss:.6f}, \
                Total NLL T1_2: {total_nll_t1_2:.6f}")
        print('para_k1:',para_k1)
        print('para_k2:',para_k2)        
        print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
        print("train: %f val: %f " %(train_mae_loss, best_val_mse_loss))
        val_mse_loss_list.append(all_loss)
    
    new_data = pd.DataFrame({'best epoch':[best_epoch], 
                                'best_val_mse_loss':[best_val_mse_loss], 
                                'best_auc_loss':[best_auc_loss], 
                                'best_cl_loss':[best_cl_loss] , 
                                'best_vdss_loss':[best_vdss_loss], 
                                'best_t1_2_loss':[best_t1_2_loss],
                                'para_k1:':[para_k1],
                                'para_k2:':[para_k2],
                                'Total NLL CL': [total_nll_cl], 
                                'Total NLL VDSS': [total_nll_vdss], 
                                'Total NLL T1_2': [total_nll_t1_2]
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

if __name__ == "__main__":
    main()