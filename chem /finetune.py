import argparse
import copy
import pickle

from loader_desc import MoleculeDataset
from torch_geometric.data import DataLoader
from torch.utils.data import Subset,random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_graphpred
from sklearn.metrics import roc_auc_score, accuracy_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
import sys
from util import add_noise_to_partial_labels

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas 
import math 

import matplotlib.pyplot as plt 

from model import MLPregression 
from sklearn.manifold import TSNE 
from torch.utils.tensorboard import SummaryWriter

criterion = nn.L1Loss()

K_1 = 16846
K_2 = 17.71 # 18

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

def train(args, model,  para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, optimizer, epoch,temperature=1.0,num_samples = 5): # para_contrastive_loss,
    model.train()

    temperature=args.temperature
    total_loss = 0 
    count = 0 
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        pred_log= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,batch.desc)
        # pred_log= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        cl_pred_log = pred_log[:,[0]]
        vdss_pred_log = pred_log[:,[1]]
        t1_2_pred_log = pred_log[:,[2]]

        k_1 = para_function_auc_cl * K_1
        k_2 = para_function_cl_vdss_t1_2 * K_2
        auc_pred_log = torch.log(k_1)-cl_pred_log
        
        auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
        cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
        vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
        t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
        
        auc_y = add_noise_to_partial_labels(auc_y,0.2,args.portion)
        # cl_y = add_noise_to_partial_labels(cl_y,0.2,args.portion)
        # vdss_y = add_noise_to_partial_labels(vdss_y,0.8,args.portion)
        # t1_2_y = add_noise_to_partial_labels(t1_2_y,0.8,args.portion)
        
        loss_auc = criterion(auc_pred_log, auc_y)
        loss_cl = criterion(cl_pred_log, cl_y)
        loss_vdss = criterion(vdss_pred_log, vdss_y)
        loss_t1_2 = criterion(t1_2_pred_log, t1_2_y)
        
        loss_k2 = criterion(
        cl_pred_log + t1_2_pred_log,
        vdss_pred_log + torch.log(k_2)
        )
 
        loss = loss_auc + loss_cl + loss_vdss + loss_t1_2 +loss_k2 *args.temperature
        # loss = loss_cl + loss_vdss + loss_t1_2 +loss_k2 *args.temperature
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    total_loss += loss
        
    return (total_loss)/count
   

def eval(args, model, para_function_auc_cl,  device, loader, epoch): # para_contrastive_loss,
    model.eval()
     
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
            
            pred_log= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,batch.desc)
            # pred_log= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            cl_pred_log = pred_log[:,[0]]
            vdss_pred_log = pred_log[:,[1]]
            t1_2_pred_log = pred_log[:,[2]]


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
    loss_all = (loss_auc + loss_cl + loss_vdss + loss_t1_2 )
    
    r2_auc = r2_score(y_auc,y_scores_auc)
    r2_cl = r2_score(y_cl,y_scores_cl)
    r2_vdss = r2_score(y_vdss,y_scores_vdss)
    r2_t1_2 = r2_score(y_t1_2,y_scores_t1_2)
   
    return loss_auc, loss_cl, loss_vdss, loss_t1_2, loss_all, r2_auc, r2_cl, r2_vdss, r2_t1_2
           
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, # 60 # 1500
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
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
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
    
    parser.add_argument('--seed', type=int, default=7, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=8, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=True)
    parser.add_argument('--experiment_name', type=str, default="graphmae")
    parser.add_argument('--portion', type=float, default=0.0)
    
    
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    
    ## add the multi-task alpha 
    parser.add_argument('--t1_2_alpha', type=float, default=1.)
    parser.add_argument('--k3_alphla', type=float, default=1.)    
    
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
    
    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device), requires_grad=True)
    para_function_cl_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    
    para_loss.data.fill_(0.5)
    para_function_cl_vdss_t1_2.data.fill_(1.)
    para_function_auc_cl.data.fill_(1.)
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    # args.seed = args.runseed 
    args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bz'+'_'+str(args.batch_size)+'_'+'drop'+'_'+str(args.dropout_ratio)+'_'+'temp'+'_'+str(args.temperature)+'_'+'port'+'_'+str(args.portion)
    save_dir = os.path.join(args.save, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)  
    os.makedirs(f'runs/0423/final_together/{args.experiment_name}', exist_ok=True)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset == "4":
        num_tasks = 3
        train_dataset_name = "4_train"
        test_dataset_name = "4_valid"
    else:
        raise ValueError("Invalid dataset name.")

    ## set up pk dataset 
    # train_dataset = MoleculeDataset("dataset_new_desc/"+train_dataset_name, dataset=train_dataset_name, motif_list=motif_list)
    # valid_dataset = MoleculeDataset("dataset_new_desc/"+valid_dataset_name, dataset=valid_dataset_name, motif_list=motif_list)
    # len_train = len(train_dataset)
    # sample_size = int(args.portion*len_train)
    # all_indices = list(range(len_train))
    # sampled_indices = np.random.choice(all_indices, size=sample_size, replace=False)
    # subset_dataset = Subset(train_dataset, sampled_indices)

    # print(f"创建的 Subset 数据集大小: {len(subset_dataset)}")
    # train_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)   
    # val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    full_train_dataset = MoleculeDataset("dataset_new_desc/" + train_dataset_name, dataset=train_dataset_name, motif_list=motif_list)
    test_dataset = MoleculeDataset("dataset_new_desc/" + test_dataset_name, dataset=test_dataset_name, motif_list=motif_list)

    # Split the original training data into a new training set and a validation set (7:1 ratio)
    num_full_train = len(full_train_dataset)
    sample_size = int(num_full_train)   #数据量实验在这里乘上portion
    all_indices = list(range(num_full_train))
    # shuffled_indices = np.random.permutation(all_indices)
    # sampled_indices = np.random.choice(all_indices, num_full_train, replace=False)
    shuffled_indices = np.random.permutation(all_indices)
    sampled_indices = shuffled_indices[:sample_size]
    subset_dataset = Subset(full_train_dataset, sampled_indices)
    val_size = sample_size // 8
    train_size = sample_size - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size], generator=generator)  #如果不用portion就改回full_train_dataset
    
    print(f"Full original training size: {num_full_train}")
    # print(f"New training split size: {len(subset_dataset)}")
    print(f"New validation split size: {len(val_dataset)}")
    print(f"Final sampled training size (portion={args.portion}): {len(train_dataset)}")
    print(f"Final test size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    desc_emb = 216
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type,desc_emb=desc_emb)
    # model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
    if not args.filename == "":
        print("load pretrained model from:", args.filename)
        checkpoint = torch.load(args.filename, map_location=device)
        model.load_state_dict(checkpoint,strict=True)
               
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    model_param_group.append({"params": para_function_cl_vdss_t1_2,"lr":args.lr*0.05})
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
    
    best_val_loss=float('inf')
    best_true_val_mse_loss = float('inf')
    
    wait = 0
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_mae_loss = train(args, model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
        train_mse_loss_list.append(train_mae_loss)
        print(para_function_cl_vdss_t1_2, para_function_auc_cl,)
        auc_val_loss, cl_val_loss, vdss_val_loss, t1_2_val_loss, val_all, r2_val_auc, r2_val_cl, r2_val_vdss, r2_val_t1_2=eval(args, model, para_function_auc_cl, device, val_loader, epoch) # para_contrastive_loss,
        test_auc, test_cl, test_vdss, test_t1_2, test_all, r2_test_auc, r2_test_cl, r2_test_vdss, r2_test_t1_2 = eval(args, model, para_function_auc_cl, device, test_loader, epoch)
    #     writer.add_scalar('Loss/train', train_mae_loss, epoch)
    #     writer.add_scalar('Loss/cl', cl_val_loss, epoch)
    #     writer.add_scalar('Loss/vdss', vdss_val_loss, epoch)
    #     writer.add_scalar('Loss/t1_2', t1_2_val_loss, epoch)
    #     writer.add_scalar('Loss/all', all_loss, epoch)
            
    #     if scheduler is not None:
    #         scheduler.step(metrics=all_loss)
            
    #     if best_val_mse_loss > all_loss:
    #         best_val_mse_loss = all_loss
    #         best_epoch = epoch 
    #         best_auc_loss = auc_val_loss
    #         best_cl_loss = cl_val_loss 
    #         best_vdss_loss = vdss_val_loss
    #         best_t1_2_loss = t1_2_val_loss
    #         best_r2_cl = r2_cl
    #         best_r2_vdss = r2_vdss
    #         best_r2_auc = r2_auc
    #         best_r2_t1_2 = r2_t1_2
    #         para_k2= para_function_cl_vdss_t1_2.item()
    #         para_k1= para_function_auc_cl.item()
    #         torch.save({'model_state_dict':model.state_dict(),
    #                     'k1':para_function_auc_cl,
    #                     'k2':para_function_cl_vdss_t1_2.data}, args.save + f"{args.experiment_name}/best_model.pth")
    #         wait = 0
    #     else:
    #         wait +=1
        
    #     if wait > 21:
    #         break
            
    #     print('best epoch:', best_epoch)
    #     print('best_val_mse_loss', best_val_mse_loss)
    #     print('best_auc_loss:', best_auc_loss)
    #     print('best_cl_loss:', best_cl_loss)
    #     print('best_vdss_loss:', best_vdss_loss)
    #     print('best_t1_2_loss:', best_t1_2_loss)
    #     print('para_k1:',para_k1)
    #     print('para_k2:',para_k2)        
    #     print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
    #     print("train: %f val: %f " %(train_mae_loss, best_val_mse_loss))
    #     val_mse_loss_list.append(all_loss)
        
    # writer.close()
    # new_data = pd.DataFrame({'best epoch':[best_epoch], 
    #                             'best_val_mse_loss':[best_val_mse_loss], 
    #                             'best_auc_loss':[best_auc_loss], 
    #                             'best_cl_loss':[best_cl_loss] , 
    #                             'best_vdss_loss':[best_vdss_loss], 
    #                             'best_t1_2_loss':[best_t1_2_loss],
    #                             'para_k1:':[para_k1],
    #                             "best_r2_auc": [best_r2_auc],
    #                             "best_r2_cl": [best_r2_cl],
    #                             "best_r2_vdss": [best_r2_vdss],
    #                             "best_r2_t1_2": [best_r2_t1_2]
    #                             }, index=[args.experiment_name])
        
        writer.add_scalar('Loss/train_all', train_mae_loss, epoch)
        writer.add_scalar('Loss/val_all', val_all, epoch)
        writer.add_scalar('Loss/test_all', test_all, epoch)
        writer.add_scalar('R2/val_auc', r2_val_auc, epoch)
        writer.add_scalar('R2/test_auc', r2_test_auc, epoch)


        if scheduler:
            scheduler.step(metrics=val_all)
            
        # --- MODIFIED: Early stopping based on VALIDATION loss ---
        if val_all < best_val_loss:
            best_val_loss = val_all
            best_epoch = epoch
            
            # Save the test metrics from this best epoch
            final_test_results = {
                'best_epoch': best_epoch,
                # 'best_val_loss_all': best_val_loss,
                # 'test_loss_all': test_all,
                'test_loss_auc': test_auc,
                # 'test_loss_cl': test_cl,
                # 'test_loss_vdss': test_vdss,
                # 'test_loss_t1_2': test_t1_2,
                # 'test_r2_auc': r2_test_auc,
                # 'test_r2_cl': r2_test_cl,
                # 'test_r2_vdss': r2_test_vdss,
                # 'test_r2_t1_2': r2_test_t1_2,
                # 'para_k1': para_function_auc_cl.item(),
                # 'para_k2': para_function_cl_vdss_t1_2.item(),
            }
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'k1': para_function_auc_cl,
                'k2': para_function_cl_vdss_t1_2.data
            }, os.path.join(save_dir, "best_model.pth"))
            
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Saving model and test scores.")
            wait = 0
        else:
            wait += 1
        
        if wait > 21:
            print("Early stopping triggered.")
            break
            
        print(f"Current best val loss: {best_val_loss:.4f} (at epoch {best_epoch})")
        print(f"  Train Loss: {train_mae_loss:.4f} | Val Loss: {val_all:.4f} | Test Loss: {test_all:.4f}")

    writer.close()
    
    # --- MODIFIED: Save the final test results from the best validation epoch ---
    print("\n--- Final Results (from best validation epoch) ---")
    print(pd.Series(final_test_results))
    
    # Convert dictionary to DataFrame for saving
    new_data = pd.DataFrame([final_test_results], index=[args.experiment_name])

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