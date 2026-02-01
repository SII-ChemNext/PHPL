import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loader_desc import MoleculeDataset
from torch_geometric.data import DataLoader
from itertools import cycle
import argparse
from tqdm import tqdm
import os

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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        """
        初始化MLP模型.
        
        参数:
        - input_dim (int): 输入特征的维度 (这里是217).
        - output_dim (int): 输出的维度 (对于回归任务，通常是1).
        """
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            # 第一个线性层 + 激活函数
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # 添加Dropout防止过拟合
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1), # 添加Dropout防止过拟合
            
            # 第二个线性层 + 激活函数
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 输出层
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        """定义前向传播."""
        x = x.view(-1,216)
        return self.network(x)

def main():
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
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for dataset loading') 
    parser.add_argument('--save', type=str, default='results/20250709/mlp/')
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # --- 2. 模型、损失函数和优化器 ---
    input_dimension = 216
    model = MLP(input_dim=input_dimension)
    model.to(device)
    # 损失函数 (对于回归任务，常用均方误差)
    criterion = nn.MSELoss()
    os.makedirs(args.save,exist_ok=True)
    # 优化器 (Adam是一个很好的通用选择)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    cl_train_dataset = MoleculeDataset("dataset_new_desc/cl_train", dataset="cl_train")
    cl_valid_dataset = MoleculeDataset("dataset_new_desc/cl_valid", dataset="cl_valid")

    vdss_train_dataset = MoleculeDataset("dataset_new_desc/vdss_train", dataset="vdss_train")
    vdss_valid_dataset = MoleculeDataset("dataset_new_desc/vdss_valid", dataset="vdss_valid")

    t1_2_train_dataset = MoleculeDataset("dataset_new_desc/t1_2_train", dataset="t1_2_train")
    t1_2_valid_dataset = MoleculeDataset("dataset_new_desc/t1_2_valid", dataset="t1_2_valid")

    ic_50_dataset = MoleculeDataset("dataset_new_desc/ic_50", dataset="ic_50", data_type="data_motif")

    ic_50_batchsize = int(args.batch_size / args.beta) if not args.beta == 0 else 1
    cl_train_loader = DataLoader(cl_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    vdss_train_loader = DataLoader(vdss_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    t1_2_train_loader = DataLoader(t1_2_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    ic_50_train_loader = DataLoader(ic_50_dataset, batch_size=ic_50_batchsize, shuffle=True, num_workers=args.num_workers,follow_batch=['desc'])
    cl_val_loader = DataLoader(cl_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    vdss_val_loader = DataLoader(vdss_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    t1_2_val_loader = DataLoader(t1_2_valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,follow_batch=['desc'])
    # ic_50_val_loader = DataLoader(ic_50_valid_dataset, batch_size=ic_50_batchsize, shuffle=False, num_workers=args.num_workers)
    # --- 3. 训练循环 ---
    test_dataset_name = "4_valid"
    test_dataset = MoleculeDataset("dataset_new_desc/"+test_dataset_name, dataset=test_dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        max_len = max(len(cl_train_loader), len(vdss_train_loader), len(t1_2_train_loader))
        
        cl_iter = cycle(iter(cl_train_loader))
        vdss_iter = cycle(iter(vdss_train_loader))
        t1_2_iter = cycle(iter(t1_2_train_loader))
        ic_50_iter = cycle(iter(ic_50_train_loader))
        count = 0
        for step in tqdm(range(max_len), desc="Training", mininterval=5):
            cl_batch = next(cl_iter).to(device)
            vdss_batch = next(vdss_iter).to(device)
            t1_2_batch = next(t1_2_iter).to(device)
            ic_50_batch = next(ic_50_iter).to(device)
            count += 1 
            cl_pred_log = model(cl_batch.desc)[:,[0]]
            vdss_pred_log = model(vdss_batch.desc)[:,[1]]
            t1_2_pred_log = model(t1_2_batch.desc)[:,[2]]
            pred_log_ic_50 = model(ic_50_batch.desc)
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
            
            optimizer.zero_grad()  
            loss.backward()    
            optimizer.step()


        # --- 4. 模型评估 ---
        model.eval() # 设置模型为评估模式
        loss_list = []
        best_val_mae_loss = float('inf')
        with torch.no_grad(): # 在评估阶段不计算梯度
            for i,loader in enumerate([cl_val_loader,vdss_val_loader,t1_2_val_loader]):
                y_scores = []
                y_list = []
                for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred_log = model(batch.desc)[:,[i]]
                        y = batch.y.reshape(batch.y.size(0), 1)
                        y_scores.append(pred_log)
                        y_list.append(y)

                y_scores = torch.cat(y_scores, dim=0)
                y_list = torch.cat(y_list, dim=0)
                loss = criterion(y_scores, y_list).cpu().detach().item()
                loss_list.append(loss)
        cl_loss,vdss_loss,t1_2_loss = loss_list
        print('cl_loss:',cl_loss,'vdss_loss:',vdss_loss,'t1_2_loss:',t1_2_loss)
        val_mae_loss = cl_loss+vdss_loss+t1_2_loss
        if best_val_mae_loss > val_mae_loss:
            best_val_mae_loss = val_mae_loss
            torch.save(model.state_dict(),args.save+'best_model.pth')
        
        with torch.no_grad():
            y_scores_cl = []
            y_scores_vdss = []
            y_scores_t1_2 = []

            y_cl = []
            y_vdss = []
            y_t1_2 = []
            for batch in test_loader:
                batch.to(device)
                pred_log= model(batch.desc)
                cl_pred_log = pred_log[:,[0]]
                vdss_pred_log = pred_log[:,[1]]
                t1_2_pred_log = pred_log[:,[2]]
                
                cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
                vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
                t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
                
                y_scores_cl.append(cl_pred_log)
                y_scores_vdss.append(vdss_pred_log)
                y_scores_t1_2.append(t1_2_pred_log)
                
                y_cl.append(cl_y)
                y_vdss.append(vdss_y)
                y_t1_2.append(t1_2_y)

        y_scores_cl = torch.cat(y_scores_cl, dim=0)
        y_scores_vdss = torch.cat(y_scores_vdss, dim=0)
        y_scores_t1_2 = torch.cat(y_scores_t1_2, dim=0)

        y_cl = torch.cat(y_cl, dim=0)
        y_vdss = torch.cat(y_vdss, dim=0)
        y_t1_2 = torch.cat(y_t1_2, dim=0)
        
        loss_cl = criterion(y_scores_cl, y_cl).cpu().detach().item()
        loss_vdss = criterion(y_scores_vdss, y_vdss).cpu().detach().item()
        loss_t1_2 = criterion(y_scores_t1_2, y_t1_2).cpu().detach().item()
        loss_all = (loss_cl + loss_vdss + loss_t1_2 )
        
        r2_cl = r2_score(y_cl,y_scores_cl)
        r2_vdss = r2_score(y_vdss,y_scores_vdss)
        r2_t1_2 = r2_score(y_t1_2,y_scores_t1_2)
    
        print('test_cl:',loss_cl,'test_vdss:',loss_vdss,'test_t1_2:',loss_t1_2)
                
        
        
if __name__ == '__main__':
    main()