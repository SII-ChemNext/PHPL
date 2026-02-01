import argparse
from functools import partial
import os
import pickle 
import sys

from loader_desc import MoleculeDataset, atom_to_motif ,mol_to_graph_data_obj_simple
from dataloader import DataLoaderMaskingPred #, DataListLoader
from torch.utils.data import DataLoader,random_split
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNNDecoder
from loss import ContrastiveLoss

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split,  random_scaffold_split
import pandas as pd

from util import MaskAtom

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from multiprocessing import Pool, cpu_count

# from tensorboardX import SummaryWriter
import random
import matplotlib.pyplot as plt
# import timeit
# import time 

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)   #内积等于余弦相似度（归一化成单位向量了）

    loss = loss.mean()
    return loss

def get_atom_level_mol_emb(batch, pred_probs_atom, batch_size):
    mol_rep = []
    count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        per_mol_rep = torch.sum(pred_probs_atom[count:count+num])
        mol_rep.append(per_mol_rep.unsqueeze(0))

    mol_representation = torch.cat(mol_rep, dim=0)
    
    # print(mol_representation.size())
    return mol_representation
  
  
def get_motif_level_mol_emb(batch, atoms_in_motif, batch_size):
    mol_rep = []
    count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        # motif_num = torch.unique(batch.motifs[count:count+num]).size(0)
        per_mol_rep = torch.sum(atoms_in_motif[count:count+num], dim=0)
        mol_rep.append(per_mol_rep.unsqueeze(0))

    mol_representation = torch.cat(mol_rep, dim=0)
    # print(mol_representation.size())
    return mol_representation

    
def get_per_motif_emb(batch, motifs_node_rep, batch_size):
    motifs_rep = []
    count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        motif_index = batch.x[count:count+num][:,0]
        all_node_rep = motifs_node_rep[count:count+num]
        for j in torch.unique(motif_index):
            per_motif_index = torch.nonzero(motif_index==j)
            motif_index = per_motif_index.squeeze(-1)
            pre_motif_emb = torch.sum(torch.index_select(all_node_rep, dim=0, index=motif_index), dim=0).unsqueeze(0)
            motifs_rep.append(pre_motif_emb)
    
    motifs_rep = torch.cat(motifs_rep, dim=0)
    return motifs_rep

def get_motif(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smile = []
    try:
        motif = atom_to_motif(mol).motif  # 获取 motif 列表
        smile += (m for m in motif)
    except:
        return smiles
    return smile  # 返回作为集合，去重


def get_motif_list_parallel(smiles_list):
    num_workers = cpu_count()  # 使用所有可用 CPU 核心
    with Pool(num_workers) as pool:
        result_list = pool.map(get_motif, smiles_list)  # 对每个 smiles 并行调用 get_motif
    # 合并所有结果
    motif_list = set().union(*result_list)  # 将每个子进程返回的 set 合并为一个唯一的集合
    return list(motif_list)  # 将结果转换为列表
    

def train_mae(args, model_list, loader_atom,  loader_motif, optimizer_list, device, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    gnn_model_atom, gnn_model_motif, atom_pred_decoder, motif_pres_decoder, bond_pred_decoder = model_list
    optimizer_gnn_model_atom, optimizer_gnn_model_motif, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_motifs = optimizer_list
    
    gnn_model_atom.train()
    gnn_model_motif.train()
    atom_pred_decoder.train()
    motif_pres_decoder.train()

    if bond_pred_decoder is not None:
        bond_pred_decoder.train()

    loss_accum = 0
    
    epoch_iter = tqdm(loader_motif , desc="Iteration")

    for step,batch1  in enumerate(epoch_iter):
        batch1 = batch1.to(device)
        node_rep = gnn_model_atom(batch1.x, batch1.edge_index, batch1.edge_attr)  #带mask的atom rep
            
        node_attr_label = batch1.node_attr_label
        masked_node_indices = batch1.masked_atom_indices
        pred_node = atom_pred_decoder(node_rep, batch1.edge_index, batch1.edge_attr, masked_node_indices)
        # print(node_attr_label.size(), masked_node_indices.size())
        ## loss for nodes
        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], batch1.mask_node_label[:,0])
        

        optimizer_gnn_model_atom.zero_grad()
        optimizer_gnn_model_motif.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        optimizer_dec_pred_motifs.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()

        optimizer_gnn_model_atom.step()
        optimizer_gnn_model_motif.step()
        optimizer_dec_pred_atoms.step()
        optimizer_dec_pred_motifs.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum/(step+1)

def eval_mae(args, model_list, loader_motif, device, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    gnn_model_atom, gnn_model_motif, atom_pred_decoder, motif_pres_decoder, bond_pred_decoder = model_list

    gnn_model_atom.eval()
    gnn_model_motif.eval()
    atom_pred_decoder.eval()
    motif_pres_decoder.eval()

    if bond_pred_decoder is not None:
        bond_pred_decoder.eval()

    loss_accum = 0
    epoch_iter = tqdm(loader_motif, desc="Evaluation")

    for step, batch1 in enumerate(epoch_iter):
        batch1 = batch1.to(device)
        node_rep = gnn_model_atom(batch1.x, batch1.edge_index, batch1.edge_attr)
        
        node_attr_label = batch1.node_attr_label
        masked_node_indices = batch1.masked_atom_indices
        pred_node = atom_pred_decoder(node_rep, batch1.edge_index, batch1.edge_attr, masked_node_indices)

        ## loss for nodes
        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], batch1.mask_node_label[:, 0])

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"eval_loss: {loss.item():.4f}")

    return loss_accum / (step + 1)

def main():
    
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--atom_mask_rate', type=float, default=0.7,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--motif_mask_rate', type=float, default=0.7,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'ic_50', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = 'checkpoint/20250801/mask_0.7/', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    
    parser.add_argument("--loss_computer", type=str, default="cosine")
    parser.add_argument("--tau", type=float, default=0.7)
    
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.atom_mask_rate, args.mask_edge))

    # input_df=pd.read_csv('dataset_reg/ic_50/raw/ic_50.csv')
    # smiles_list = list(input_df.smiles)
    # global motif_list
    # motif_list = get_motif_list(smiles_list)
    # 指定路径
    global motif_list
    motif_list_path = 'dataset_reg/motif_list.pkl'

    # 读取数据
    input_df = pd.read_csv('dataset_reg/ic_50/raw/ic_50.csv')
    smiles_list = list(input_df.smiles)

    # 检查路径文件是否存在
    if os.path.exists(motif_list_path):
        print(f"文件 {motif_list_path} 存在，从文件中加载 motif_list...")
        with open(motif_list_path, 'rb') as f:
            motif_list = pickle.load(f)
            print(len(motif_list))
    else:
        print(f"文件 {motif_list_path} 不存在，运行 get_motif_list...")
        motif_list = get_motif_list_parallel(smiles_list)
        # 保存到指定路径
        os.makedirs(os.path.dirname(motif_list_path), exist_ok=True)  # 确保输出文件夹存在
        with open(motif_list_path, 'wb') as f:
            pickle.dump(motif_list, f)
        print(f"motif_list 已保存到 {motif_list_path}")
    
    os.makedirs(args.output_model_file, exist_ok=True)
    
    dataset_name = args.dataset
    dataset_atom = MoleculeDataset("dataset_reg/" + dataset_name, dataset=dataset_name, data_type="data_atom",motif_list=motif_list)
    dataset_motif = MoleculeDataset("dataset_new_desc/" + dataset_name, dataset=dataset_name, data_type="data_motif",motif_list=motif_list)
    total_len = len(dataset_motif)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    train_dataset, val_dataset = random_split(dataset_motif, [train_len, val_len])

    #loader为一个dataloader对象，长度为1，传入的dataset应该是moleculedataset类，在调用dataloader时会调用dataset的get函数
    loader_atom = DataLoaderMaskingPred(dataset_atom, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.atom_mask_rate, mask_edge=args.mask_edge, num_atom_type=119)  
    loader_motif = DataLoaderMaskingPred(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.motif_mask_rate, mask_edge=args.mask_edge, num_atom_type=len(motif_list)+119)  
    loader_val = DataLoaderMaskingPred(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.motif_mask_rate, mask_edge=args.mask_edge, num_atom_type=len(motif_list)+119)  
    # set up models, one for pre-training and one for context embeddings
    gnn_model_atom = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio).to(device)
    gnn_model_motif = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio).to(device)
    ## set the atom-level and motif-level contrastive loss 
    contrastive_loss = ContrastiveLoss(args.loss_computer, args.tau, args)  #对比学习的loss函数，默认为
    
    
    if args.input_model_file is not None and args.input_model_file != "":
        checkpoint = torch.load(args.input_model_file, map_location=torch.device('cuda'))
        gnn_model_atom.load_state_dict(torch.load(args.input_model_file)["model_atom_state_dict"])
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119 # + 3 
    atom_pred_decoder = GNNDecoder(args.emb_dim, len(motif_list)+119, JK=args.JK, gnn_type=args.gnn_type).to(device)
    motif_pres_decoder = GNNDecoder(args.emb_dim, len(motif_list), JK=args.JK, gnn_type=args.gnn_type).to(device)
    
    if args.mask_edge:   #如果需要edge的值
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [gnn_model_atom, gnn_model_motif,  atom_pred_decoder, motif_pres_decoder, bond_pred_decoder] 

    # set up optimizers
    optimizer_gnn_model_atom = optim.Adam(gnn_model_atom.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_gnn_model_motif = optim.Adam(gnn_model_motif.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_motifs = optim.Adam(motif_pres_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:   #是否动态调整退火率
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_gnn_model_atom = torch.optim.lr_scheduler.LambdaLR(optimizer_gnn_model_atom, lr_lambda=scheduler)
        scheduler_gnn_model_motif = torch.optim.lr_scheduler.LambdaLR(optimizer_gnn_model_motif, lr_lambda=scheduler)
        scheduler_atom_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_motif_pred = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_motifs, lr_lambda=scheduler)
        scheduler_list = [scheduler_gnn_model_atom, scheduler_gnn_model_motif,scheduler_atom_dec, scheduler_motif_pred]
    else:
        scheduler_gnn_model_atom = None
        scheduler_gnn_model_motif = None
        scheduler_atom_dec= None
        scheduler_motif_pred = None 
        

    optimizer_list = [optimizer_gnn_model_atom, optimizer_gnn_model_motif, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_motifs]
    
    optim_loss = torch.tensor(float('inf')).to(device)   #初始值正无穷大
    
    epoch_list, loss_list = [], []
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
  
        train_loss = train_mae(args, model_list, loader_atom,loader_motif, optimizer_list, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        val_loss= eval_mae(args, model_list, loader_val, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 20 == 0 or epoch == 1:
                torch.save({'model_atom_state_dict':gnn_model_atom.state_dict(),
                            'model_motif_state_dict':gnn_model_motif.state_dict()
                            }, args.output_model_file + f"_{epoch}.pth")
        if val_loss < optim_loss:
            optim_loss = val_loss
            torch.save({'model_atom_state_dict':gnn_model_atom.state_dict(),
                            'model_motif_state_dict':gnn_model_motif.state_dict()
                            }, args.output_model_file + f"best_model.pth")
       
        if scheduler_gnn_model_atom is not None:
            scheduler_gnn_model_atom.step()
        if scheduler_gnn_model_motif is not None:
            scheduler_gnn_model_motif.step()
        if scheduler_atom_dec is not None:
            scheduler_atom_dec.step()
        if scheduler_motif_pred is not None:
            scheduler_motif_pred.step()
            
        epoch_list.append(epoch)
        loss_list.append(val_loss)
        
    x = np.array(epoch_list)
    y = np.array(loss_list)
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color="red", linewidth=1 )
    plt.yscale('log')
    plt.savefig(f'{args.output_model_file}loss.png',dpi=120, bbox_inches='tight')  

if __name__ == "__main__":
    main()

