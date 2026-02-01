import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, loss_computer='cosine', tau=0.5, args=None):
        """
        初始化对比学习损失函数
        Args:
            loss_computer (str): 相似度计算方式 ('cosine' 或 'euclidean')
            tau (float): 温度参数
            args (Namespace): 其他超参数
        """
        super(ContrastiveLoss, self).__init__()
        self.loss_computer = loss_computer
        self.tau = tau
        self.args = args

    def forward(self, z1, z2):
        """
        计算对比损失
        Args:
            z1 (Tensor): 第一组增强视图的嵌入 (batch_size, embed_dim)
            z2 (Tensor): 第二组增强视图的嵌入 (batch_size, embed_dim)
        Returns:
            loss (Tensor): 对比损失值
        """
        # 标准化嵌入
        # z1 = F.normalize(z1, dim=1)
        # z2 = F.normalize(z2, dim=1)

        # 最小-最大归一化
        min_z1 = z1.min()
        max_z1 = z1.max()
        z1_normalized = (z1 - min_z1) / (max_z1 - min_z1)  # 缩放到 [0, 1]
        z1 = z1_normalized.unsqueeze(1)
        
        min_z2 = z2.min()
        max_z2 = z2.max()
        z2_normalized = (z2 - min_z2) / (max_z2 - min_z2)
        z2 = z2_normalized.unsqueeze(1)
        # 拼接正负样本对
        z = torch.cat([z1, z2], dim=0) # (2*batch_size, embed_dim)
        batch_size = z1.size(0)

        # 相似性矩阵
        if self.loss_computer == 'cosine':
            similarity_matrix = torch.mm(z, z.T)  # 余弦相似性
        elif self.loss_computer == 'euclidean':
            dist_matrix = torch.cdist(z, z, p=2)  # 欧几里得距离
            similarity_matrix = -dist_matrix  # 距离转为相似性（负号）

        # 创建掩码，避免自己与自己匹配
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # 正样本对（同一分子的不同增强视图）
        sim_pos = torch.cat([similarity_matrix[i, batch_size + i].unsqueeze(0) for i in range(batch_size)] +
                            [similarity_matrix[batch_size + i, i].unsqueeze(0) for i in range(batch_size)])

        # 负样本对（掩码除去对角线）
        sim_neg = similarity_matrix[~mask].view(2 * batch_size, -1)

        # InfoNCE Loss 公式
        numerator = torch.exp(sim_pos / self.tau)
        denominator = torch.exp(sim_neg / self.tau).sum(dim=1)
        loss = -torch.log(numerator / denominator).mean()

        return loss
