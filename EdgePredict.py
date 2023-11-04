# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 9:33
# @Author  : yang chen
import torch
import torch.nn as nn

# ----- 损失函数 -----
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.BCELoss = nn.BCELoss(reduction='none')

    def forward(self, pred, label):
        '''
        :param preds:经过sigmoid  shape->(B)
        :param labels: shape->(B)
        :return:
        '''
        pt = label * (1 - pred) + (1 - label) * pred
        # focal_weight = (label * self.alpha + (1 - label) * (1 - self.alpha)) * pt.pow(self.gamma)

        loss = self.BCELoss(pred, label) * pt
        return loss.mean()

# ----- 模型构建 -----
class LayerCorrelation(nn.Module):
    def __init__(self, input_dim=[256, 128, 64], hidden_dim=128, out_dim=256):
        super().__init__()
        # 所有输入特征进行维度统一
        self.FC1 = nn.Linear(input_dim[0], hidden_dim, bias=False)
        self.FC2 = nn.Linear(input_dim[1], hidden_dim, bias=False)
        self.FC3 = nn.Linear(input_dim[2], hidden_dim, bias=False)

        #
        self.Attention_W = nn.Linear(hidden_dim, 3)

        #
        self.out_W = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        # 所有特征维度统一
        f1 = self.FC1(x1)
        f2 = self.FC2(x2)
        f3 = self.FC3(x3)

        # 堆叠所有特征
        F = torch.stack([f1, f2, f3], dim=1)

        # 计算layer-wise corrlation
        attenion_score = self.Attention_W(F)
        attenion_score = attenion_score.softmax(dim=-1)

        #
        out = torch.bmm(attenion_score, F) + F
        out = self.relu(self.out_W(out))
        return out

class ElementCorrelation(nn.Module):
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        #
        self.W1 = nn.Linear(1, in_dim // 2)
        self.W2 = nn.Linear(1, in_dim // 2)
        #
        self.out_W = nn.Linear(in_dim, out_dim)
        self.out_bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #
        b1 = self.W1(x)  # [B,dim,dim//2]
        b2 = self.W2(x)  # [B,dim,dim//2]
        #
        attention_score = torch.bmm(b1, b2.transpose(1, 2))  # [B,dim,dim]
        attention_score = attention_score.softmax(dim=-1)  # [B,dim,dim]
        #
        out = torch.bmm(attention_score, x)  # [B,dim,1]
        out = torch.squeeze(out + x)
        out = self.out_W(out)
        out = self.out_bn(out)
        out = self.relu(out)
        return out

class PyramidFeatureExtractModule(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64]):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.BatchNorm1d(num_features=channels[1]),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Linear(channels[1], channels[2]),
            nn.BatchNorm1d(num_features=channels[2]),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Linear(channels[2], channels[3]),
            nn.BatchNorm1d(num_features=channels[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return stage1, stage2, stage3

class MDA(nn.Module):
    def __init__(self, data_helper):
        super().__init__()

        self.num_miRNA = data_helper.data['num_of_mirna']
        self.num_disease = data_helper.data['num_of_disease']

        # local view
        self.miRNA_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['md_local_embedding'][:self.num_miRNA]))
        self.disease_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['md_local_embedding'][self.num_miRNA:]))

        # global view
        self.miRNA_global_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['md_global_embedding'][:self.num_miRNA]))
        self.disease_global_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['md_global_embedding'][self.num_miRNA:]))

        # semantic view
        self.mdm_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['mdm_embedding']))  # m-d-m metapath miRNA-miRNA graph
        self.dmd_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(data_helper.data['dmd_embedding']))  # d-m-d metapath disease-disease graph

        #
        self.PFEM = PyramidFeatureExtractModule(channels=[512, 256, 128, 64])
        self.LayerCorrelation = LayerCorrelation()
        self.ElementCorrelation = ElementCorrelation()

        #
        self.classificer = nn.Sequential(
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_index):
        # 获取边索引
        miRNA_index, disease_index = edge_index[:, 0], edge_index[:, 1]

        # 局部视角的特征
        m_local_embed, d_local_embed = self.miRNA_embedding(miRNA_index), self.disease_embedding(disease_index)

        # 全局视角的特征
        m_global_embed, d_global_embed = self.miRNA_global_embedding(miRNA_index), self.disease_global_embedding(disease_index)

        # 元路径视角的语义特征
        mdm_embed, dmd_embed = self.mdm_embedding(miRNA_index), self.dmd_embedding(disease_index)

        # 局域图特征与语义信息特征融合
        F0 = torch.cat([mdm_embed, m_local_embed + m_global_embed, d_local_embed + d_global_embed, dmd_embed], dim=-1)
        # F0 = torch.cat([m_local_embed + m_global_embed, d_local_embed + d_global_embed], dim=-1)    #local+global
        # F0 = torch.cat([m_local_embed , d_local_embed ], dim=-1)    #only local
        # F0 = torch.cat([m_global_embed, d_global_embed], dim=-1)    #only global
        F1, F2, F3 = self.PFEM(F0)
        F = self.LayerCorrelation(F1, F2, F3)  # [B,3,256]
        batch_size, c, dim = F.shape
        F = F.view(batch_size, c * dim, -1)  # [B,3*256,1]
        F = self.ElementCorrelation(F)

        out = self.classificer(F)
        return out.flatten()