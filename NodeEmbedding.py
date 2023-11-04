# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 14:14
# @Author  : yang chen

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv,GAE
from torch_geometric.utils import to_undirected, negative_sampling

from Dataloader import Data_Helper
from Logger import Logger
data_helper = Data_Helper(2)
logger = Logger()
device = 'cuda:0'
out_dim = 999999
num_epoch = 999999
lr = 1

class GCN_Encoder(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()

        layer_channel = [
            in_dim,(in_dim+out_dim)//2,(in_dim+out_dim)//4,out_dim
        ]
        # 3*(GCN+MLP)
        self.GCN1 = GCNConv(layer_channel[0], layer_channel[1])
        self.FC2 = nn.Linear(layer_channel[1], layer_channel[1], bias=False)
        self.GCN3 = GCNConv(layer_channel[1], layer_channel[2])
        self.FC4 = nn.Linear(layer_channel[2], layer_channel[2], bias=False)
        self.GCN5 = GCNConv(layer_channel[2], layer_channel[3])
        self.FC6 = nn.Linear(layer_channel[3], layer_channel[3], bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        z1 = self.activation(self.GCN1(x, edge_index))
        z1 = self.activation(z1 + self.FC2(z1))
        z2 = self.activation(self.GCN3(z1, edge_index))
        z2 = self.activation(z2 + self.FC4(z2))
        z3 = self.activation(self.GCN5(z2, edge_index))
        z3 = self.activation(z3 + self.FC6(z3))
        return z3

def local_GAE():
    md_feature = torch.FloatTensor(data_helper.md_feature).to(device)
    md_edge_index = torch.LongTensor([
        list(data_helper.md_pos_edge_index[:,0]),
        list(data_helper.md_pos_edge_index[:,1])
    ]).to(device)
    graph = Data(x=md_feature,edge_index=to_undirected(md_edge_index))

    model = GAE(encoder=(GCN_Encoder(in_dim=md_feature.shape[1],out_dim=out_dim))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1,num_epoch+1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(graph.x,graph.edge_index)
        pos_edge_index = graph.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,num_nodes=graph.num_nodes,num_neg_samples=graph.num_edges
        )
        loss = model.recon_loss(z,pos_edge_index,neg_edge_index)
        loss.backward()
        optimizer.step()
        print('[Local GAE] epoch:{} Loss:{}'.format(epoch,loss.item()))
        logger.add_loss(loss.item(), 'LOCAL', epoch)
    np.savetxt('./miRNA_disease_local_embedding.txt', z.detach().cpu().numpy(), fmt='%.4f')

def global_GAE():
    md_ones_feature = torch.FloatTensor(data_helper.md_ones_feature).to(device)
    md_ones_edge_index = torch.LongTensor([
        list(data_helper.md_ones_pos_edge_index[:,0]),
        list(data_helper.md_ones_pos_edge_index[:,1])
    ]).to(device)
    graph = Data(x=md_ones_feature,edge_index=to_undirected(md_ones_edge_index))

    model = GAE(encoder=(GCN_Encoder(in_dim=md_ones_feature.shape[1],out_dim=out_dim))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1,num_epoch+1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(graph.x,graph.edge_index)
        pos_edge_index = graph.edge_index
        loss = -torch.log(model.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        loss.backward()
        optimizer.step()
        print('[Global GAE] epoch:{} MSELoss:{}'.format(epoch, loss.item()))
        logger.add_loss(loss.item(), 'GLOBAL', epoch)
    np.savetxt('./miRNA_disease_global_embedding.txt', z.detach().cpu().numpy(), fmt='%.4f')

def semantic_GAE_for_mdm():
    mdm_feature = torch.FloatTensor(data_helper.mdm_feature).to(device)
    mdm_edge_index = torch.LongTensor([
        list(data_helper.mdm_pos_edge_index[:, 0]),
        list(data_helper.mdm_pos_edge_index[:, 1])
    ]).to(device)
    graph = Data(x=mdm_feature, edge_index=to_undirected(mdm_edge_index))
    model = GAE(encoder=(GCN_Encoder(in_dim=mdm_feature.shape[1], out_dim=out_dim))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epoch + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(graph.x, graph.edge_index)
        pos_edge_index = graph.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=graph.num_nodes, num_neg_samples=graph.num_edges
        )
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        print('[Semantic miRNA GAE] epoch:{} Loss:{}'.format(epoch,loss.item()))
        logger.add_loss(loss.item(), 'SEMANTIC_MDM', epoch)
    np.savetxt('./miRNA_semantic_embedding.txt', z.detach().cpu().numpy(), fmt='%.4f')

def semantic_GAE_for_dmd():
    dmd_feature = torch.FloatTensor(data_helper.dmd_feature).to(device)
    dmd_edge_index = torch.LongTensor([
        list(data_helper.dmd_pos_edge_index[:, 0]),
        list(data_helper.dmd_pos_edge_index[:, 1])
    ]).to(device)
    graph = Data(x=dmd_feature, edge_index=to_undirected(dmd_edge_index))
    model = GAE(encoder=(GCN_Encoder(in_dim=dmd_feature.shape[1], out_dim=out_dim))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epoch + 1):
        model.train()
        optimizer.zero_grad()
        # encode
        z = model.encode(graph.x, graph.edge_index)
        # negtive sample
        pos_edge_index = graph.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=graph.num_nodes, num_neg_samples=graph.num_edges
        )
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        print('[Semantic disease GAE] epoch:{} Loss:{}'.format(epoch, loss.item()))
        logger.add_loss(loss.item(), 'SEMANTIC_DMD', epoch)
    np.savetxt('./disease_semantic_embedding.txt', z.detach().cpu().numpy(), fmt='%.4f')