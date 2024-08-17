from collections import OrderedDict
from os.path import join
import pdb

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from models.genomic import Genomic_Muti_SNN

            
################################
### Deep Sets Implementation ###
################################
class MIL_Sum_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Deep Sets Implementation.

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Sum_FC_surv, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {
            "small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        # Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(
            *[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        # Constructing Genomic SNN
        if self.fusion != None:
            # hidden = [256, 256]
            hidden = [256, 256,256,256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(
                    SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(
                    *[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(
                    dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(
                self.phi, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        h_path = self.phi(x_path).sum(axis=1)
        h_path = self.rho(h_path)

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic).squeeze(dim=0)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0),
                            h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path  # [256] vector

        # logits needs to be a [1 x 4] vector
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risks = -torch.sum(S, dim=1)
        
        res = {
            "hazards":hazards,
            "survival":S,
            "risks":risks,
            "y_hat":Y_hat
        }

        return res


################################
# Attention MIL Implementation #
################################
class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg="small", dropout=0.25, n_classes=4,drop_instance=0.0):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv, self).__init__()
        self.fusion = fusion
        self.drop_instance = drop_instance
        self.size_dict_path = {
            "small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        # Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        # fc = [nn.Linear(size[0], size[1]), nn.ELU(), nn.AlphaDropout(dropout)]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        # self.rho = nn.Sequential(
        #     *[nn.Linear(size[1], size[2]), nn.ELU(), nn.AlphaDropout(dropout)])
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        # Constructing Genomic SNN
        if self.fusion is not None:
            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(
                    SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(
                    *[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(
                    dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        if self.drop_instance!=0.0 and self.training:
            B,N,L = x_path.shape
            index = torch.LongTensor(random.sample(range(N), int(N*(1-self.drop_instance)))).to(x_path.device)
            x_path = torch.index_select(x_path,dim=1,index=index)

        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 2, 1)

        A = F.softmax(A, dim=2)
        h_path = torch.bmm(A, h_path)
        h_path = self.rho(h_path).squeeze(1)

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0),
                            h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=1))
            elif self.fusion=='add':
                h = h_path+h_omic
        else:
            h = h_path  # [256] vector

        # logits needs to be a [1 x 4] vector
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        res = {
            "hazards": hazards,
            "survival": S,
            "y_hat": Y_hat,
        }

        return res

    def forward_one_wsi(self,wsi):
        A, h_path = self.attention_net(wsi)
        
        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)
        h= torch.mm(A, h_path)
        h = self.rho(h)
        logits = self.classifier(h)
        risk = torch.sigmoid(logits)
        
        h = self.rho(h_path)
        logits = self.classifier(h)
        patch_risk= torch.sigmoid(logits)
        res = {
            "risk":risk.squeeze(),
            "patch_risk":patch_risk.squeeze(),
            "attention":A.squeeze()
        }
        return res

#############
# AMIL+SnnMIL#
#############
class TwoAMIL(nn.Module):
    def __init__(self, fusion=None, size_arg="small", omic_sizes=[82, 328, 513, 452, 1536, 452],
                 omic_embedding_size=256,  dropout=0.25, n_classes=4):
        super(TwoAMIL, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {
            "small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        # path
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        # gene
        self.muti_snn = Genomic_Muti_SNN(omic_sizes, omic_embedding_size)
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=omic_embedding_size, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(
            omic_encoder_layer, num_layers=2)
        self.omic_attention_net = Attn_Net_Gated(
            L=omic_embedding_size, D=omic_embedding_size, dropout=dropout, n_classes=1)


        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                *[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(
                dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]

        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        
        
        h_omic = self.muti_snn(x_omic)
        h_omic_bag = torch.stack(h_omic)
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_net(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic).squeeze()
        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0),
                            h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        res = {
            "hazards": hazards,
            "survival": S,
            "y_hat": Y_hat,
        }

        return res


######################################
# Deep Attention MISL Implementation #
######################################
class MIL_Cluster_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, num_clusters=10, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC_surv, self).__init__()
        self.size_dict_path = {
            "small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.num_clusters = num_clusters
        self.fusion = fusion

        # FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        # WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        # Genomic SNN Construction + Multimodal Fusion
        if fusion is not None:
            hidden = self.size_dict_omic['small']
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(
                    SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if fusion == 'concat':
                self.mm = nn.Sequential(
                    *[nn.Linear(size[2]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(
                    dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        cluster_id = x_path['path_clusters'].detach().cpu().numpy()
        x_path = x_path['path_features']

        # FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id == i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1, 512)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(
                h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        # Attention MIL
        A, h_path = self.attention_net(h_cluster)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        # Attention MIL + Genomic Fusion
        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0),
                            h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None
