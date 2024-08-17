import torch.nn as nn
from models.model_utils import *


class Genomic_SNN_Transformer_MIL(nn.Module):

    def __init__(self, omic_sizes=[82, 328, 513, 452, 1536, 452],
        omic_embedding_size=256, n_classes=2, dropout=0.25):
        super(Genomic_SNN_Transformer_MIL, self).__init__()
        self.muti_snn = Genomic_Muti_SNN(omic_sizes, omic_embedding_size)
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=omic_embedding_size, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(
            omic_encoder_layer, num_layers=2)
        self.omic_attention_net = Attn_Net_Gated(
            L=omic_embedding_size, D=omic_embedding_size, dropout=dropout, n_classes=1)
        self.classifier = nn.Linear(omic_embedding_size, n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        # Muti SNN
        h_omic = self.muti_snn(x_omic)
        h_omic_bag = torch.stack(h_omic)
        # Transformer
        h_omic_trans = self.omic_transformer(h_omic_bag)
        # AMIL
        A_omic, h_omic = self.omic_attention_net(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        # classifier
        h = self.classifier(h_omic)
        h = torch.sigmoid(h)
        return h


class Genomic_Muti_SNN(nn.Module):

    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600],
                embedding_size=256):
        super(Genomic_Muti_SNN, self).__init__()
        hidden_size = [256, 256]
        snns = []
        for dim in omic_sizes:
            snn = [SNN_Block(dim, hidden_size[0])]
            for i, _ in enumerate(hidden_size[1:]):
                snn.append(SNN_Block(hidden_size[i], hidden_size[i+1]))
            snns.append(nn.Sequential(*snn))
        self.snns = nn.ModuleList(snns)

    def forward(self,x):
        feat = [self.snns[idx].forward(sig_feat) for idx, sig_feat in enumerate(x)]
        return feat

class GenomicTransformer(nn.Module):
    
    def __init__(self,omic_input_dim,embedding_size=256,num_layer=2,dropout=0.25,fuse='amil',n_classes=4) -> None:
        super().__init__()
        self.fuse = fuse
        self.token_num = omic_input_dim//embedding_size
        self.embedding_size = embedding_size
        self.fc = SNN_Block(omic_input_dim,self.token_num*embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8, dim_feedforward=embedding_size*2, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        if fuse == 'amil':
            self.amil = AMIL(L=embedding_size, D=embedding_size, dropout=dropout)
        elif fuse=='cls_token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.classifier = nn.Linear(embedding_size,n_classes)
        
    def forward(self,**kwargs):
        # pdb.set_trace()
        x_omic = kwargs['x_omic']
        if len(x_omic.shape)==1:
            x_omic = x_omic.unsqueeze(0)
        x_omic = self.fc(x_omic)
        x_omic = x_omic.split(self.embedding_size,dim=1)
        x_omic = torch.stack(x_omic,dim=1)
        if self.fuse =='cls_token':
            cls_tokens = self.cls_token.expand(x_omic.shape[0], -1, -1)
            x_omic = torch.cat([cls_tokens,x_omic],dim=1)
        x_omic = self.transformer(x_omic)
        if self.fuse=='amil':
            x_omic,attention = self.amil(x_omic)
        elif self.fuse =='cls_token':
            x_omic = x_omic[:,0]
            attention = None  
        elif self.fuse == 'avgpool':
            x_omic =torch.mean(x_omic,dim=1)
            attention = None 
        logits = self.classifier(x_omic)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        res = {
            "hazards":hazards,
            "survival":S,
            "y_hat":Y_hat,
            'attention_scores':attention
        }
        return res
           
    