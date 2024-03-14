import torch
import torch.nn as nn
from models.transformer import TransformerEncoderLayer,TransformerEncoder,Transformer,TransformerDecoderLayer
from models.position_encoding import PositionEmbeddingAbsoluteLearned_1D
class trans_block(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu"):
        super(trans_block, self).__init__()
        self.time_encoder_layer=TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.position_encoder_layer=TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(50, d_model)
        self.position_embed_layer = PositionEmbeddingAbsoluteLearned_1D(50, d_model)
    def forward(self, x):
        N, B, T, C = x.shape
        device=x.device

        #time
        temp_T = x.reshape(N*B,T, -1)
        # time embed
        time_ids = torch.arange(1, T + 1, device=device).repeat(B*N, 1).cuda()
        time_seq = self.time_embed_layer(time_ids)  # (B* N, T, 256)
        temp_T=temp_T+time_seq
        temp_T=self.time_encoder_layer(temp_T)

        #spitial
        temp_S=x.permute(1,2,0,3).contiguous().reshape(B*T,N,-1)
        spitial_ids = torch.arange(1, N + 1, device=device).repeat(B * T, 1).cuda()
        spitial_seq = self.position_embed_layer(spitial_ids)  # (B* T, N, 256)
        temp_S = temp_S + spitial_seq
        temp_S=self.position_encoder_layer(temp_S)

        tempST=self.time_encoder_layer(x.reshape(N*B,T, -1)+temp_S.reshape(B,T,N,-1).permute(0,2,1,3).contiguous().reshape(B*N,T,-1))
        tempTS=self.position_encoder_layer(x.permute(1,2,0,3).contiguous().reshape(B*T,N,-1)+temp_T.reshape(N,B,T,-1).permute(1,2,0,3).contiguous().reshape(B*T,N,-1))
        memory1 = tempST.reshape(N, B, T, -1)
        memory2 = tempTS.reshape(T, B, N, -1).permute(2, 1, 0, 3).contiguous().reshape(N, B, T, -1)
        memory = memory1 + memory2
        return memory
class STTR_trans(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(STTR_trans, self).__init__()
        self.num_STTR_layers = num_encoder_layers
        self.STTR_module = nn.ModuleList()
        for i in range(self.num_STTR_layers):
            self.STTR_module.append(trans_block(d_model, nhead, dim_feedforward, dropout, activation))
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,x):
        N,B,T,C = x.shape
        memory=x
        for i in range(self.num_STTR_layers):
            memory = self.STTR_module[i](memory)
        memory = self.dropout(memory)
        return memory
