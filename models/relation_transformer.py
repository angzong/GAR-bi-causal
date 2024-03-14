import torch
import torch.nn as nn
from models.STTR_trans import STTR_trans
from utils.misc import build_mlp
class Relation_Trans(nn.Module):
    def __init__(self,d_model=256, num_encoder_layers=2, num_decoder_layers=1,
                                               dim_feedforward=1024,args=None):
        super(Relation_Trans, self).__init__()
        self.RM = STTR_trans(d_model=d_model, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                               dim_feedforward=dim_feedforward)

        self.time_track_projection_layer = build_mlp(input_dim=args.T * args.hidden_dim,
                                                     hidden_dims=[args.hidden_dim],
                                                     output_dim=args.hidden_dim,
                                                     use_batchnorm=False,
                                                     dropout=args.projection_dropout)
        self.interaction_track_projection_layer = build_mlp(input_dim=args.hidden_dim * 2,
                                                            hidden_dims=[args.hidden_dim],
                                                            output_dim=args.hidden_dim,
                                                            use_batchnorm=args.projection_batchnorm,
                                                            dropout=args.projection_dropout)
        self.relation_indexes = [
            args.N * i + j for i in range(args.N)
            for j in range(args.N) if args.N * i + j != args.N * i + i]
        # STTR_trans
    def forward(self, person_feats_withT):
        B = person_feats_withT.size(0)
        N = person_feats_withT.size(1)
        T = person_feats_withT.size(2)
        person_feats_withT = self.RM(person_feats_withT.view(N, B, T, -1))
        person_feats_withT = person_feats_withT.detach()
        person_feature = self.time_track_projection_layer(
            person_feats_withT.view(B* N, -1)
        ).view(B, N, -1)
        # person_feature=self.dropout(person_feature)
        person_feats_withT = person_feats_withT.view(B, N, T, -1)
        # form sequence of person-person-interaction-track tokens
        tem1 = person_feature.repeat(1, N, 1).reshape(B, N, N, -1).transpose(1, 2).flatten(1,2)  # (B, N^2, d)
        tem2 = person_feature.repeat(1, N, 1)  # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interaction_track_feats_thisbatch = tem3[:, self.relation_indexes, :]  # (B, N*(N-1), 2*d)
        relation_feature = self.interaction_track_projection_layer(
            interaction_track_feats_thisbatch.flatten(0, 1)).view(B, N * (N - 1), -1)  # (B, N*(N-1), d)
        return relation_feature,person_feats_withT,person_feature

class Relation_Trans_token(nn.Module):
    def __init__(self,d_model=256, num_encoder_layers=2, num_decoder_layers=1,
                                               dim_feedforward=1024,args=None):
        super(Relation_Trans_token, self).__init__()
        self.RM = STTR_trans(d_model=d_model, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                               dim_feedforward=dim_feedforward)

        self.time_track_projection_layer = build_mlp(input_dim=args.T * args.hidden_dim,
                                                     hidden_dims=[args.hidden_dim],
                                                     output_dim=args.hidden_dim,
                                                     use_batchnorm=False,
                                                     dropout=args.projection_dropout)
        self.interaction_track_projection_layer = build_mlp(input_dim=args.hidden_dim * 2,
                                                            hidden_dims=[args.hidden_dim],
                                                            output_dim=args.hidden_dim,
                                                            use_batchnorm=args.projection_batchnorm,
                                                            dropout=args.projection_dropout)
        self.relation_indexes = [
            args.N * i + j for i in range(args.N)
            for j in range(args.N) if args.N * i + j != args.N * i + i]
        # STTR_trans
    def forward(self, person_feats_withT,token):
        B = person_feats_withT.size(0)
        N = person_feats_withT.size(1)
        T = person_feats_withT.size(2)
        person_feats_out = self.RM(torch.cat([token,person_feats_withT],dim=1).view(N+1, B, T, -1)).detach()
        person_feats_withT = person_feats_out[1:,:,:,:]
        token=person_feats_out[0,:,:,:].view(B,1,T,-1)
        person_feature = self.time_track_projection_layer(
            person_feats_withT.view(B* N, -1)
        ).view(B, N, -1)
        # person_feature=self.dropout(person_feature)
        person_feats_withT = person_feats_withT.view(B, N, T, -1)
        # form sequence of person-person-interaction-track tokens
        tem1 = person_feature.repeat(1, N, 1).reshape(B, N, N, -1).transpose(1, 2).flatten(1,2)  # (B, N^2, d)
        tem2 = person_feature.repeat(1, N, 1)  # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interaction_track_feats_thisbatch = tem3[:, self.relation_indexes, :]  # (B, N*(N-1), 2*d)
        relation_feature = self.interaction_track_projection_layer(
            interaction_track_feats_thisbatch.flatten(0, 1)).view(B, N * (N - 1), -1)  # (B, N*(N-1), d)
        return relation_feature,person_feats_withT,person_feature,token



