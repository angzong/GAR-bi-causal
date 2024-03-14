import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from kmeans_pytorch import kmeans
from utils.misc import build_mlp
from utils.misc import GraphConvolution, get_joint_graph
from models.position_encoding import PositionEmbeddingAbsoluteLearned_1D
from models.position_encoding import LearnedFourierFeatureTransform
from models.gcn import Geo_gcn_withT
from models.relation_transformer import Relation_Trans


class bicausal(nn.Module):
    def __init__(self, args):
        super(bicausal, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.args = args
         
        embedding_dim = args.joint_initial_feat_dim
        self.joint_class_embed_layer = nn.Embedding(args.J, embedding_dim)
        gcn_layers = [
            GraphConvolution(in_features=embedding_dim, out_features=embedding_dim,
                             dropout=0, act=F.relu, use_bias=True) 
            for l in range(self.args.num_gcn_layers)] 
        self.joint_class_gcn_layers = nn.Sequential(*gcn_layers)

        self.adj = get_joint_graph(num_nodes=args.J, joint_graph_path=args.joint_graph_path)
        
        self.special_token_embed_layer = nn.Embedding(args.max_num_tokens, args.hidden_dim)
        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.max_times_embed, embedding_dim)
        self.coords_embed_layer = LearnedFourierFeatureTransform(2, embedding_dim // 2)
        
        # joint track projection layer
        self.joint_projection_layer = build_mlp(input_dim=args.T*embedding_dim*4, 
                                                hidden_dims=[args.hidden_dim], 
                                                output_dim=args.hidden_dim,
                                                use_batchnorm=args.projection_batchnorm,
                                                dropout=args.projection_dropout)

        # person track projection layer
        self.person_track_projection_layer = build_mlp(input_dim=args.T*embedding_dim,
                                                 hidden_dims=[args.hidden_dim], 
                                                 output_dim=args.hidden_dim,
                                                 use_batchnorm=args.projection_batchnorm,
                                                 dropout=args.projection_dropout)

        self.person_feats_T_layer = build_mlp(input_dim=args.J * self.args.joint_wo_coords_dim,
                                                       hidden_dims=[args.hidden_dim],
                                                       output_dim=args.hidden_dim,
                                                       use_batchnorm=args.projection_batchnorm,
                                                       dropout=args.projection_dropout)
        
        # group track projection layer
        if self.args.dataset_name == 'volleyball':
            self.person_to_group_projection = build_mlp(input_dim=(args.N//2)*args.hidden_dim,
                                                      hidden_dims=[args.hidden_dim], 
                                                      output_dim=args.hidden_dim,
                                                      use_batchnorm=args.projection_batchnorm,
                                                      dropout=args.projection_dropout)

            self.ball_track_projection_layer = build_mlp(input_dim=(2*args.joint_initial_feat_dim+4)*args.T,
                                                     hidden_dims=[args.hidden_dim], 
                                                     output_dim=args.hidden_dim,
                                                     use_batchnorm=args.projection_batchnorm,
                                                     dropout=args.projection_dropout)

            self.ball_feats_T_layer = build_mlp(input_dim=(2 * args.joint_initial_feat_dim + 4),
                                                         hidden_dims=[args.hidden_dim],
                                                         output_dim=args.hidden_dim,
                                                         use_batchnorm=args.projection_batchnorm,
                                                         dropout=args.projection_dropout)


        self.person_ball_feats_T_projection_layer = build_mlp(input_dim=(args.T * args.hidden_dim),
                                                           hidden_dims=[args.hidden_dim],
                                                           output_dim=args.hidden_dim,
                                                           use_batchnorm=args.projection_batchnorm,
                                                           dropout=args.projection_dropout)
        
        # Fusion_trans blocks
        if self.args.dataset_name == 'volleyball':
            from models.causal_fusion_volleyball import Fusion_trans
        else:
            print('Please check the dataset name!')
            os._exit(0)
            
        self.Fusion_trans = Fusion_trans(args, args.hidden_dim, args.trans_layers, final_norm=True, return_intermediate=True)
        
        
        # Prediction
        self.classifier = build_mlp(input_dim=args.hidden_dim, 
                                    hidden_dims=None, output_dim=args.num_classes, 
                                    use_batchnorm=args.classifier_use_batchnorm, 
                                    dropout=args.classifier_dropout)

        self.inter_track=build_mlp(input_dim=args.hidden_dim * (args.N+1),
                                      hidden_dims=None, output_dim=args.hidden_dim,
                                      use_batchnorm=args.classifier_use_batchnorm,
                                      dropout=args.classifier_dropout)
        self.group_track = build_mlp(input_dim=args.hidden_dim * 2,
                                     hidden_dims=None, output_dim=args.hidden_dim,
                                     use_batchnorm=args.classifier_use_batchnorm,
                                     dropout=args.classifier_dropout)
        # Prototypes
        self.prototypes = nn.Linear(args.hidden_dim, args.nmb_prototypes, bias=False)
        if self.args.dataset_name == 'volleyball':
            self.IM=Geo_gcn_withT(args.hidden_dim,args.hidden_dim,(args.N+1)*(args.N+1),args.N+1)
            self.person_classifier = build_mlp(input_dim=args.hidden_dim,
                                               hidden_dims=None, output_dim=args.num_person_action_classes,
                                               use_batchnorm=args.classifier_use_batchnorm,
                                               dropout=args.classifier_dropout)
            self.RM = Relation_Trans(d_model=args.hidden_dim, num_encoder_layers=1, num_decoder_layers=1,
                                                       dim_feedforward=1024, args=args)
        
         
    def forward(self, joint_feats, ball_feats_thisbatch):
        B = joint_feats.size(0)
        N = joint_feats.size(1)
        J = joint_feats.size(2)
        T = joint_feats.size(3)
        
        d = self.args.hidden_dim
        
        device = joint_feats.device
        
        joint_feats_basic = joint_feats[:,:,:,:,:self.args.joint_wo_coords_dim] 
          
        # coords positional encoding
        joint_coords = joint_feats[:,:,:,:,-2:].to(torch.int64).cuda()
        coords_h = np.linspace(0, 1, self.args.image_h, endpoint=False)
        coords_w = np.linspace(0, 1, self.args.image_w, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
        image_coords_learned =  self.coords_embed_layer(xy_grid).squeeze(0).permute(1, 2, 0)
        image_coords_embeded = image_coords_learned[joint_coords[:,:,:,:,1], joint_coords[:,:,:,:,0]]
        # (B, N, J, T, 256)
        
        # update by joint_feats removing the raw joint coordinates dim (last 2 dims by default)
        joint_feats = joint_feats[:,:,:,:,:-2]    

            
        # time positional encoding
        time_ids = torch.arange(1, T+1, device=device).repeat(B, N, J, 1).cuda()
        time_seq = self.time_embed_layer(time_ids)  # (B, N, J, T, 256)
        
        # joint classes embedding learning as tokens/nodes
        joint_class_ids = joint_feats[:,:,:,:,-1]  # note that the last dim is the joint class id by default
        joint_classes_emb = self.joint_class_embed_layer(joint_class_ids.type(torch.LongTensor).cuda()) # (B, N, J, T, d_0)
        joint_classes_emb = (joint_classes_emb.transpose(2, 3).reshape(B*N*T,J,-1),
                 self.adj.repeat(B*N*T, 1, 1).cuda())  # adj: # (B*N*T, J, J)
        joint_classes_encode = self.joint_class_gcn_layers(joint_classes_emb)[0].view(B, N, T, J, -1).transpose(2, 3) # (B, N, J, T, 256)

        # update by joint_feats removing the joint class dim (last dim by default)
        joint_feats = joint_feats[:,:,:,:,:-1] # (B, N, J, T, 8)
            

        # CLS_token embedding
        CLS_token = self.special_token_embed_layer(torch.arange(1, device=device).repeat(B, 1))

        
        joint_feats_composite_encoded = torch.cat(
            [joint_feats, time_seq, image_coords_embeded, joint_classes_encode], 
            dim=-1) 


        
        # PROJECTIONS
        # joint track projection
        joints_feature = self.joint_projection_layer(
            joint_feats_composite_encoded.flatten(3, 4).flatten(0, 1).flatten(0, 1)  # (B*N*J, T*d_0)
        ).view(B, N*J, -1)
        # (B, N*J, d)
    

        person_feats_withT = self.person_feats_T_layer(
            joint_feats_basic.transpose(2, 3).contiguous().view(B*N*T, -1)
        ).view(B, N, T, -1)

        relation_feature,person_feats_withT,person_feature = self.RM(person_feats_withT)

        # obtain group feature
        if self.args.dataset_name == 'volleyball':
            people_middle_hip_coords = (
                joint_feats_basic[:,:,11,self.args.group_person_frame_idx,-2:] + 
                joint_feats_basic[:,:,12,self.args.group_person_frame_idx,-2:]) / 2
            # (B, N, 2)  - W, H (X, Y)

            people_idx_sort_by_middle_hip_xcoord = torch.argsort(people_middle_hip_coords[:,:,0], dim=-1)  # (B, N)
            left_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, :int(self.args.N//2)]  # (B, N/2)
            right_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, int(self.args.N//2):]  # (B, N/2)
      
            # form sequence of group track tokens
            left_group_people_repre = person_feature.flatten(
                0,1)[left_group_people_idx.flatten(0,1)].view(B, int(self.args.N//2), -1)  # (B, N/2, d)
            right_group_people_repre = person_feature.flatten(
                0,1)[right_group_people_idx.flatten(0,1)].view(B, int(self.args.N//2), -1)  # (B, N/2, d)
            left_group_feats = self.person_to_group_projection(left_group_people_repre.flatten(1,2))   # (B, d)
            right_group_feats = self.person_to_group_projection(right_group_people_repre.flatten(1,2))   # (B, d)
            group_feats = torch.stack([left_group_feats, right_group_feats], dim=1)  # (B, 2, d)

               

        
        if self.args.dataset_name == 'volleyball':
            ball_coords = ball_feats_thisbatch[:, :, -2:].to(
                torch.int64).cuda()  # the last 2 dims are [x, y], (B, T, 2)
            ball_coords_embeded = image_coords_learned[ball_coords[:, :, 1], ball_coords[:, :, 0]]  # (B, T, d)
            ball_feats_thisbatch = torch.cat([ball_feats_thisbatch[:, :, :-2],  # (B, T, 4)
                                              time_seq[:, 0, 0, :, :],  # (B, T, d)
                                              ball_coords_embeded], dim=-1)
            object_feature = self.ball_track_projection_layer(
                ball_feats_thisbatch.flatten(1, 2)).unsqueeze(1)  # (B, 1, d)
            ball_feats_withT = self.ball_feats_T_layer(ball_feats_thisbatch.view(B * T, -1)).view(B, T,
                                                                                                  -1).unsqueeze(
                1)  # B,1,T,D
            person_ball_feats = torch.cat([ball_feats_withT, person_feats_withT], dim=1)
            inter_person_ball_feats = self.IM(person_ball_feats).transpose(1, 2)  # (B,X, T, d)
            interaction_feature = self.person_ball_feats_T_projection_layer(
                inter_person_ball_feats.flatten(2, 3).view(B * 13, -1)).view(B, 13, -1)

            # Multiscale Transformer Blocks 
            outputs = self.Fusion_trans(CLS_token.transpose(0, 1),  # (1, B, d)
                                        object_feature.transpose(0, 1),  # (1, B, d)
                                        joints_feature.transpose(0, 1),  # (N*J, B, d)
                                        person_feature.transpose(0, 1),  # (N, B, d)
                                        relation_feature.transpose(0, 1),  # (N*(N-1), B, d)
                                        interaction_feature.transpose(0, 1),  # (X, B, d)
                                        group_feats.transpose(0, 1),  # (2, B, d)
                                        left_group_people_idx,
                                        right_group_people_idx
                                        )
        else:
            print('Please check the dataset name!')
            os._exit(0)

         
            
        # CLASSIFIER
        pred_logits = []
        for l in range(self.args.trans_layers):

            relation_cls = outputs[l][0].transpose(0, 1).squeeze(1)  # (B, d)
            inter_cls=outputs[l][1].transpose(0, 1).squeeze(1)  # (B, d)
            group_cls = outputs[l][2].transpose(0, 1).squeeze(1)  # (B, d)
            gi_cls = outputs[l][-1].transpose(0, 1).squeeze(1)  # (B, d)

            relation_cls_normed = nn.functional.normalize(relation_cls, dim=1, p=2)
            inter_cls_normed = nn.functional.normalize(inter_cls, dim=1, p=2)
            group_cls_normed = nn.functional.normalize(group_cls, dim=1, p=2)
            gi_cls_normed = nn.functional.normalize(gi_cls, dim=1, p=2)


            pred_logit_c = self.classifier(relation_cls_normed)
            pred_logit_i=self.classifier(inter_cls_normed)
            pred_logit_g = self.classifier(group_cls_normed)
            pred_logit_gi = self.classifier(gi_cls_normed)

            pred_logits.append([pred_logit_c,pred_logit_i, pred_logit_g,pred_logit_gi])


        scores_r = self.prototypes(relation_cls_normed)
        scores_i=self.prototypes(inter_cls_normed)
        scores_g = self.prototypes(group_cls_normed)
        scores_gi = self.prototypes(gi_cls_normed)
        scores = [scores_r,scores_i, scores_g, scores_gi]
       
    
       
        # PERSON CLASSIFIER
        pred_logits_person = []
        for l in range(self.args.trans_layers):
            person_feats = outputs[l][-2].transpose(0, 1).flatten(0,1)  # (BxN, d)
            pred_logit_person = self.person_classifier(person_feats)
            pred_logits_person.append(pred_logit_person)
        
        return pred_logits, pred_logits_person, scores, group_cls_normed
        
        
