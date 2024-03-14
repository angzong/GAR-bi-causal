import torch
import copy
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm

from models.transformer import TransformerEncoderLayer
from utils.misc import build_mlp


class Fusion_trans(Module):
    def __init__(self, args, d_model, num_layers, final_norm=None, return_intermediate=False):
        '''
        d_model: representation dimension
        num_layers: number of Fusion_trans blocks
        final_norm: whether to use layer norm for final output
        return_intermediate: whether to return output from every Fusion_trans block
        '''
        super(Fusion_trans, self).__init__()
        
        encoder_layer = Fusion_Block(args, d_model)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm = final_norm
        self.return_intermediate = return_intermediate

        if self.final_norm:
            self.norm_output_fine = LayerNorm(d_model)
            self.norm_output_middle = LayerNorm(d_model)
            self.norm_output_middle_cls = LayerNorm(d_model)
            self.norm_output_relation = LayerNorm(d_model)
            self.norm_output_group =  LayerNorm(d_model)
            self.norm_output_inter = LayerNorm(d_model)
            self.norm_output_cls_gi=LayerNorm(d_model)

    def forward(self, CLS_token, ball, joints, person, relation,inter, group, left_group_people_idx, right_group_people_idx):
        '''
        CLS_token: (1, B, d)
        ball: (1, B, d)
        joints: (F, B, d)
        person: (M, B, d)
        relation: (C, B, d)
        group: (2, B, d)
        '''
        output_CLS = CLS_token
        output_ball = ball
        output_joints = joints
        output_person = person
        output_rela = relation
        output_group = group
        output_inter=inter
        intermediate = []

        for mod in self.layers:
            CLS_f, CLS_m, CLS_c,CLS_i, output_CLS, output_ball, output_joints, output_person, output_rela,output_inter, output_group,CLS_gi = mod(
                output_CLS, output_ball, output_joints, output_person, output_rela,output_inter, output_group,
                left_group_people_idx, right_group_people_idx
            )
            if self.return_intermediate:
                intermediate.append(
                    [CLS_c,CLS_i, output_CLS,output_person, CLS_gi])
                
        if self.final_norm is not None:
            CLS_c = self.norm_output_relation(CLS_c)
            CLS_i=self.norm_output_inter(CLS_i)
            output_CLS = self.norm_output_group(output_CLS)
            output_person = self.norm_output_middle(output_person)
            CLS_gi=self.norm_output_cls_gi(CLS_gi)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append([CLS_c,CLS_i, output_CLS,output_person, CLS_gi])
        if self.return_intermediate:
            return intermediate
        else:
            return CLS_c,CLS_i, output_CLS, output_person, CLS_gi




class Fusion_Block(Module):
    def __init__(self, args, d_model):
        super(Fusion_Block, self).__init__()
        
        self.joints_trans_block = TransformerEncoderLayer(
            d_model, args.innerTx_nhead, args.innerTx_dim_feedforward, 
            args.innerTx_dropout, args.innerTx_activation)
        
        self.joints2person = Joint2Person(args, args.N, args.J, d_model)
        
        self.person_trans_block = TransformerEncoderLayer(
            d_model, args.middleTx_nhead, args.middleTx_dim_feedforward,
            args.middleTx_dropout, args.middleTx_activation)
        
        self.middle2rela = Person2Relation(args, args.N, args.J, d_model)
        self.person2inter = Person2Interaction(args, args.N, args.J, d_model)
        self.outer_tblock = TransformerEncoderLayer(
            d_model, args.outerTx_nhead, args.outerTx_dim_feedforward, 
            args.outerTx_dropout, args.outerTx_activation)
        self.inter_trans_block = TransformerEncoderLayer(
            d_model, args.outerTx_nhead, args.outerTx_dim_feedforward,
            args.outerTx_dropout, args.outerTx_activation)

        self.rela2group = Rela2Group(args.N//2, d_model)
        
        self.group_trans_block = TransformerEncoderLayer(
            d_model, args.groupTx_nhead, args.groupTx_dim_feedforward,
            args.groupTx_dropout, args.outerTx_activation)
        self.inter2group=Inter2Group(args.N//2, d_model)
        
    def forward(self, CLS_token, ball, joints, person, relation,inter, group, left_group_people_idx, right_group_people_idx):
        '''
        CLS_token: (1, B, d)
        joints: (N*J, B, d)
        person: (N, B, d)
        relation: (N*(N-1), B, d)
        inter: (N+O, B, d)
        group: (2, B, d)
        '''
         
        output_joints = self.joints_trans_block(torch.cat([CLS_token, ball, joints], dim=0))
        CLS_f = output_joints[0, :, :].unsqueeze(0)
        ball_f = output_joints[1, :, :].unsqueeze(0)
        output_joints = output_joints[2:, :, :]
        person_update = person + self.joints2person(output_joints.transpose(0, 1)).transpose(0, 1)
        output_person = self.person_trans_block(torch.cat([CLS_f, ball_f, person_update], dim=0))
        CLS_m = output_person[0, :, :].unsqueeze(0)
        ball_m = output_person[1, :, :].unsqueeze(0)
        output_person = output_person[2:, :, :]
        inter_update = inter + self.person2inter(output_person,ball_m)
        output_inter = self.inter_trans_block(torch.cat([CLS_m, ball_m, inter_update], dim=0))
        CLS_i = output_inter[0, :, :].unsqueeze(0)
        ball_i = output_inter[1, :, :].unsqueeze(0)
        output_inter = output_inter[2:, :, :]
        inter_no_ball=output_inter

        rela_update = relation + self.middle2rela(output_person.transpose(0, 1)).transpose(0, 1)
        output_rela = self.outer_tblock(torch.cat([CLS_m, ball_m, rela_update], dim=0))
        CLS_c = output_rela[0, :, :].unsqueeze(0)
        ball_c = output_rela[1, :, :].unsqueeze(0)
        output_rela = output_rela[2:, :, :]

        group_rela=self.rela2group(output_rela.transpose(0, 1), left_group_people_idx, right_group_people_idx).transpose(0, 1)
        group_inter=self.inter2group(inter_no_ball.transpose(0, 1), left_group_people_idx, right_group_people_idx).transpose(0, 1)
        group_update = group + group_rela+ group_inter

        output_group = self.group_trans_block(torch.cat([CLS_c,CLS_i, ball_c,ball_i, group_update], dim=0))
        CLS_g = output_group[0, :, :].unsqueeze(0)
        ball_g = output_group[2, :, :].unsqueeze(0)
        CLS_gi=output_group[1, :, :].unsqueeze(0)
        ball_gi = output_group[3, :, :].unsqueeze(0)
        output_group = output_group[4:, :, :]


        return CLS_f, CLS_m, CLS_c,CLS_i, CLS_g, ball_g, output_joints, output_person, output_rela,output_inter, output_group, CLS_gi



class Joint2Person(Module):
    def __init__(self, args, N, J, d_model):
        super(Joint2Person, self).__init__()
        
        self.N, self.J, self.d = N, J, d_model
        
        # person identity projection layer
        self.person_projection_layer = build_mlp(input_dim=J*d_model,
                                                 hidden_dims=[d_model], 
                                                 output_dim=d_model)
                      
    def forward(self, joint_feats):
        # joint_feats: (B, N*J, d)
        B = joint_feats.size(0)
        N, J, d = self.N, self.J, self.d
        
        # person identity projection
        person_feats_thisbatch_proj = self.person_projection_layer(
            joint_feats.view(-1, N, J, d).flatten(2, 3)  # (B, N, J*d)
        )
        # (B, N, d)
        return person_feats_thisbatch_proj


class Inter2Group(Module):
    def __init__(self,num_person_per_group, d_model):
        super(Inter2Group, self).__init__()
        self.num_person_per_group = num_person_per_group
        self.person_to_group_projection = build_mlp(input_dim=num_person_per_group * d_model,
                                                    hidden_dims=[d_model],
                                                    output_dim=d_model)

    def forward(self, person_feats_thisbatch_proj, left_group_people_idx, right_group_people_idx):
        B = person_feats_thisbatch_proj.size(0)
        # form sequence of group track tokens
        left_group_people_repre = person_feats_thisbatch_proj.flatten(
            0, 1)[left_group_people_idx.flatten(0, 1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        right_group_people_repre = person_feats_thisbatch_proj.flatten(
            0, 1)[right_group_people_idx.flatten(0, 1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        left_group_feats = self.person_to_group_projection(
            left_group_people_repre.flatten(1, 2))  # (B, d)
        right_group_feats = self.person_to_group_projection(
            right_group_people_repre.flatten(1, 2))  # (B, d)
        group_feats = torch.stack(
            [left_group_feats, right_group_feats], dim=1)  # (B, 2, d)

        return group_feats


class Person2Interaction(Module):
    def __init__(self, args, N, J, d_model):
        super(Person2Interaction, self).__init__()

        self.N, self.J, self.d = N, J, d_model

        # person interaction projection layer
        self.interaction_projection_layer = build_mlp(input_dim=d_model,
                                                      hidden_dims=[d_model],
                                                      output_dim=d_model)

    def forward(self, person_feats_thisbatch_proj,ball_feats):
        feats=torch.cat([person_feats_thisbatch_proj,ball_feats],dim=0)
        interactions_thisbatch_proj = self.interaction_projection_layer(feats)  # (B, N*(N-1), d)
        return interactions_thisbatch_proj
    
class Person2Relation(Module):
    def __init__(self, args, N, J, d_model):
        super(Person2Relation, self).__init__()
        
        self.N, self.J, self.d = N, J, d_model
        
        self.interaction_indexes = [N*i+j for i in range(N) for j in range(N) if N*i+j != N*i+i]
        
        # person interaction projection layer
        self.interaction_projection_layer = build_mlp(input_dim=d_model*2,
                                                      hidden_dims=[d_model], 
                                                      output_dim=d_model)
                      
    def forward(self, person_feats_thisbatch_proj):
        B = person_feats_thisbatch_proj.size(0)
        N, J, d = self.N, self.J, self.d
        
        # form sequence of person-person-interaction tokens
        tem1 = person_feats_thisbatch_proj.repeat(1, N, 1).reshape(B,N,N,d).transpose(1, 2).flatten(1, 2)  # (B, N^2, d)
        tem2 = person_feats_thisbatch_proj.repeat(1, N, 1) # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interactions_thisbatch = tem3[:, self.interaction_indexes, :]  # (B, N*(N-1), 2*d)
        interactions_thisbatch_proj = self.interaction_projection_layer(interactions_thisbatch)  # (B, N*(N-1), d)
        return interactions_thisbatch_proj


class Rela2Group(Module):
    def __init__(self, num_person_per_group, d_model):
        super(Rela2Group, self).__init__()
        self.num_person_per_group = num_person_per_group
        self.person_to_group_projection = build_mlp(input_dim=num_person_per_group*d_model,
                                                      hidden_dims=[d_model], 
                                                      output_dim=d_model)
                      
    def forward(self, person_feats_thisbatch_proj, left_group_people_idx, right_group_people_idx):
        B = person_feats_thisbatch_proj.size(0)
        # form sequence of group track tokens
        left_group_people_repre = person_feats_thisbatch_proj.flatten(
            0,1)[left_group_people_idx.flatten(0,1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        right_group_people_repre = person_feats_thisbatch_proj.flatten(
            0,1)[right_group_people_idx.flatten(0,1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        left_group_feats = self.person_to_group_projection(left_group_people_repre.flatten(1,2))   # (B, d)
        right_group_feats = self.person_to_group_projection(right_group_people_repre.flatten(1,2))   # (B, d)
        group_feats = torch.stack([left_group_feats, right_group_feats], dim=1)  # (B, 2, d)

        return group_feats
    

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
