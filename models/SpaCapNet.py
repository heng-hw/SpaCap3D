import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.transformer_captioner import TransformerDecoderModel

class SpaCapNet(nn.Module):
    def __init__(self, num_class, vocabulary, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=256, vote_factor=1, sampling="vote_fps", no_caption=False,
                 ### transformer-related parameters
                 N=6, h=8, d_model=128, d_ff=2048, transformer_dropout=0.1, bn_momentum=0.1,
                 src_pos_type = None, use_transformer_encoder=False, early_guide=False, check_relation=False):
        super().__init__()
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                       sampling, size_decoded=True if src_pos_type=='loc' else False)

        # ---------  CAPTION GENERATION ---------
        if not no_caption:
            self.caption = TransformerDecoderModel(vocabulary, N, h, d_model, d_ff, transformer_dropout, bn_momentum=bn_momentum,
                                                   src_pos_type = src_pos_type, use_transformer_encoder=use_transformer_encoder, early_guide=early_guide,
                                                   check_relation=check_relation)

    def forward(self, data_dict, is_eval=False):

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- FEATURE EXTRACTION --------- 40000 pts, 1024 seeds
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING --------- 1024 seeds, 1024 votes
        xyz = data_dict["fp2_xyz"] #b,1024,3
        features = data_dict["fp2_features"] #b,256,1024
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz #b,1024,3
        data_dict["seed_features"] = features #b,256,1024

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))

        data_dict["vote_xyz"] = xyz #b,1024, 3
        data_dict["vote_features"] = features #b,256,1024

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict) # vote aggregation -> net (conv-conv-conv) -> decode

        #######################################
        #                                     #
        #    Transformer CAPTION BRANCH       #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, is_eval)

        return data_dict

