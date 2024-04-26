import os.path as osp 
import torch
import torch.nn as nn

from lib.models.spin import spin_backbone_init
from lib.models.pooling import *

class CFST(nn.Module):
    def __init__(self, 
                 d_model=512,
                 ) :
        super().__init__()
        """
        Tree
        - SPIN backbone
        - Temporal brach
            - Pooling
            - Transformer
        - Spatial barch
            - Pooling
            - Transformer
        - Fine branch
            - ST pooing
            - 2Aggergation
        """
        super(CFST, self).__init__()
        self.d_model = d_model
        ##########################
        # SPIN Backbone
        ##########################
        self.spin_backbone = spin_backbone_init()

        ##########################
        # Temporal branch
        ##########################



    def forward(self, x):
        """
        x : [B, T, 3, H, W]
        """
        ##########################
        # SPIN Backbone
        ##########################
        B, T, _, H, W = x.shape
        x = x.reshape(B*T, -1, H, W)
        featmap = self.spin_backbone(x)
        st_feat = featmap.permute(0, 2, 3, 1).reshape(B, T, -1, self.d_model)   # []

        s_feat = temporal_pool(st_feat)
        t_feat = sptial_pool(st_feat)
        




