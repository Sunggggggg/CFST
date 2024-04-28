import os.path as osp 
import torch
import torch.nn as nn

from lib.models.spin import spin_backbone_init
from lib.models.encoder import STencoder
from lib.models.trans_operator import CrossAttention

class CFST(nn.Module):
    def __init__(self, 
                 seqlen=16,
                 n_layers=3,
                 d_model=2048,
                 num_head=8, 
                 dropout=0., 
                 drop_path_r=0.,
                 atten_drop=0.,
                 mask_ratio=0.,
                 stride_short=4,
                 short_n_layers = 3,
                 device=torch.device('cuda:0'),
                 ) :
        super().__init__()
        super(CFST, self).__init__()
        self.seqlen = seqlen
        self.stride_short = stride_short
        self.mid_frame = seqlen // 2
        self.d_model = d_model 
        ##########################
        # SPIN Backbone
        ##########################
        self.spin_backbone = spin_backbone_init(device)
        self.patchfiy = nn.Conv2d(d_model, d_model, 2, 2)
        self.num_patch = num_patch = int((224/8/2)**(2))

        ##########################
        # STBranch
        ##########################
        self.stencoder = STencoder(seqlen=seqlen, hw=num_patch, embed_dim=d_model, stride_short=stride_short,
                              n_layers=n_layers, short_n_layers=short_n_layers, num_head=num_head, dropout=dropout, drop_path_r=drop_path_r, 
                              atten_drop=atten_drop, mask_ratio=mask_ratio)
        
        ##########################
        # Aggregation
        ##########################
        self.s_proj = nn.Linear(d_model, d_model*2)
        self.t_proj = nn.Linear(d_model, d_model*2)
        self.local_spa_atten = CrossAttention(d_model*2, num_heads=num_head, qk_scale=True, qkv_bias=None)
        self.local_tem_atten = CrossAttention(d_model*2, num_heads=num_head, qk_scale=True, qkv_bias=None)

        self.to(device)


    def forward(self, x, is_train=False):
        """
        x : [B, T, 3, H, W]
        """
        ##########################
        # SPIN Backbone
        ##########################
        B = x.shape[0]
        x = torch.flatten(x, 0, 1)
        featmap = self.spin_backbone(x) # [BT, d, H, W]
        global_st_feat = self.patchfiy(featmap).flatten(-2).permute(0, -1, 1)           
        global_st_feat = featmap.reshape(B, self.seqlen, self.num_patch, self.d_model)              # [B, T, N, d]

        local_st_feat, global_temporal_feat, global_spatial_feat = self.stencoder(global_st_feat, is_train=is_train)
        proj_spatial_feat = self.s_proj(global_spatial_feat)
        proj_temporal_feat = self.t_proj(global_temporal_feat)

        local_st_feat = self.local_spa_atten(local_st_feat, proj_spatial_feat)
        local_st_feat = self.local_tem_atten(local_st_feat, proj_temporal_feat)

        return local_st_feat

        
        




