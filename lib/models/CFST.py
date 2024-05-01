import math
import os.path as osp 
import torch
import torch.nn as nn

from lib.models.spin import spin_backbone_init, Regressor
from lib.models.encoder import STencoder
from lib.models.trans_operator import CrossAttention, Mlp
from lib.models.HSCR import HSCR

class CFST(nn.Module):
    def __init__(self, 
                 seqlen=16,
                 n_layers=3,
                 d_model=1024,
                 num_head=8, 
                 dropout=0., 
                 drop_path_r=0.,
                 atten_drop=0.,
                 mask_ratio=0.,
                 stride_short=4,
                 short_n_layers = 3,
                 device=torch.device('cuda'),
                 ) :
        super().__init__()
        super(CFST, self).__init__()
        self.seqlen = seqlen
        self.stride_short = stride_short
        self.mid_frame = seqlen // 2
        self.d_model = d_model 
        self.num_patch = num_patch = int((224/8/2)**(2))    # 14**2

        ##########################
        # STBranch
        ##########################
        self.stencoder = STencoder(seqlen=seqlen, hw=num_patch, embed_dim=d_model, stride_short=stride_short,
                              n_layers=n_layers, short_n_layers=short_n_layers, num_head=num_head, dropout=dropout, drop_path_r=drop_path_r, 
                              atten_drop=atten_drop, mask_ratio=mask_ratio, device=device)
        
        ##########################
        # Aggregation
        ##########################
        self.d_local = d_local = d_model//2
        self.s_proj = nn.Linear(d_local, d_local)
        self.t_proj = nn.Linear(d_local, d_local)
        self.local_spa_atten = CrossAttention(d_local, num_heads=num_head, qk_scale=True, qkv_bias=None)
        self.local_tem_atten = CrossAttention(d_local, num_heads=num_head, qk_scale=True, qkv_bias=None)
        self.ffl1 = Mlp(in_features=d_local, hidden_features=d_local*2)
        self.ffl2 = Mlp(in_features=d_local, hidden_features=d_local*2)

        self.fusion = nn.Linear(196, 1)
        self.output_proj = nn.Linear(d_local, 2048)

        ##########################
        # KTD Regressor
        ##########################
        self.ktd_regressor = HSCR(d_local)
        self.apply(self._init_weights)

        ##########################
        # SPIN Backbone, Regressor (Pre-trained)
        ##########################
        self.spin_backbone, self.regressor = spin_backbone_init(device)
        self.patchfiy = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2)

        self.to(device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, is_train=False, J_regressor=None):
        """
        x : [B, T, 3, H, W]
        """
        ##########################
        # SPIN Backbone
        ##########################
        B = x.shape[0]
        x = torch.flatten(x, 0, 1)
        feat_map = self.spin_backbone(x) # [BT, d, H, W]
        global_st_feat = self.patchfiy(feat_map).flatten(-2).permute(0, -1, 1)       
        global_st_feat = global_st_feat.reshape(B, self.seqlen, self.num_patch, self.d_model)              # [B, T, N, d]

        ##########################
        # STBranch
        ##########################
        local_st_feat, global_temporal_feat, global_spatial_feat = self.stencoder(global_st_feat, is_train=is_train)

        ##########################
        # Aggregation
        ##########################
        proj_spatial_feat = self.s_proj(global_spatial_feat)
        proj_temporal_feat = self.t_proj(global_temporal_feat)

        local_st_feat = torch.flatten(local_st_feat, 1, 2)
        local_st_feat = self.local_spa_atten(local_st_feat, proj_spatial_feat)
        local_st_feat = self.ffl1(local_st_feat)
        local_st_feat = self.local_tem_atten(local_st_feat, proj_temporal_feat)             # [B, tn, d/2]
        local_st_feat = self.ffl2(local_st_feat)

        local_st_feat = local_st_feat.reshape(B, self.stride_short*2 + 1, self.d_local, -1) # [B, t, d/2, n]
        print(local_st_feat.shape)
        local_t_feat = self.fusion(local_st_feat).reshape(B, self.stride_short*2 + 1, -1)   # [B, t, 256]
        global_t_feat = self.output_proj(local_t_feat)                                      # [B, t, 2048]

        ##########################
        # Regressor
        ##########################
        _, pred_global = self.regressor(global_t_feat, is_train=is_train, J_regressor=J_regressor, n_iter=3)

        if not is_train:
            feature = local_t_feat[:, self.mid_frame][:, None, :] 
        else:
            feature = local_t_feat
        smpl_output = self.ktd_regressor(feature, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)

        if not is_train:    # Eval
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)         
                s['verts'] = s['verts'].reshape(B, -1, 3)      
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        else:
            size = self.stride_short * 2 + 1
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)           # [B, size, 10]
                s['verts'] = s['verts'].reshape(B, size, -1, 3)        # [B, size, 6980]
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)        # [B, size, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)        # [B, size, 3]
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)   # [B, size, 3, 3]

        return smpl_output

        
        




