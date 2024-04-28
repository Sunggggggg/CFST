import numpy as np
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange

from lib.models.smpl import SMPL_MEAN_PARAMS
from lib.models.trans_operator import Block
from lib.models.pooling import *

class MaskTransformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, head=8, 
                 drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16, device=torch.device("cuda")) :
        super().__init__()
        mean_params = np.load(SMPL_MEAN_PARAMS)
        self.init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).to(device)
        self.init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0).to(device)
        self.init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).to(device)

        self.mask_token_mlp = nn.Sequential(
            nn.Linear(24 * 6 + 13, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim // 2)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Encoder
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=head, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)  
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim // 2))
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim // 2, num_heads=head, mlp_hidden_dim=embed_dim * 2, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.decoder_norm = norm_layer(embed_dim // 2)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1) # assgin value from ids_restore
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_flag=False, mask_ratio=0.):
        # x = x + self.pos_embed
        if mask_flag:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        if ids_restore is not None:
            mean_pose = torch.cat((self.init_pose, self.init_shape, self.init_cam), dim=-1)
            mask_tokens = self.mask_token_mlp(mean_pose)
            mask_tokens = mask_tokens.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        else:
            x_ = x
        x = x_ + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def forward(self, x, is_train=True, mask_ratio=0.):
        if is_train:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=True, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        else:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=False,mask_ratio=0.)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask
    
class STtransformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, head=8, 
                 drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0.) :
        super().__init__()
        self.depth = depth
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=head, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=head, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

    def SpaTemHead(self, x, spatial_pos, temporal_pos):
        b, t, n, c = x.shape
        x = rearrange(x, 'b t n c  -> (b t) n c')
        x = x + spatial_pos
        x = self.pos_drop(x)
        spablock = self.SpatialBlocks[0]
        x = spablock(x)
        x = self.norm_s(x)
        
        x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
        x = x + temporal_pos
        x = self.pos_drop(x)
        temblock = self.TemporalBlocks[0]
        x = temblock(x)
        x = self.norm_t(x)

        return x

    def forward(self, x, spatial_pos, temporal_pos):
        """
        x : [B, t, n, d]
        """
        b, t, n, c = x.shape
        x = self.SpaTemHead(x, spatial_pos, temporal_pos)

        for i in range(1, self.depth):
            SpaAtten = self.SpatialBlocks[i]
            TemAtten = self.TemporalBlocks[i]
            x = rearrange(x, '(b n) t c -> (b t) n c', n=n)
            x = SpaAtten(x)
            x = self.norm_s(x)
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
            x = TemAtten(x)
            x = self.norm_t(x)

        x = rearrange(x, '(b n) t c -> b t n c', n=n)
        return x

class STencoder(nn.Module) :
    def __init__(self, 
                 seqlen, 
                 hw, 
                 embed_dim=512,
                 stride_short=4,
                 n_layers=2,
                 short_n_layers=3,
                 num_head=8,
                 dropout=0., 
                 drop_path_r=0.,
                 atten_drop=0.,
                 mask_ratio=0.,
                 device=torch.device("cuda")
                 ):
        super().__init__()
        self.mid_frame = int(seqlen // 2)
        self.stride_short = stride_short
        self.mask_ratio = mask_ratio
        
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, hw, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, seqlen, embed_dim))

        self.temporal_trans = MaskTransformer(depth=n_layers, embed_dim=embed_dim, 
                            mlp_hidden_dim=embed_dim*2, head=num_head, 
                            drop_rate=dropout, drop_path_rate=drop_path_r, 
                            attn_drop_rate=atten_drop, length=seqlen, device=device)
        
        self.spatial_trans = MaskTransformer(depth=n_layers, embed_dim=embed_dim, 
                            mlp_hidden_dim=embed_dim*2, head=num_head, 
                            drop_rate=dropout, drop_path_rate=drop_path_r, 
                            attn_drop_rate=atten_drop, length=hw, device=device)
        
        self.st_trans = STtransformer(depth=short_n_layers, embed_dim=embed_dim, 
                            mlp_hidden_dim=embed_dim*2, head=num_head, 
                            drop_rate=dropout, drop_path_rate=drop_path_r, 
                            attn_drop_rate=atten_drop)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim//2)
        

    def forward(self, x, is_train=False) :
        """
        x : [B, T, N, d]
        """
        ###############################
        # Global aggregation
        ###############################
        global_spatial_feat = temporal_pool(x)      # [B, N, d]
        global_temporal_feat = sptial_pool(x)       # [B, T, d]

        global_spatial_feat = global_spatial_feat + self.spatial_pos_embed
        global_temporal_feat = global_temporal_feat + self.temporal_pos_embed

        global_temporal_feat, temporal_mask = self.temporal_trans(global_temporal_feat, is_train=is_train, mask_ratio=self.mask_ratio) # [B, T, d]
        global_spatial_feat, spatial_mask = self.spatial_trans(global_spatial_feat, is_train=is_train, mask_ratio=self.mask_ratio)     # [B, N, d]

        ###############################
        # Local st-transformer
        ###############################
        x_local = x[:, self.mid_frame - self.stride_short:self.mid_frame + self.stride_short + 1, 0::self.stride_short] # [B, t, n, d]
        spatial_pos_embed_local = self.spatial_pos_embed[:, 0::self.stride_short]                                       # [B, n, d]
        temporal_pos_embed_local = self.temporal_pos_embed[:, self.mid_frame - self.stride_short:self.mid_frame + self.stride_short + 1] # [B, t, d]

        local_st_feat = self.st_trans(x_local, spatial_pos_embed_local, temporal_pos_embed_local) # [B, t, n, d]
        local_st_feat = self.out_proj(local_st_feat)

        return local_st_feat, global_temporal_feat, global_spatial_feat




