import os
import torch
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR
from lib.models.transformer_global import Transformer

class GMM(nn.Module):
    def __init__(self,
                 seqlen,
                 n_layers=3,
                 d_model=512,
                 num_head=8, 
                 dropout=0., 
                 drop_path_r=0.,
                 atten_drop=0.,
                 mask_ratio=0.,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 ):
        super(GMM, self).__init__()
        self.seqlen = seqlen
        self.mask_ratio = mask_ratio

        self.trans = Transformer(depth=n_layers, embed_dim=d_model, 
                    mlp_hidden_dim=d_model*4, h=num_head, drop_rate=dropout, 
                    drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=seqlen)
        
    
    def forward(self, x, is_train=False):
        """
        """
        if is_train:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(x, mask_flag=True, mask_ratio=self.mask_ratio)
        else:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(x, mask_flag=False, mask_ratio=0.)
        pred = self.trans.forward_decoder(mem, ids_restore)  # [N, L, p*p*3]

        if is_train:
            output = pred
        else:
            output = pred[:, self.seqlen // 2][:, None, :]
        
        return output