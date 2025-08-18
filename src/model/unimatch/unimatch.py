from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from .backbone import CNNEncoder
from einops import rearrange
import torch
from .transformer import FeatureTransformer

@dataclass
class UniMatchConfig(BaseModelConfig):
    configu: str

class UniMatch(BaseModel):
    def __init__(self, cfg: UniMatchConfig):
        super().__init__(cfg)
        self.backbone = CNNEncoder(output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 return_all_scales=False,
                 )
        # self.transformer = nn.Transformer(d_model=128, nhead=2, num_encoder_layers=6, num_decoder_layers=6)
        self.transformer = FeatureTransformer(num_layers=6, d_model=128, nhead=1, ffn_dim_expansion=4)
        self.conv1 = nn.Conv2d(128*2, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        print(self.cfg.configu) 

    def forward(self, x: BatchedViews):
        B, V, C, H, W = x['images'].shape
        # print(x['images'].shape)
        x0 = self.backbone(x['images'][:,0])[0] # [B, C, H, W]
        x1 = self.backbone(x['images'][:,1])[0] # [B, C, H, W]
        B, C, H, W = x0.shape
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        # x0 = rearrange(x0, 'b c h w -> b (h w) c')
        # x1 = rearrange(x1, 'b c h w -> b (h w) c')
        # print('before transformer')
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        
        nx0,nx1 = self.transformer(x0, x1, attn_type='swin', attn_num_splits=2) # [B, (h w), C]
        # nx1 = self.transformer(x1, x0, attn_type='none') # [B, (h w), C]
        # print('after transformer')
        # print('nx0:', nx0.shape)
        # print('nx1:', nx1.shape)
        # x0 = rearrange(nx0, 'b (h w) c -> b c h w', h=H, w=W)
        # x1 = rearrange(nx1, 'b (h w) c -> b c h w', h=H, w=W)
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        x = torch.cat([x0, x1], dim=1)
        # print('x:', x.shape)

        x = self.conv1(x)
        # print('x:', x.shape)
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.upsample(x)
        # print('x:', x.shape)
        # exit()
        return x.unsqueeze(1)

