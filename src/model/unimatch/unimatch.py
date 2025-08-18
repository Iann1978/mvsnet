from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from .backbone import CNNEncoder
from einops import rearrange, repeat
import torch
from .transformer import FeatureTransformer
from .matching import correlation_softmax_depth

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
        
        x0,x1 = self.transformer(x0, x1, attn_type='swin', attn_num_splits=2) # [B, (h w), C]
        # nx1 = self.transformer(x1, x0, attn_type='none') # [B, (h w), C]
        # print('after transformer')    
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        # x0 = rearrange(nx0, 'b (h w) c -> b c h w', h=H, w=W)
        # x1 = rearrange(nx1, 'b (h w) c -> b c h w', h=H, w=W)
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        # construct depth candidates [B, D, H, W]
        intrinsics = x['intrinsics'][:,0]
        extrinsics0 = x['extrinsics'][:,0]
        extrinsics1 = x['extrinsics'][:,1]
        pose = torch.inverse(extrinsics0) @ extrinsics1
        depth_candidates = torch.arange(500, 1000, 10, device=x['images'].device, dtype=x['images'].dtype)
        depth_candidates = repeat(depth_candidates, 'd -> b d h w', b=B, h=H, w=W)
        # print('intrinsics:', intrinsics.shape)
        # print('pose:', pose.shape)
        # print('depth_candidates:', depth_candidates.shape)

        # print('depth_candidates:', depth_candidates.shape)
        # exit()
        depth, match_prob = correlation_softmax_depth(x0, x1, intrinsics, pose, depth_candidates)
        # print('depth:', depth.shape)
        # print('match_prob:', match_prob.shape)
        depth = self.upsample(depth)
        depth = self.upsample(depth)
        depth = self.upsample(depth)
        return depth.unsqueeze(1)
        exit()
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

