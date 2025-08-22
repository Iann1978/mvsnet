from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from ..unimatch.backbone import CNNEncoder
from jaxtyping import Float, Int64
from torch import Tensor
import torch
from .ldm_unet.unet import UNetModel
from ..unimatch.transformer import FeatureTransformer
from ..unimatch.matching import warp_with_pose_depth_candidates, correlation_softmax_depth
from einops import repeat

@dataclass
class MVSPlatConfig(BaseModelConfig):
    configm: str
    feature_number: int = 128
    depth_candidates: int = 192
    depth_min: int = 425
    depth_max: int = 910

class MVSPlat(BaseModel):
    def __init__(self, cfg: MVSPlatConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.backbone = CNNEncoder(output_dim=cfg.feature_number,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 return_all_scales=False,
                 )
        self.transformer = FeatureTransformer(num_layers=6,
                                              d_model=cfg.feature_number,
                                              nhead=1,
                                              ffn_dim_expansion=4)
        self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        input_channels = cfg.feature_number*2+1+cfg.depth_candidates
        channels = cfg.feature_number
        num_views = 2
        num_depth_candidates = cfg.depth_candidates
        costvolume_unet_attn_res = ()
        costvolume_unet_channel_mult = (1,1,1)

        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=costvolume_unet_attn_res,
                channel_mult=costvolume_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
        ]
        self.corr_refine_net = nn.Sequential(*modules)
        self.conv1 = nn.Conv2d(num_depth_candidates, 1, kernel_size=3, stride=1, padding=1);
        print(self.cfg.configm)

    def forward(self, x: BatchedViews) -> Float[Tensor, "batch _ 1 _ _"]:

        # print('MVSPlat.forward:')
        # get x0, x1
        D = self.cfg.depth_candidates
        B, V, C, H, W = x['images'].shape
        F = self.cfg.feature_number
        print('B, V, C, F, H, W:', B, V, C, F, H, W)
        x0 = x['images'][:,0]
        x1 = x['images'][:,1]
        assert x0.shape == (B, C, H, W), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, C, H, W), f'x1.shape: {x1.shape}'

        # get features of x0, x1 through backbone(cnn)
        x0 = self.backbone(x0)[0] # [B, C, H, W]
        x1 = self.backbone(x1)[0] # [B, C, H, W]
        assert x0.shape == (B, F, H//8, W//8), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, F, H//8, W//8), f'x1.shape: {x1.shape}'

        # # enhance features of x0, x1 through transformer
        x0,x1 = self.transformer(x0, x1, attn_type='swin', attn_num_splits=2) 
        assert x0.shape ==  (B, F, H//8, W//8), f'x0.shape: {x0.shape}'
        assert x1.shape ==  (B, F, H//8, W//8), f'x1.shape: {x1.shape}'

                # get warpping parameters
        intrinsics = x['intrinsics'][:,0]
        intrinsics[:,0,0] = intrinsics[:,0,0] / 2.0
        intrinsics[:,1,1] = intrinsics[:,1,1] / 2.0
        intrinsics[:,0,2] = intrinsics[:,0,2] / 2.0
        intrinsics[:,1,2] = intrinsics[:,1,2] / 2.0
        extrinsics0 = x['extrinsics'][:,0]
        extrinsics1 = x['extrinsics'][:,1]
        pose = extrinsics1 @ torch.inverse(extrinsics0)
        depth_candidates = torch.linspace(self.cfg.depth_min, self.cfg.depth_max, D, device=x['images'].device, dtype=x['images'].dtype)
        depth_candidates = repeat(depth_candidates, 'd -> b d h w', b=B, h=H//8, w=W//8)
        assert depth_candidates.shape == (B, D, H//8, W//8), f'depth_candidates.shape: {depth_candidates.shape}'

        # save warped feature1
        self.warped_feature1 = warp_with_pose_depth_candidates(x1, intrinsics, pose, depth_candidates)
        assert self.warped_feature1.shape == (B, 128, D, H//8, W//8), f'self.warped_feature1.shape: {self.warped_feature1.shape}'

        # depth estimation
        depth, match_prob = correlation_softmax_depth(x0, x1, intrinsics, pose, 1.0/depth_candidates)
        depth = 1.0/depth
        assert depth.shape == (B, 1, H//8, W//8), f'depth.shape: {depth.shape}'
        assert match_prob.shape == (B, D, H//8, W//8), f'match_prob.shape: {match_prob.shape}'

        x0 = self.upsample2x(x0)
        x1 = self.upsample2x(x1)
        depth = self.upsample2x(depth)
        match_prob = self.upsample2x(match_prob)
        assert x0.shape == (B, F, H//4, W//4), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, F, H//4, W//4), f'x1.shape: {x1.shape}'
        assert depth.shape == (B, 1, H//4, W//4), f'depth.shape: {depth.shape}'
        assert match_prob.shape == (B, D, H//4, W//4), f'match_prob.shape: {match_prob.shape}'

        x = torch.cat([x0, x1, depth, match_prob], dim=1)  
        assert x.shape == (B, F+F+1+D, H//4, W//4), f'x.shape: {x.shape}'

        x = self.corr_refine_net(x) # [B, D, H//4, W//4]
        assert x.shape == (B, D, H//4, W//4), f'x.shape: {x.shape}'







        x = self.conv1(x)
        assert x.shape == (B, 1, H//4, W//4), f'x.shape: {x.shape}'

        x = self.upsample4x(x)
        assert x.shape == (B, 1, H, W), f'x.shape: {x.shape}'



        x = x.unsqueeze(1)
        assert x.shape == (B, 1, 1, H, W), f'x.shape: {x.shape}'

        return x