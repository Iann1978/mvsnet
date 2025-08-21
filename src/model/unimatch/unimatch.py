from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from .backbone import CNNEncoder
from einops import rearrange, repeat
import torch
from .transformer import FeatureTransformer
from .matching import correlation_softmax_depth, warp_with_pose_depth_candidates
from .attention import SelfAttnPropagation

@dataclass
class UniMatchConfig(BaseModelConfig):
    configu: str
    feature_number: int = 128
    depth_candidates: int = 192
    depth_min: int = 425
    depth_max: int = 910

class UniMatch(BaseModel):
    def __init__(self, cfg: UniMatchConfig):
        super().__init__(cfg)
        self.backbone = CNNEncoder(output_dim=cfg.feature_number,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 return_all_scales=False,
                 )
        # self.transformer = nn.Transformer(d_model=128, nhead=2, num_encoder_layers=6, num_decoder_layers=6)
        self.transformer = FeatureTransformer(num_layers=6, d_model=cfg.feature_number, nhead=1, ffn_dim_expansion=4)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=cfg.feature_number)
        self.conv1 = nn.Conv2d(cfg.feature_number*2, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        print(self.cfg.configu) 

    def forward(self, x: BatchedViews):
        # get x0, x1
        D = self.cfg.depth_candidates
        B, V, C, H, W = x['images'].shape
        # print('B, V, C, H, W:', B, V, C, H, W)
        x0 = x['images'][:,0]
        x1 = x['images'][:,1]
        assert x0.shape == (B, C, H, W), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, C, H, W), f'x1.shape: {x1.shape}'

        
        # get features of x0, x1 through backbone(cnn)
        x0 = self.backbone(x0)[0] # [B, C, H, W]
        x1 = self.backbone(x1)[0] # [B, C, H, W]
        assert x0.shape == (B, self.cfg.feature_number, H//8, W//8), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, self.cfg.feature_number, H//8, W//8), f'x1.shape: {x1.shape}'

        # # enhance features of x0, x1 through transformer
        x0,x1 = self.transformer(x0, x1, attn_type='swin', attn_num_splits=2) 
        assert x0.shape ==  (B, self.cfg.feature_number, H//8, W//8), f'x0.shape: {x0.shape}'
        assert x1.shape ==  (B, self.cfg.feature_number, H//8, W//8), f'x1.shape: {x1.shape}'

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

        depth = self.feature_flow_attn(x0, depth)
        assert depth.shape == (B, 1, H//8, W//8), f'depth.shape: {depth.shape}'




        # upsample depth
        depth = self.upsample(depth)
        depth = self.upsample(depth)
        depth = self.upsample(depth)
        # depth = self.conv2(depth)
        depth = depth.unsqueeze(1)
        assert depth.shape == (B, 1, 1, H, W), f'depth.shape: {depth.shape}'

        return depth



    # def scale_intrinsics(self, intrinsics, scale_x, scale_y):
