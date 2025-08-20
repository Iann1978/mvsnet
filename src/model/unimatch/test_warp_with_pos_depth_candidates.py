print('test_warp_with_pos_depth_candidates')

from src.dataset.dtu import DTUDataset, DTUDatasetConfig
from src.model.unimatch.matching import warp_with_pose_depth_candidates
import torch
from einops import repeat
import torch.nn.functional as F


def test_homography_warp():


    view_number = 4
    cfg = DTUDatasetConfig(root='./datasets/dtu', view_number=view_number, type='DTU')
    dataset = DTUDataset(cfg, stage='val')
    unbatched_views = dataset[0]
    intrinsics = unbatched_views['intrinsics']
    extrinsics = unbatched_views['extrinsics']
    imgs = unbatched_views['imgs']
    depth = unbatched_views['targets']

    h, w = 512, 640
    img_to_warp = imgs[1].unsqueeze(0)
    img_to_warp = F.interpolate(img_to_warp, size=(h, w), mode='bilinear', align_corners=True)
    intrinsics_to_warp = intrinsics[0]
    intrinsics_to_warp[0,0] = intrinsics_to_warp[0,0] * 4
    intrinsics_to_warp[1,1] = intrinsics_to_warp[1,1] * 4
    intrinsics_to_warp[0,2] = intrinsics_to_warp[0,2] * 4
    intrinsics_to_warp[1,2] = intrinsics_to_warp[1,2] * 4
    intrinsics_to_warp = intrinsics_to_warp.unsqueeze(0)
    extrinsics0 = extrinsics[0]
    extrinsics1 = extrinsics[1]
    extrinsics_to_warp = extrinsics1 @ torch.inverse(extrinsics0)
    pose = extrinsics_to_warp.unsqueeze(0)
    depth_candidates = torch.linspace(500, 1000, 4, device=imgs.device, dtype=imgs.dtype)
    depth_candidates = repeat(depth_candidates, 'd -> b d h w', b=1, h=h, w=w)
    warped_feature1 = warp_with_pose_depth_candidates(img_to_warp, intrinsics_to_warp, pose, depth_candidates)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    axs = axs.flatten()
    for i in range(4):
        axs[i].imshow(imgs[i].cpu().permute(1, 2, 0))
        axs[i].set_title(f"Img {i+1}")
        axs[i].axis('off')
    axs[4].imshow(warped_feature1[0,:,0].cpu().permute(1, 2, 0))
    axs[4].set_title(f"Depth")
    axs[4].axis('off')
    axs[5].imshow(warped_feature1[0,:,1].cpu().permute(1, 2, 0))
    axs[5].set_title(f"Depth")
    axs[5].axis('off')
    axs[6].imshow(warped_feature1[0,:,2].cpu().permute(1, 2, 0))
    axs[6].set_title(f"Depth")
    axs[6].axis('off')
    axs[7].imshow(warped_feature1[0,:,3].cpu().permute(1, 2, 0))
    axs[7].set_title(f"Depth")
    axs[7].axis('off')



    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_homography_warp()