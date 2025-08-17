from typing_extensions import TypedDict
from jaxtyping import Float, Int64
from torch import Tensor

class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    images: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width

class UnBatchedViews(TypedDict, total=False):
    intrinsics: Float[Tensor, "V 3 3"]
    extrinsics: Float[Tensor, "V 4 4"]
    imgs: Float[Tensor, "V 3 H W"]
    targets: Float[Tensor, "1 H W"]