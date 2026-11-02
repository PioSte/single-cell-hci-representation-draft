"""fuse-channels: padding-free channel-set fusion for HCI foundation models."""

from .config import (
    ABLATIONS,
    DEFAULT_CONFIG,
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    SharedCore,
    ablated_config,
    build_model,
    describe_ablations,
)
from .container import ChannelSet
from .crops import (
    CellCrop,
    CellCropDataset,
    CropGeometry,
    TiffStack,
    cell_n_tokens,
    collate_crops,
    patch_size_for_pixel_size,
    place_box,
    plan_patches,
    read_crop_tensors,
    token_coverage,
    token_coverage_torch,
    write_crop_cache,
)
from .decoder import (
    ChannelAwareDecoder,
    DecoderBlock,
    canonical_patch_size_for,
)
from .encoder import TransformerBlock, TransformerEncoder
from .fleet import (
    FLEET_PIXEL_SIZES,
    canonical_fleet_pixel_size,
    fleet_patch_sizes,
    sorted_fleet_pixel_sizes,
)
from .flexi import pi_resize_patch_embed
from .fusion import (
    ChannelFusion,
    IdentityGatedSum,
    MultiheadAttentionPooling,
)
from .identity import ChannelEmbedding
from .losses import (
    cha_maevit_recon_loss,
    normalized_patch_mse,
    patch_offsets,
    patch_row_terms,
)
from .mae import ChannelFusionMAE
from .masking import DCPVisibilitySampler, HierarchicalChannelSampler
from .ops import (
    grouped_attention,
    grouped_cross_attention,
    grouped_pool_attention,
    grouped_projected_self_attention,
    grouped_self_attention,
    grouped_softmax,
)
from .patch_embed import ChannelTiedPatchEmbed, ChannelTokenizer
from .pos_embed import (
    POS_EMBED_KINDS,
    FractionalSpatialPE,
    add_spatial_pos,
    build_spatial_pos,
    fractional_1d_sincos,
    get_2d_sincos_pos_embed,
    get_fractional_sincos_pos_embed,
)
from .rope import (
    apply_rope,
    axial_rope_cache,
    identity_rope_cache,
    normalized_grid_coords,
    rope_frequencies,
)
from .tokenizer import ChannelFusionTokenizer

__all__ = [
    "ABLATIONS",
    "DEFAULT_CONFIG",
    "FLEET_PIXEL_SIZES",
    "POS_EMBED_KINDS",
    "CellCrop",
    "CellCropDataset",
    "ChannelAwareDecoder",
    "ChannelEmbedding",
    "ChannelFusion",
    "ChannelFusionMAE",
    "ChannelFusionTokenizer",
    "ChannelSet",
    "ChannelTiedPatchEmbed",
    "ChannelTokenizer",
    "CropGeometry",
    "DCPVisibilitySampler",
    "DecoderBlock",
    "DecoderConfig",
    "EncoderConfig",
    "FractionalSpatialPE",
    "HierarchicalChannelSampler",
    "IdentityGatedSum",
    "ModelConfig",
    "MultiheadAttentionPooling",
    "SharedCore",
    "TiffStack",
    "TransformerBlock",
    "TransformerEncoder",
    "ablated_config",
    "add_spatial_pos",
    "apply_rope",
    "axial_rope_cache",
    "build_model",
    "build_spatial_pos",
    "canonical_fleet_pixel_size",
    "canonical_patch_size_for",
    "cell_n_tokens",
    "cha_maevit_recon_loss",
    "collate_crops",
    "describe_ablations",
    "fleet_patch_sizes",
    "fractional_1d_sincos",
    "get_2d_sincos_pos_embed",
    "get_fractional_sincos_pos_embed",
    "grouped_attention",
    "grouped_cross_attention",
    "grouped_pool_attention",
    "grouped_projected_self_attention",
    "grouped_self_attention",
    "grouped_softmax",
    "identity_rope_cache",
    "normalized_grid_coords",
    "normalized_patch_mse",
    "patch_offsets",
    "patch_row_terms",
    "patch_size_for_pixel_size",
    "pi_resize_patch_embed",
    "place_box",
    "plan_patches",
    "read_crop_tensors",
    "rope_frequencies",
    "sorted_fleet_pixel_sizes",
    "token_coverage",
    "token_coverage_torch",
    "write_crop_cache",
]

__version__ = "0.1.0"
