"""fuse-channels: padding-free channel-set fusion for HCI foundation models."""

from .container import ChannelSet
from .crops import (
    CellCrop,
    cell_n_tokens,
    CellCropDataset,
    CropGeometry,
    TiffStack,
    collate_crops,
    patch_size_for_pixel_size,
    place_box,
    plan_patches,
    read_crop_tensors,
    token_coverage,
    token_coverage_torch,
    write_crop_cache,
)
from .fleet import (
    FLEET_PIXEL_SIZES,
    canonical_fleet_pixel_size,
    fleet_patch_sizes,
    sorted_fleet_pixel_sizes,
)
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
from .decoder import (
    ChannelAwareDecoder,
    DecoderBlock,
    canonical_patch_size_for,
)
from .encoder import TransformerBlock, TransformerEncoder
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
from .flexi import pi_resize_patch_embed
from .patch_embed import ChannelTiedPatchEmbed, ChannelTokenizer
from .tokenizer import ChannelFusionTokenizer
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

__all__ = [
    "ChannelSet",
    "CellCrop",
    "cell_n_tokens",
    "CellCropDataset",
    "CropGeometry",
    "TiffStack",
    "collate_crops",
    "patch_size_for_pixel_size",
    "place_box",
    "plan_patches",
    "read_crop_tensors",
    "token_coverage",
    "token_coverage_torch",
    "write_crop_cache",
    "ChannelFusion",
    "IdentityGatedSum",
    "MultiheadAttentionPooling",
    "ChannelEmbedding",
    "ChannelTokenizer",
    "ChannelFusionTokenizer",
    "ChannelTiedPatchEmbed",
    "POS_EMBED_KINDS",
    "FractionalSpatialPE",
    "add_spatial_pos",
    "build_spatial_pos",
    "fractional_1d_sincos",
    "get_2d_sincos_pos_embed",
    "get_fractional_sincos_pos_embed",
    "apply_rope",
    "axial_rope_cache",
    "identity_rope_cache",
    "normalized_grid_coords",
    "rope_frequencies",
    "pi_resize_patch_embed",
    "TransformerEncoder",
    "TransformerBlock",
    "ChannelAwareDecoder",
    "canonical_patch_size_for",
    "FLEET_PIXEL_SIZES",
    "canonical_fleet_pixel_size",
    "fleet_patch_sizes",
    "sorted_fleet_pixel_sizes",
    "DecoderBlock",
    "DCPVisibilitySampler",
    "HierarchicalChannelSampler",
    "ChannelFusionMAE",
    "normalized_patch_mse",
    "patch_offsets",
    "patch_row_terms",
    "cha_maevit_recon_loss",
    "grouped_attention",
    "grouped_cross_attention",
    "grouped_pool_attention",
    "grouped_projected_self_attention",
    "grouped_self_attention",
    "grouped_softmax",
]

__version__ = "0.1.0"
