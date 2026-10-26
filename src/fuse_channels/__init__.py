"""fuse-channels: padding-free channel-set fusion for HCI foundation models."""

from .container import ChannelSet
from .decoder import ChannelAwareDecoder, DecoderBlock
from .encoder import TransformerBlock, TransformerEncoder
from .fusion import (
    ChannelFusion,
    IdentityGatedSum,
    MultiheadAttentionPooling,
)
from .identity import ChannelEmbedding
from .losses import cha_maevit_recon_loss, normalized_patch_mse
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
    add_spatial_pos,
    build_spatial_pos,
    get_2d_sincos_pos_embed,
)

__all__ = [
    "ChannelSet",
    "ChannelFusion",
    "IdentityGatedSum",
    "MultiheadAttentionPooling",
    "ChannelEmbedding",
    "ChannelTokenizer",
    "ChannelTiedPatchEmbed",
    "POS_EMBED_KINDS",
    "add_spatial_pos",
    "build_spatial_pos",
    "get_2d_sincos_pos_embed",
    "TransformerEncoder",
    "TransformerBlock",
    "ChannelAwareDecoder",
    "DecoderBlock",
    "DCPVisibilitySampler",
    "HierarchicalChannelSampler",
    "ChannelFusionMAE",
    "normalized_patch_mse",
    "cha_maevit_recon_loss",
    "grouped_attention",
    "grouped_cross_attention",
    "grouped_pool_attention",
    "grouped_projected_self_attention",
    "grouped_self_attention",
    "grouped_softmax",
]

__version__ = "0.1.0"
