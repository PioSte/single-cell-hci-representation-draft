"""fuse-channels: padding-free channel-set fusion for HCI foundation models."""

from .container import ChannelSet
from .decoder import ChannelAwareDecoder, DecoderBlock
from .encoder import SpatialTransformer, TransformerBlock
from .fusion import (
    ChannelFusion,
    HypernetworkWeightedSum,
    MultiheadAttentionPooling,
    dense_fusion_reference,
)
from .identity import ChannelIdentity
from .losses import per_channel_normalized_mse
from .mae import ChannelFusionMAE
from .masking import DCPMasker, visible_to_masked
from .ops import (
    grouped_cross_attention,
    grouped_pool_attention,
    grouped_self_attention,
    segment_softmax,
)
from .patch_embed import ChannelFrontend, ChannelTiedPatchEmbed

__all__ = [
    "ChannelSet",
    "ChannelFusion",
    "HypernetworkWeightedSum",
    "MultiheadAttentionPooling",
    "dense_fusion_reference",
    "ChannelIdentity",
    "ChannelFrontend",
    "ChannelTiedPatchEmbed",
    "SpatialTransformer",
    "TransformerBlock",
    "ChannelAwareDecoder",
    "DecoderBlock",
    "DCPMasker",
    "visible_to_masked",
    "ChannelFusionMAE",
    "per_channel_normalized_mse",
    "grouped_cross_attention",
    "grouped_pool_attention",
    "grouped_self_attention",
    "segment_softmax",
]

__version__ = "0.1.0"
