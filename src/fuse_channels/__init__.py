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
from .losses import normalized_patch_mse
from .mae import ChannelFusionMAE
from .masking import DCPVisibilitySampler
from .ops import (
    grouped_cross_attention,
    grouped_pool_attention,
    grouped_self_attention,
    grouped_softmax,
)
from .patch_embed import ChannelTiedPatchEmbed, ChannelTokenizer

__all__ = [
    "ChannelSet",
    "ChannelFusion",
    "IdentityGatedSum",
    "MultiheadAttentionPooling",
    "ChannelEmbedding",
    "ChannelTokenizer",
    "ChannelTiedPatchEmbed",
    "TransformerEncoder",
    "TransformerBlock",
    "ChannelAwareDecoder",
    "DecoderBlock",
    "DCPVisibilitySampler",
    "ChannelFusionMAE",
    "normalized_patch_mse",
    "grouped_cross_attention",
    "grouped_pool_attention",
    "grouped_self_attention",
    "grouped_softmax",
]

__version__ = "0.1.0"
