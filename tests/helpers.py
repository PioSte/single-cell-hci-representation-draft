import torch
from fuse_channels import ChannelSet


def make_channel_set(C_list, N, D, n_channel_ids=16, seed=0, with_channel_embed=True):
    """Build a ChannelSet with full (unmasked) channel sets."""
    g = torch.Generator().manual_seed(seed)
    tokens, sample_idx, channel_idx, patch_idx, channel_id = [], [], [], [], []
    channel_embed, channel_group_id = [], []
    chan_global = 0
    for b, C in enumerate(C_list):
        for c in range(C):
            cid = int(torch.randint(0, n_channel_ids, (1,), generator=g).item())
            embed = torch.randn(D, generator=g)
            for n in range(N):
                tokens.append(torch.randn(D, generator=g))
                sample_idx.append(b)
                channel_idx.append(c)
                patch_idx.append(n)
                channel_id.append(cid)
                channel_embed.append(embed)
                channel_group_id.append(chan_global)
            chan_global += 1
    tokens = torch.stack(tokens)
    return ChannelSet(
        tokens=tokens,
        sample_idx=torch.tensor(sample_idx),
        channel_idx=torch.tensor(channel_idx),
        patch_idx=torch.tensor(patch_idx),
        channel_id=torch.tensor(channel_id),
        n_samples=len(C_list),
        n_patches=N,
        channel_embed=torch.stack(channel_embed) if with_channel_embed else None,
        channel_group_id=torch.tensor(channel_group_id),
    )


def permute_channels(cs):
    """Return a copy with channel order shuffled within each sample."""
    import random

    rng = random.Random(0)
    tokens, sample_idx, channel_idx, patch_idx, channel_id = [], [], [], [], []
    channel_embed, channel_group_id = [], []
    for b in range(cs.n_samples):
        sel = (cs.sample_idx == b).nonzero(as_tuple=True)[0]
        # group by channel
        chans = sorted(set(cs.channel_idx[sel].tolist()))
        rng.shuffle(chans)
        for new_c, old_c in enumerate(chans):
            sub = sel[cs.channel_idx[sel] == old_c]
            perm = sub[torch.argsort(cs.patch_idx[sub])]
            for t in perm.tolist():
                tokens.append(cs.tokens[t])
                sample_idx.append(b)
                channel_idx.append(new_c)
                patch_idx.append(int(cs.patch_idx[t]))
                channel_id.append(int(cs.channel_id[t]))
                channel_embed.append(
                    cs.channel_embed[t] if cs.channel_embed is not None else None
                )
                channel_group_id.append(
                    int(cs.channel_group_id[t])
                    if cs.channel_group_id is not None
                    else -1
                )
    return ChannelSet(
        tokens=torch.stack(tokens),
        sample_idx=torch.tensor(sample_idx),
        channel_idx=torch.tensor(channel_idx),
        patch_idx=torch.tensor(patch_idx),
        channel_id=torch.tensor(channel_id),
        n_samples=cs.n_samples,
        n_patches=cs.n_patches,
        channel_embed=(
            torch.stack(channel_embed) if cs.channel_embed is not None else None
        ),
        channel_group_id=(
            torch.tensor(channel_group_id) if cs.channel_group_id is not None else None
        ),
    )
