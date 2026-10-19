import torch
from fuse_channels import ChannelSet


def make_channel_set(C_list, N, D, n_markers=16, seed=0, with_identity=True):
    """Build a ChannelSet with full (unmasked) channel sets."""
    g = torch.Generator().manual_seed(seed)
    tokens, sample_idx, channel_idx, patch_idx, channel_id, identity = [], [], [], [], [], []
    for b, C in enumerate(C_list):
        for c in range(C):
            cid = int(torch.randint(0, n_markers, (1,), generator=g).item())
            ident = torch.randn(D, generator=g)
            for n in range(N):
                tokens.append(torch.randn(D, generator=g) + ident)
                sample_idx.append(b)
                channel_idx.append(c)
                patch_idx.append(n)
                channel_id.append(cid)
                identity.append(ident)
    tokens = torch.stack(tokens)
    cs = ChannelSet(
        tokens=tokens,
        sample_idx=torch.tensor(sample_idx),
        channel_idx=torch.tensor(channel_idx),
        patch_idx=torch.tensor(patch_idx),
        channel_id=torch.tensor(channel_id),
        n_samples=len(C_list),
        n_patches=N,
        identity=torch.stack(identity) if with_identity else None,
    )
    cs.seq_len = torch.bincount(cs.sample_idx, minlength=len(C_list))
    return cs


def permute_channels(cs):
    """Return a copy with channel order shuffled within each sample."""
    import random

    rng = random.Random(0)
    tokens, sample_idx, channel_idx, patch_idx, channel_id, identity = [], [], [], [], [], []
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
                identity.append(cs.identity[t] if cs.identity is not None else None)
    new = ChannelSet(
        tokens=torch.stack(tokens),
        sample_idx=torch.tensor(sample_idx),
        channel_idx=torch.tensor(channel_idx),
        patch_idx=torch.tensor(patch_idx),
        channel_id=torch.tensor(channel_id),
        n_samples=cs.n_samples,
        n_patches=cs.n_patches,
        identity=torch.stack(identity) if cs.identity is not None else None,
    )
    new.seq_len = torch.bincount(new.sample_idx, minlength=cs.n_samples)
    return new
