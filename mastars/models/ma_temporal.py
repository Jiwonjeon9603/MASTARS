from typing import Tuple

import einops
import torch
from torch import nn

from .temporal import TemporalSelfAttention, TemporalUnet


class SharedConvAttentionDeconv(nn.Module):
    """Multi-agent denoiser (MADiff architecture).

    A single parameter-shared temporal U-Net processes every agent's trajectory
    independently; agents exchange information through attention layers applied
    at the bottleneck and at every skip connection of the decoder.
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        n_agents: int,
        dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.returns_condition = returns_condition

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.net = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            condition_dropout=condition_dropout,
            kernel_size=kernel_size,
        )

        def attention(channels):
            return TemporalSelfAttention(
                channels,
                channels // 16,
                channels // 4,
                embed_dim=self.net.embed_dim,
                residual=residual_attn,
            )

        # one attention block at the bottleneck, then one per decoder skip connection
        self.self_attn = nn.ModuleList(
            [attention(in_out[-1][1])] + [attention(d_out) for _, d_out in reversed(in_out)]
        )

    def forward(self, x, time, returns=None, use_dropout: bool = True, force_dropout: bool = False):
        """
        x       : [batch x horizon x agent x transition]
        time    : [batch]
        returns : [batch x 1 x agent]
        """
        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )
        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.net.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        if self.returns_condition:
            assert returns is not None
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.net.returns_mlp(returns)
            if use_dropout:
                # the same dropout mask is shared by all agents
                mask = self.net.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        def flat(z):  # b a ... -> (b a) ...
            return z.reshape(z.shape[0] * z.shape[1], *z.shape[2:])

        def unflat(z):  # (b a) ... -> b a ...
            return z.reshape(bs, z.shape[0] // bs, *z.shape[1:])

        h = []
        x, t_flat = flat(x), flat(t)
        for resnet, resnet2, downsample in self.net.downs:
            x = resnet(x, t_flat)
            x = resnet2(x, t_flat)
            h.append(x)
            x = downsample(x)

        x = self.net.mid_block1(x, t_flat)
        x = self.net.mid_block2(x, t_flat)

        x = flat(self.self_attn[0](unflat(x), t))

        for layer_idx, (resnet, resnet2, upsample) in enumerate(self.net.ups):
            hiddens = flat(self.self_attn[layer_idx + 1](unflat(h.pop()), t))
            x = torch.cat((x, hiddens), dim=1)
            x = resnet(x, t_flat)
            x = resnet2(x, t_flat)
            x = upsample(x)

        x = unflat(self.net.final_conv(x))
        return einops.rearrange(x, "b a f t -> b t a f")
