import math
from typing import Tuple

import einops
import torch
import torch.nn as nn
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from .helpers import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d


class TemporalSelfAttention(nn.Module):
    """Attention across agents; queries/keys/values are conditioned on the
    diffusion-time embedding. Input / output: [batch x agent x channel x horizon]."""

    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int,
        v_n_channels: int,
        embed_dim: int,
        nheads: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.nheads = nheads

        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)

        def time_mlp(out_channels):
            return nn.Sequential(
                nn.Mish(),
                nn.Linear(embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )

        self.query_time_mlp = time_mlp(qk_n_channels * nheads)
        self.key_time_mlp = time_mlp(qk_n_channels * nheads)
        self.value_time_mlp = time_mlp(v_n_channels * nheads)

        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, time):
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        time = rearrange(time, "b a f -> (b a) f")
        query = self.query_layer(x_flat) + self.query_time_mlp(time)
        key = self.key_layer(x_flat) + self.key_time_mlp(time)
        value = self.value_layer(x_flat) + self.value_time_mlp(time)

        split = lambda z: rearrange(z, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1])
        query, key, value = split(query), split(key), split(value)

        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(query.shape[-1])
        attn = self.attend(dots)
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")

        out = rearrange(out, "h b a f -> b a (h f)").reshape(x.shape)
        if self.residual:
            out = x + self.gamma * out
        return out


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """x: [batch x inp_channels x horizon], t: [batch x embed_dim]"""
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """1-D temporal U-Net over a single agent's trajectory, conditioned on the
    diffusion timestep and (optionally, with classifier-free dropout) the return."""

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")
        act_fn = nn.Mish()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        embed_dim = dim

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        if returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - condition_dropout)
            embed_dim += dim
        self.embed_dim = embed_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim, kernel_size),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim, kernel_size),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, kernel_size),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim, kernel_size),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        """x: [batch x horizon x transition], returns: [batch x 1]"""
        x = einops.rearrange(x, "b t f -> b f t")
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1))
                returns_embed = mask.to(returns_embed.device) * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b f t -> b t f")
