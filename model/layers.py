"""Building blocks of PinFuse: positional-induced attention (PiT), the
advection-diffusion transport operator (PiTFlow) and the mixture prior."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_adaptive_layer_norm(x, cond, norm, proj):
    if cond is None:
        return x

    shift, scale = proj(cond).chunk(2, dim=-1)
    x = norm(x)
    if shift.ndim == x.ndim - 1:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    elif shift.ndim != x.ndim:
        raise ValueError(
            f"Unsupported modulation shapes: x={tuple(x.shape)}, shift={tuple(shift.shape)}"
        )
    return x * (1 + scale) + shift


class PiTLayer(nn.Module):
    """Single-head PiT layer: attention weights are a learned function of distance."""

    def __init__(self, in_features, out_features, cond_dim=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.v_proj = nn.Linear(in_features, in_features, bias=False)
        self.lams = nn.Parameter(torch.ones(1))

        if cond_dim is not None:
            self.ada_ln1 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
            self.ada_proj1 = nn.Linear(cond_dim, in_features * 2)
            self.ada_ln2 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
            self.ada_proj2 = nn.Linear(cond_dim, in_features * 2)
        else:
            self.ada_ln1 = self.ada_proj1 = self.ada_ln2 = self.ada_proj2 = None

        self.out_proj = nn.Linear(in_features * 2, out_features)

    def forward(self, x, distance, mask, cond=None, residual=True):
        # x: (B, N, in) on the source grid; distance/mask: (B, M, N)
        x_mod1 = _apply_adaptive_layer_norm(x, cond, self.ada_ln1, self.ada_proj1)
        x_mod2 = _apply_adaptive_layer_norm(x, cond, self.ada_ln2, self.ada_proj2)

        attn = distance * torch.tan(0.25 * np.pi * (1 - 1e-7) * (1.0 + torch.sin(self.lams)))
        inf = torch.full_like(attn, torch.finfo(attn.dtype).max)
        attn = torch.where(mask, attn, inf)
        attn = torch.softmax(-attn, dim=-2)                 # (B, M, N)
        # A fully masked column comes out uniform; zero it explicitly.
        attn = attn * mask.to(attn.dtype)
        h_out = F.gelu(attn @ self.v_proj(x_mod1))          # (B, M, in)

        if not residual:
            h_res = h_out
        else:
            if h_out.shape[1] != x_mod2.shape[1]:
                raise ValueError(
                    f"residual attention needs one grid: got M={h_out.shape[1]}, "
                    f"N={x_mod2.shape[1]}. Pass residual=False for cross-grid input."
                )
            h_res = x_mod2
        return self.out_proj(torch.cat([h_out, h_res], dim=-1))  # (B, M, out)


class PiTMultiheadLayer(nn.Module):
    """Multi-head PiT layer, one learned distance scale per head."""

    def __init__(self, in_features, out_features, num_heads=4, cond_dim=None):
        super().__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.v_proj = nn.Linear(in_features, out_features)
        self.lams = nn.Parameter(torch.ones(num_heads))

        if cond_dim is not None:
            self.ada_ln1 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
            self.ada_proj1 = nn.Linear(cond_dim, in_features * 2)
            self.ada_ln2 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
            self.ada_proj2 = nn.Linear(cond_dim, in_features * 2)
        else:
            self.ada_ln1 = self.ada_proj1 = self.ada_ln2 = self.ada_proj2 = None

        self.res_proj = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)

    def forward(self, x, distance, mask, cond=None, residual=True):
        # x: (B, N, in) on the source grid; distance/mask: (B, M, N)
        B, N, _ = x.shape
        _, M, N_d = distance.shape
        if N_d != N:
            raise ValueError(f"distance last dim ({N_d}) must match source tokens ({N})")
        if residual and M != N:
            raise ValueError(
                f"residual attention needs one grid: got M={M}, N={N}. "
                f"Pass residual=False for cross-grid input."
            )

        x_mod1 = _apply_adaptive_layer_norm(x, cond, self.ada_ln1, self.ada_proj1)
        x_mod2 = _apply_adaptive_layer_norm(x, cond, self.ada_ln2, self.ada_proj2)

        lam_scale = torch.tan(
            0.25 * np.pi * (1 - 1e-7) * (1.0 + torch.sin(self.lams))
        ).view(1, self.num_heads, 1, 1)
        attn = distance.unsqueeze(1) * lam_scale            # (B, H, M, N)

        inf = torch.full_like(attn, torch.finfo(attn.dtype).max)
        attn = torch.where(mask.unsqueeze(1), attn, inf)
        attn = torch.softmax(-attn, dim=-1)
        # A fully masked row comes out uniform; zero it explicitly.
        attn = attn * mask.unsqueeze(1).to(attn.dtype)

        v = self.v_proj(x_mod1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        heads_out = torch.matmul(attn, v)                   # (B, H, M, head_dim)
        multihead_out = heads_out.transpose(1, 2).contiguous().view(B, M, self.out_features)

        if residual:
            multihead_out = F.gelu(multihead_out + self.res_proj(x_mod2))
        else:
            multihead_out = F.gelu(multihead_out)
        return self.out_proj(multihead_out)


class PiTFlow(nn.Module):
    """Column-stochastic transport operator ``flow = diffusion @ advection``.

    * diffusion: heat kernel ``softmax(-D^2 * coef)`` with a learned per-node
      diffusivity ``coef = {softplus|exp}(S) + diff_min``.
    * advection: attention ``softmax(K Q^T / sqrt(H) / T)`` over destinations.

    ``qk_norm`` layer-normalizes Q/K; ``advect_temp`` sets the softmax
    temperature (``None``: T=1 with qk_norm, T=10 without; ``'learn'``: a trained
    scalar floored at ``advect_temp_min``).
    """

    def __init__(self, in_features, hidden_dim=20, lam=0.0, use_dst=False,
                 mask_diffusion=False, qk_norm=False, diff_min=0.0,
                 advect_temp=None, advect_temp_init=1.0,
                 advect_temp_min=0.05, diff_kernel='exp'):
        super().__init__()
        self.in_features = in_features

        self.q_proj = PiTLayer(in_features, hidden_dim)   # advection, source side
        self.k_proj = PiTLayer(in_features, hidden_dim)   # advection, destination side
        self.s_proj = PiTLayer(in_features, 1)            # diffusivity

        self.advect_temp = advect_temp
        self.learn_temp = isinstance(advect_temp, str) and advect_temp == 'learn'
        self.advect_temp_min = float(advect_temp_min)
        if self.learn_temp:
            self.advect_temp = None
            init = float(advect_temp_init) - self.advect_temp_min
            # inverse softplus, so softplus(raw) + min == advect_temp_init
            raw = math.log(math.expm1(init)) if init > 1e-6 else -6.0
            self.advect_temp_raw = nn.Parameter(torch.tensor(float(raw)))
        self.scale = hidden_dim ** 0.5
        self.use_dst = use_dst
        self.mask_diffusion = mask_diffusion
        self.qk_norm = qk_norm
        self.diff_min = float(diff_min)
        if diff_kernel not in ('softplus', 'exp'):
            raise ValueError(f"diff_kernel must be 'softplus' or 'exp', got {diff_kernel!r}")
        self.diff_kernel = str(diff_kernel)

        self.gamma = nn.Parameter(torch.ones(1) * 0.0)
        self.lam = torch.nn.Parameter(torch.ones(1) * lam)

    def forward(
        self,
        dst,
        src,
        src_dst_distance,
        src_dst_mask,
        src_distance=None,
        src_mask=None,
        dst_distance=None,
        dst_mask=None,
        src_valid=None,
        residual=True,
    ):
        """
        dst, src : (B, Nd, F), (B, Ns, F). ``dst`` only supplies the shape unless
            ``use_dst`` is set, in which case it gives the destination-side key.
        src_valid : (B, Ns) bool, padding mask of the source tokens (no locality cut).
        returns flow (B, Nd, Ns), diffusive (B, Nd), diffusion, advection, extra
        """
        B, Nd, _ = dst.shape
        _, Ns, _ = src.shape

        def _default_self_graph(n_tokens, device, dtype):
            D = torch.zeros((B, n_tokens, n_tokens), device=device, dtype=dtype)
            M = torch.ones((B, n_tokens, n_tokens), device=device, dtype=torch.bool)
            return D, M

        # Square self-grid call (PinFuse_Cell): unmasked advection.
        self_grid = (
            src_distance is None and src_mask is None
            and dst_distance is None and dst_mask is None
        )

        if src_distance is None or src_mask is None:
            if src_dst_distance.shape[-2] == Ns and src_dst_distance.shape[-1] == Ns:
                src_distance = src_dst_distance
                src_mask = src_dst_mask
            else:
                src_distance, src_mask = _default_self_graph(Ns, src.device, src.dtype)

        if dst_distance is None or dst_mask is None:
            if src_dst_distance.shape[-2] == Nd and src_dst_distance.shape[-1] == Nd:
                dst_distance = src_dst_distance
                dst_mask = src_dst_mask
            else:
                dst_distance, dst_mask = _default_self_graph(Nd, dst.device, dst.dtype)

        extra = {}
        Q = self.q_proj(src, src_distance, src_mask)                                  # (B, Ns, H)
        if self.use_dst:
            K = self.k_proj(dst, dst_distance, dst_mask)                              # (B, Nd, H)
        else:
            K = self.k_proj(src, src_dst_distance, src_dst_mask, residual=residual)   # (B, Nd, H)
        S = self.s_proj(src, src_dst_distance, src_dst_mask, residual=residual)       # (B, Nd, 1)

        # Bounded for numerical safety (exp overflows float32 near 88.7).
        diffusive = torch.clamp(S.reshape(B, Nd), max=20.0)
        extra["diffusive"] = diffusive.detach()

        # Diffusion: heat-kernel Green's function, column-normalized.
        D2 = dst_distance ** 2
        diffuse_coef = (torch.exp(diffusive) if self.diff_kernel == 'exp'
                        else F.softplus(diffusive)) + self.diff_min
        if self.diff_min or self.diff_kernel != 'softplus':
            extra["diffuse_coef"] = diffuse_coef.detach()
        diffusion_logits = -D2 * diffuse_coef[:, None, :]
        diffusion_logits = torch.clamp(diffusion_logits, min=-10, max=10)
        if self.mask_diffusion:
            # Masked after the clamp, to a finite value so empty columns stay finite.
            diffusion_logits = diffusion_logits.masked_fill(~dst_mask, -1e4)
        diffusion = F.softmax(diffusion_logits, dim=-2)

        # Advection.
        if self.qk_norm:
            Q = F.layer_norm(Q, Q.shape[-1:])
            K = F.layer_norm(K, K.shape[-1:])
        if self_grid:
            affinity = (Q @ K.transpose(-1, -2)) / self.scale
            logits = torch.clamp(affinity, min=-10, max=10)
        else:
            affinity = (K @ Q.transpose(-1, -2)) / self.scale    # (B, Nd, Ns)
            # Column-centering is exact under the dim=-2 softmax and keeps the clamp meaningful.
            logits = affinity - affinity.amax(dim=-2, keepdim=True).detach()
            logits = torch.clamp(logits, min=-10, max=10)
            logits = logits.masked_fill(~src_dst_mask, -1e4)

        if self.learn_temp:
            t = F.softplus(self.advect_temp_raw) + self.advect_temp_min
            advection = F.softmax(logits / t, dim=-2)
            extra["advect_temp"] = t.detach()
        else:
            temp = self.advect_temp
            if temp is None:
                temp = 1.0 if self.qk_norm else 10.0
            advection = F.softmax(logits if float(temp) == 1.0 else logits / float(temp), dim=-2)

        # Padding source columns carry no mass.
        if src_valid is not None:
            keep_src = src_valid.reshape(B, 1, Ns).to(advection.dtype)
            advection = advection * keep_src
            affinity = affinity * keep_src
        extra["affinity"] = affinity.detach()

        flow = diffusion @ advection
        return flow, diffusive, diffusion, advection, extra

    def prior(self, pos, sigma2):
        """Gaussian-kernel prior on the transport, used by the Pyro model."""
        eye = torch.eye(pos.shape[1], device=pos.device).unsqueeze(0)
        distance = torch.cdist(pos, pos, p=2)
        attn_logits = -torch.clamp((distance ** 2) / (2 * sigma2), max=20.0)
        attn_logits = attn_logits + F.softplus(self.lam) * eye
        return F.softmax(attn_logits, dim=-2)


class MoGPrior(nn.Module):
    """Mixture-of-Gaussians prior over the latent space (resolVI-style init)."""

    def __init__(self, latent_dim, num_cats=50):
        super().__init__()
        self.K = num_cats
        self.latent_dim = latent_dim
        self.mu = nn.Parameter(torch.randn(self.K, latent_dim))
        self.logvar = nn.Parameter(torch.full((self.K, latent_dim), -2.0))
        self.pi = nn.Parameter(torch.zeros(self.K))   # mixture logits

    def forward(self):
        return self.mu, self.logvar, self.pi
