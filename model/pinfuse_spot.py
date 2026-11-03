"""PinFuse at the joint cell + out-of-cell (spot / molecule) level.

Two-stage training:

* **Stage 1** -- train :class:`~model.pinfuse_cell.PinFuse_Cell` on the cells alone.
* **Stage 2** -- freeze it inside :class:`PinFuse_Spot` and train on the joint
  object (``data.type = 'celllevel'``), learning only how mass leaves the cells.

The frozen model supplies the cell latent, the denoised cell rate
``px_rate_cell`` and the cell -> cell transport ``beta_cc``. The wrapper encodes
the molecule grid, predicts a per-cell, per-gene leak ``w`` and learns a
cell -> all transport ``beta_ac``. With the default ``beta_mix = 'leak'``::

    rate_mol  = px_rate_cell * w
    rate_cell = px_rate_cell - rate_mol
    recon     = px_prop * (beta_cc @ rate_cell) + (1 - px_prop) * rate_cell
                + beta_ac @ rate_mol

Other ``beta_mix`` modes:

* ``'compose'``       -- denoise at the cell level first, then leak the result:
  ``C = px_prop * (beta_cc @ R) + (1 - px_prop) * R``, ``recon = beta_ac @ (w C) + (1 - w) C``.
* ``'leak_molecule'`` -- as ``'leak'`` but the leaked half only reaches molecule rows.
* ``'retrain_prop'``  -- ``beta_ac`` replaces ``beta_cc``: ``recon = (1 - q) (beta_ac @ R) + q R``,
  ``q = 1 - px_prop`` on cells, 0 on molecules (no leak head; use ``lam5 = 0``).
* ``'retrain_leak'``  -- ``recon = beta_ac @ (w R) + (1 - w) R``.
* ``'renorm'``        -- parameter-free baseline ``colnorm(beta_cc + beta_ac) @ R``.

All tensors live on one dense grid of ``Na`` tokens (seed at index 0); cells
(``in_tissue == 2``) and molecules are told apart by masks. The frozen decoder is
given the joint (cell + molecule) mass budget.
"""

import contextlib
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pinfuse_cell import PinFuse_Cell
from model.layers import PiTLayer, PiTMultiheadLayer, PiTFlow, MoGPrior

SingleLayer = PiTLayer
MultiLayer = PiTMultiheadLayer


def _masked_quantile(distance, src_mask, q):
    """Per-row quantile of ``distance`` over the masked source columns only. Returns (B, M, 1)."""
    if bool(src_mask.all()):
        return torch.quantile(distance, q, dim=-1, keepdim=True)

    B, M, _ = distance.shape
    big = torch.finfo(distance.dtype).max
    d = distance.masked_fill(~src_mask.unsqueeze(-2), big)
    d, _ = torch.sort(d, dim=-1)

    n = src_mask.sum(dim=-1).clamp_min(1)
    pos = q * (n - 1).to(distance.dtype)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    w = (pos - lo.to(distance.dtype)).view(B, 1, 1)
    lo = lo.view(B, 1, 1).expand(B, M, 1)
    hi = hi.view(B, 1, 1).expand(B, M, 1)
    v_lo = d.gather(-1, lo)
    v_hi = d.gather(-1, hi)
    return v_lo + (v_hi - v_lo) * w


def _masked_graph(distance, dst_mask, src_mask, locality):
    """Locality mask between two token subsets of one dense grid."""
    t = _masked_quantile(distance, src_mask, locality)
    return (distance <= t) & src_mask.unsqueeze(-2) & dst_mask.unsqueeze(-1)


def _col_normalize(beta, valid_dst, valid_src=None, eps=1e-8):
    """Zero the excluded destination rows, renormalize columns, then zero excluded source columns."""
    beta = beta * valid_dst.unsqueeze(-1).to(beta.dtype)
    beta = beta / beta.sum(dim=-2, keepdim=True).clamp_min(eps)
    if valid_src is not None:
        beta = beta * valid_src.unsqueeze(-2).to(beta.dtype)
    return beta


class MoleculeEncoder(nn.Module):
    """Molecule-grid encoder; also gives the cells context about surrounding molecules."""

    def __init__(self, in_feats, hidden_dim, latent_dim, n_heads=8):
        super().__init__()
        emb_dim = 32
        self.tissue_emb = nn.Embedding(3, emb_dim)

        self.enc_h1 = MultiLayer(in_feats + emb_dim, hidden_dim, n_heads)
        self.enc_h2 = MultiLayer(hidden_dim, hidden_dim, n_heads)
        self.enc_cross = MultiLayer(hidden_dim, hidden_dim, n_heads)   # unused (ablated)
        self.enc_ctx = MultiLayer(hidden_dim, hidden_dim, n_heads)     # molecules -> cells

        self.enc_mu = SingleLayer(hidden_dim, latent_dim)
        self.enc_logvar = SingleLayer(hidden_dim, latent_dim)

    def forward(self, genes_mol, in_tissue_mol, h_cell, graphs):
        d_mm, m_mm = graphs['mm']
        d_c2m, m_c2m = graphs['c2m']

        emb = self.tissue_emb(in_tissue_mol.flatten().long())
        x = torch.cat([genes_mol, emb.reshape(genes_mol.shape[0], genes_mol.shape[1], -1)], dim=2)

        h = F.gelu(self.enc_h1(x, d_mm, m_mm))
        h_mol = h + F.gelu(self.enc_h2(h, d_mm, m_mm))
        h_cell_ctx = h_cell + F.gelu(self.enc_ctx(h_mol, d_c2m, m_c2m))

        mu = self.enc_mu(h_mol, d_mm, m_mm)
        logvar = self.enc_logvar(h_mol, d_mm, m_mm)

        return h_mol, h_cell_ctx, mu, logvar


class LeakHead(nn.Module):
    """Per-cell (per-gene) logit of the share of cell mass that leaves the cell."""

    def __init__(self, in_feats, hidden_dim, n_heads=8, gene_leak=True):
        super().__init__()
        self.leak = SingleLayer(hidden_dim, in_feats if gene_leak else 1)

    def forward(self, h_cell_ctx, distance_cell, mask_cell):
        return self.leak(h_cell_ctx, distance_cell, mask_cell)


class MoleculeNBEncoder(nn.Module):
    """Count latent of the molecule tokens (cells keep the frozen model's)."""

    def __init__(self, in_feats, hidden_dim, latent_dim_nb=5, n_heads=8):
        super().__init__()
        self.enc_h1_nb = MultiLayer(in_feats, hidden_dim, n_heads)
        self.enc_h2_nb = MultiLayer(hidden_dim, hidden_dim, n_heads)
        self.enc_mu_nb = SingleLayer(hidden_dim, latent_dim_nb)
        self.enc_logvar_nb = SingleLayer(hidden_dim, latent_dim_nb)

    def forward(self, genes_mol, distance_mol, mask_mol):
        h = F.gelu(self.enc_h1_nb(genes_mol, distance_mol, mask_mol))
        h = h + F.gelu(self.enc_h2_nb(h, distance_mol, mask_mol))
        mu_nb = self.enc_mu_nb(h, distance_mol, mask_mol)
        logvar_nb = self.enc_logvar_nb(h, distance_mol, mask_mol)
        return h, mu_nb, logvar_nb


class MoleculeDecoderNB(nn.Module):
    """ZINB dispersion and dropout of the molecule tokens."""

    def __init__(self, in_feats, hidden_dim, latent_dim_nb=5, n_heads=8,
                 gene_disper=False, gene_dropout=True):
        super().__init__()
        self.dec_h1 = MultiLayer(latent_dim_nb, hidden_dim, n_heads)
        self.dec_h2 = MultiLayer(hidden_dim, hidden_dim, n_heads)
        self.px_r_decoder = SingleLayer(hidden_dim, in_feats if gene_disper else 1)
        self.px_dropout_decoder = SingleLayer(hidden_dim, in_feats if gene_dropout else 1)

    def forward(self, z_nb_mol, distance_mol, mask_mol):
        h = F.gelu(self.dec_h1(z_nb_mol, distance_mol, mask_mol))
        h = h + F.gelu(self.dec_h2(h, distance_mol, mask_mol))
        px_r = F.softplus(self.px_r_decoder(h, distance_mol, mask_mol))
        px_r = torch.clamp(px_r, min=0.0, max=1e6)
        px_dropout = self.px_dropout_decoder(h, distance_mol, mask_mol)
        return px_r, px_dropout


class PinFuse_Spot(nn.Module):
    """Frozen :class:`PinFuse_Cell` wrapper for joint cell + out-of-cell denoising."""

    VERSION = "v8.2"
    MAX_LOGVAR = 10.0
    BETA_MIX = ('leak', 'compose', 'renorm', 'leak_molecule',
                'retrain_prop', 'retrain_leak')

    def __init__(
        self,
        in_feats,
        hidden_dim,
        latent_dim,
        latent_dim_nb=5,
        train_beta=True,
        n_step=1,
        gene_disper=False,
        gene_dropout=True,
        locality=0.5,
        stage1_locality=None,
        gvae=None,
        freeze_gvae=True,
        beta_mix='leak',
        gene_leak=True,
        share_prior=True,
        flow_dst=True,
        mask_diffusion=True,
        qk_norm=True,
        diff_min=0.0,
        advect_temp=None,
        advect_temp_init=1.0,
        advect_temp_min=0.05,
        mol_latent=False,
    ):
        """
        flow_dst : the cell -> all transport reads destination features, carrying
            the joint latent (the only path from the molecule latent to ``recon``).
        mask_diffusion : restrict the transport's diffusion kernel to the locality graph.
        qk_norm / diff_min / advect_temp* : see :class:`~model.layers.PiTFlow` (stage-2 operator only).
        mol_latent : also feed the frozen stage-1 encoder's latent of the
            molecule rows to the transport destinations (requires ``flow_dst``).
        stage1_locality : stage-1 locality, for bare state-dict checkpoints that do not carry it.
        """
        super().__init__()
        if beta_mix not in self.BETA_MIX:
            raise ValueError(f"beta_mix must be one of {self.BETA_MIX}, got {beta_mix!r}")

        self.locality = locality
        self.beta_mix = beta_mix
        self.train_beta = train_beta
        self.n_step = max(1, int(n_step))
        if self.n_step != 1:
            logging.warning(f"PinFuse_Spot ignores model.n_step={self.n_step}")
        self.latent_dim = latent_dim
        self.latent_dim_nb = latent_dim_nb
        self.gene_disper = gene_disper
        self.gene_dropout = gene_dropout
        self.gene_leak = gene_leak
        self.eps = 1e-6

        # Stage-1 model; built untrained so load_pretrained can be called after.
        self.gvae = gvae if gvae is not None else PinFuse_Cell(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            latent_dim_nb=latent_dim_nb,
            train_beta=train_beta,
            gene_disper=gene_disper,
            gene_dropout=gene_dropout,
            locality=locality,
        )
        self.stage1_locality = stage1_locality
        if stage1_locality is not None:
            self.gvae.locality = float(stage1_locality)

        self.molecule_encoder = MoleculeEncoder(in_feats, hidden_dim, latent_dim)
        self.leak_head = LeakHead(in_feats, hidden_dim, gene_leak=gene_leak)
        self.encoder_nb = MoleculeNBEncoder(in_feats, hidden_dim, latent_dim_nb=latent_dim_nb)
        self.decoder_nb = MoleculeDecoderNB(
            in_feats, hidden_dim,
            latent_dim_nb=latent_dim_nb,
            gene_disper=gene_disper,
            gene_dropout=gene_dropout,
        )
        # The only channel from the frozen model into the trainable part.
        self.cell_proj = nn.Linear(latent_dim, hidden_dim)

        self.flow_dst = flow_dst
        self.mask_diffusion = mask_diffusion
        self.qk_norm = qk_norm
        self.diff_min = float(diff_min)
        self.advect_temp = None if advect_temp is None else (
            advect_temp if isinstance(advect_temp, str) else float(advect_temp))
        self.enc_beta2 = PiTFlow(hidden_dim, use_dst=flow_dst,
                                 mask_diffusion=mask_diffusion,
                                 qk_norm=qk_norm, diff_min=diff_min,
                                 advect_temp=self.advect_temp,
                                 advect_temp_init=float(advect_temp_init),
                                 advect_temp_min=float(advect_temp_min))
        if flow_dst:
            # Zero-init, so the latent pathway starts as a no-op.
            self.z_proj = nn.Linear(latent_dim, hidden_dim)
            nn.init.zeros_(self.z_proj.weight)
            nn.init.zeros_(self.z_proj.bias)

        self.mol_latent = bool(mol_latent)
        if self.mol_latent and not flow_dst:
            raise ValueError("model.stage2_mol_latent needs model.stage2_flow_dst")
        if self.mol_latent:
            self.z1_proj = nn.Linear(latent_dim, hidden_dim)
            nn.init.zeros_(self.z1_proj.weight)
            nn.init.zeros_(self.z1_proj.bias)

        # Sharing the frozen mixture keeps the molecule latent in the cell latent space.
        self.share_prior = share_prior
        self.prior = self.gvae.prior if share_prior else MoGPrior(latent_dim)

        self.freeze_gvae = False
        if freeze_gvae:
            self.freeze_gvae_()

    # ------------------------------------------------------------------ setup

    def freeze_gvae_(self):
        """Freeze the stage-1 model and keep it in eval mode."""
        self.freeze_gvae = True
        for p in self.gvae.parameters():
            p.requires_grad_(False)
        self.gvae.eval()
        return self

    def unfreeze_gvae_(self):
        self.freeze_gvae = False
        for p in self.gvae.parameters():
            p.requires_grad_(True)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_gvae:
            self.gvae.eval()
        return self

    def load_pretrained(self, ckpt, strict=True):
        """Load stage-1 weights from a checkpoint path / run dir, a ``PinFuse_Cell``, or a state dict.

        The stage-1 operator flags (locality, qk_norm, diff_kernel,
        advect_temp) are plain attributes, not weights, so they are restored from
        the checkpoint when it is a pickled module.
        """
        src = None
        if isinstance(ckpt, PinFuse_Cell):
            state = ckpt.state_dict()
            src = ckpt
        elif isinstance(ckpt, dict):
            state = ckpt.get('state_dict', ckpt)
        else:
            path = os.path.abspath(str(ckpt))
            if os.path.isdir(path):
                for candidate in (os.path.join(path, 'checkpoints', 'checkpoint.pth'),
                                  os.path.join(path, 'checkpoint.pth')):
                    if os.path.isfile(candidate):
                        path = candidate
                        break
                else:
                    raise FileNotFoundError(f"no checkpoint.pth found under {ckpt}")
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(loaded, dict):
                state = loaded.get('state_dict', loaded)
            else:
                state = loaded.state_dict()
                src = getattr(loaded, 'gvae', loaded)

        # A stage-2 checkpoint carries the whole wrapper; keep the stage-1 half.
        if any(k.startswith('gvae.') for k in state):
            state = {k[len('gvae.'):]: v for k, v in state.items() if k.startswith('gvae.')}

        flow = self.gvae.encoder.enc_beta
        if src is not None:
            src_flow = src.encoder.enc_beta
            restored = {
                'locality': float(src.locality),
                'qk_norm': bool(src_flow.qk_norm),
                'advect_temp': src_flow.advect_temp,
                'diff_kernel': src_flow.diff_kernel,
            }
            for name, value in restored.items():
                target = self.gvae if name == 'locality' else flow
                own = getattr(target, name)
                if own != value:
                    logging.warning(f"stage-1 {name}: built as {own!r}, checkpoint has {value!r}; "
                                    "using the checkpoint value")
                setattr(target, name, value)
        elif self.stage1_locality is None:
            raise ValueError(
                "the stage-1 source is a bare state dict, which carries no locality: "
                "set model.stage1_locality or pass a pickled checkpoint"
            )
        else:
            logging.warning("the stage-1 source is a bare state dict; the frozen operator "
                            "flags (qk_norm, diff_kernel, ...) are the class defaults")

        result = self.gvae.load_state_dict(state, strict=strict)
        if self.freeze_gvae:
            self.freeze_gvae_()
        logging.info(f"stage-1 weights loaded into {type(self).__name__}.gvae")
        return result

    # ----------------------------------------------------------- frozen branch

    def _build_all_graph(self, distance, in_tissue):
        threshold = torch.quantile(distance, self.locality, dim=-1, keepdim=True)
        return (distance <= threshold) | (in_tissue <= 1.0).reshape(in_tissue.shape[0], 1, -1)

    def _reparam(self, mu, logvar):
        logvar = logvar.clamp(min=-self.MAX_LOGVAR, max=self.MAX_LOGVAR)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu), logvar

    def _cell_branch(self, genes_cell, cell_mask, d_cc, m_cc, total_mol):
        """Run the frozen stage-1 model on the cell tokens of the dense grid.

        The cell half of the mass budget is ``genes_cell``'s total, dropout-
        corrected at eval time with the frozen decoder's own gate.
        """
        in_tissue_cell = torch.full(
            genes_cell.shape[:2] + (1,), 2.0,
            device=genes_cell.device, dtype=genes_cell.dtype,
        )
        ctx = torch.no_grad() if self.freeze_gvae else contextlib.nullcontext()
        with ctx:
            beta_cc, _, _, _, mu_c, logvar_c, mu_nb_c, logvar_nb_c, _ = self.gvae.encoder(
                genes_cell, in_tissue_cell, d_cc, m_cc,
            )
            z_c, logvar_c = self._reparam(mu_c, logvar_c)
            z_nb_c, _ = self._reparam(mu_nb_c, logvar_nb_c)

            if self.training:
                total_cell = genes_cell.sum(dim=(-1, -2), keepdim=True)
            else:
                _, _, _, _, px_dropout_probe, _ = self.gvae.decoder(
                    z_c, z_nb_c, genes_cell, d_cc, m_cc, valid=cell_mask,
                )
                gate = torch.sigmoid(px_dropout_probe).detach()
                corrected_cell = genes_cell / torch.clamp(1 - gate, min=1e-2, max=1.0)
                corrected_cell = corrected_cell * cell_mask.unsqueeze(-1)
                total_cell = corrected_cell.sum(dim=(-1, -2), keepdim=True)

            total = total_cell + total_mol
            _, _, px_r_cell, px_rate_cell, px_dropout_cell, px_prop_cell = self.gvae.decoder(
                z_c, z_nb_c, genes_cell, d_cc, m_cc, total=total, valid=cell_mask,
            )

        # Keep the budget on the cell rows only.
        px_rate_cell = px_rate_cell * cell_mask.unsqueeze(-1).to(px_rate_cell.dtype)
        if not bool(cell_mask.all()):
            got = px_rate_cell.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
            px_rate_cell = px_rate_cell * (total / got)

        return (_col_normalize(beta_cc, cell_mask, cell_mask), mu_c, logvar_c, px_rate_cell,
                px_r_cell, px_dropout_cell, px_prop_cell * cell_mask.unsqueeze(-1))

    def _stage1_encode_mol(self, genes_mol, in_tissue_mol, d_mm, m_mm):
        """Frozen stage-1 expression latent of the molecule rows (``in_tissue_mol`` must be 2.0)."""
        enc = self.gvae.encoder
        emb = enc.tissue_emb(in_tissue_mol.flatten().long())
        emb = emb.reshape(genes_mol.shape[0], genes_mol.shape[1], -1)
        x = torch.cat([genes_mol, emb], dim=2)

        h1 = F.gelu(enc.enc_h1(x, d_mm, m_mm))
        h = h1 + F.gelu(enc.enc_h2(h1, d_mm, m_mm))
        mu = enc.enc_mu(h, d_mm, m_mm)
        logvar = enc.enc_logvar(h, d_mm, m_mm)
        return mu, logvar

    # ---------------------------------------------------------------- forward

    def forward(self, genes, in_tissue, pos, mass=None):
        B, Na, G = genes.shape
        cell_mask = (in_tissue.squeeze(-1) == 2.0)
        mol_mask = ~cell_mask
        cm = cell_mask.unsqueeze(-1).to(genes.dtype)
        mm = mol_mask.unsqueeze(-1).to(genes.dtype)

        # One distance matrix; graphs differ only by their masks.
        distance = torch.cdist(pos, pos, p=2)
        mask_all = self._build_all_graph(distance, in_tissue)

        genes_cell = genes * cm
        genes_mol = genes * mm
        in_tissue_mol = in_tissue * mm

        # The frozen cell graph uses the stage-1 locality.
        m_cc = _masked_graph(distance, cell_mask, cell_mask, self.gvae.locality)
        m_mm = _masked_graph(distance, mol_mask, mol_mask, self.locality)
        m_m2c = _masked_graph(distance, mol_mask, cell_mask, self.locality)
        m_a2c = _masked_graph(distance, torch.ones_like(cell_mask), cell_mask, self.locality)
        graphs = {
            'cc': (distance, m_cc),
            'mm': (distance, m_mm),
            'm2c': (distance, m_m2c),
            'c2m': (distance.transpose(-1, -2), m_m2c.transpose(-1, -2)),
        }

        # Molecule-free batch: skip the molecule heads but keep the RNG draws aligned.
        no_mol = not bool(mol_mask.any())
        mu_mol1 = None

        # 1) Molecule count latent and nuisance parameters.
        if no_mol:
            mu_nb_mol = genes.new_zeros((B, Na, self.latent_dim_nb))
            logvar_nb_mol = genes.new_zeros((B, Na, self.latent_dim_nb))
            z_nb_mol, logvar_nb_mol = self._reparam(mu_nb_mol, logvar_nb_mol)
            px_r_mol = genes.new_zeros((B, Na, G if self.gene_disper else 1))
            px_dropout_mol = genes.new_zeros((B, Na, G if self.gene_dropout else 1))
        else:
            _, mu_nb_mol, logvar_nb_mol = self.encoder_nb(genes_mol, distance, m_mm)
            z_nb_mol, logvar_nb_mol = self._reparam(mu_nb_mol, logvar_nb_mol)
            px_r_mol, px_dropout_mol = self.decoder_nb(z_nb_mol, distance, m_mm)

        # 2) Molecule half of the joint mass budget (dropout-corrected at eval).
        counts_mol = genes_mol
        if not self.training:
            dropout_mol = torch.sigmoid(px_dropout_mol).detach()
            counts_mol = genes_mol / torch.clamp(1 - dropout_mol, min=1e-2, max=1.0)
        total_mol = counts_mol.sum(dim=(-1, -2), keepdim=True)

        # 3) Frozen cell branch.
        beta_cc, mu_c, logvar_c, px_rate_cell, px_r_cell, px_dropout_cell, px_prop_cell = self._cell_branch(
            genes_cell, cell_mask, distance, m_cc, total_mol,
        )

        # 4) Molecule branch, leak logits and the cell -> all transport.
        if no_mol:
            mu_mol = genes.new_zeros((B, Na, self.latent_dim))
            logvar_mol = genes.new_zeros((B, Na, self.latent_dim))
            z_mol, logvar_mol = self._reparam(mu_mol, logvar_mol)
            leak_logits = genes.new_zeros((B, Na, G if self.gene_leak else 1))
            beta_ac = genes.new_zeros((B, Na, Na))
            diffusive = genes.new_zeros((B, Na))
            diffuse = genes.new_zeros((B, Na, Na))
            advect = genes.new_zeros((B, Na, Na))
            extra_beta = {'diffusive': diffusive, 'affinity': genes.new_zeros((B, Na, Na))}
        else:
            h_cell = self.cell_proj(mu_c.detach() if self.freeze_gvae else mu_c)
            h_mol, h_cell_ctx, mu_mol, logvar_mol = self.molecule_encoder(
                genes_mol, in_tissue_mol, h_cell, graphs,
            )
            z_mol, logvar_mol = self._reparam(mu_mol, logvar_mol)
            leak_logits = self.leak_head(h_cell_ctx, distance, m_cc)

            if self.mol_latent:
                in_tissue_stage1 = torch.full(
                    genes_mol.shape[:2] + (1,), 2.0,
                    device=genes_mol.device, dtype=genes_mol.dtype,
                )
                with (torch.no_grad() if self.freeze_gvae else contextlib.nullcontext()):
                    mu_mol1, _ = self._stage1_encode_mol(genes_mol, in_tissue_stage1, distance, m_mm)

            # Destination features: cell features on cell rows, molecule features
            # on molecule rows, plus the joint latent under flow_dst.
            h_all = h_cell_ctx * cm + h_mol * mm
            if self.flow_dst:
                z_all = (mu_c.detach() if self.freeze_gvae else mu_c) * cm + z_mol * mm
                h_all = h_all + self.z_proj(z_all)
            if mu_mol1 is not None:
                h_all = h_all + self.z1_proj(mu_mol1) * mm
            beta_ac, diffusive, diffuse, advect, extra_beta = self.enc_beta2(
                h_all,
                h_cell_ctx,
                src_dst_distance=distance,
                src_dst_mask=m_a2c,
                src_distance=distance,
                src_mask=m_cc,
                dst_distance=distance,
                dst_mask=mask_all,
                src_valid=cell_mask,
            )

        # 5) ZINB nuisance parameters on the whole grid.
        px_r = px_r_cell * cm + px_r_mol * mm
        px_dropout = px_dropout_cell * cm + px_dropout_mol * mm

        # 6) Split the cell rate into retained / leaked parts and transport each.
        has_mol = mol_mask.any(dim=-1).reshape(B, 1, 1).to(px_rate_cell.dtype)
        beta_used = beta_ac          # regularized by lam3 / lam4
        beta_leak = beta_ac          # carries the leaked half
        has_leak = True
        if self.beta_mix == 'leak':
            prop_leak = torch.sigmoid(leak_logits) * cell_mask.unsqueeze(-1) * has_mol
            rate_mol = px_rate_cell * prop_leak
            rate_cell = px_rate_cell - rate_mol
            leak_base = px_rate_cell
            moved_cell = beta_cc @ rate_cell
            recon_cell = px_prop_cell * moved_cell + (1.0 - px_prop_cell) * rate_cell
            recon = recon_cell + beta_ac @ rate_mol
            share = prop_leak.mean(dim=-1).unsqueeze(-2)
            beta_all = beta_cc * (1.0 - share) + beta_ac * share
        elif self.beta_mix == 'compose':
            prop_leak = torch.sigmoid(leak_logits) * cell_mask.unsqueeze(-1) * has_mol
            moved_cell = beta_cc @ px_rate_cell
            cell_recon = px_prop_cell * moved_cell + (1.0 - px_prop_cell) * px_rate_cell
            rate_mol = cell_recon * prop_leak
            rate_cell = cell_recon - rate_mol
            recon = beta_ac @ rate_mol + rate_cell
            leak_base = cell_recon
            share = prop_leak.mean(dim=-1).unsqueeze(-2)
            beta_all = beta_cc * (1.0 - share) + beta_ac * share
        elif self.beta_mix == 'leak_molecule':
            # beta_ac restricted to molecule destinations, columns renormalized.
            beta_am = _col_normalize(beta_ac, mol_mask, cell_mask, self.eps)
            beta_leak = beta_am
            beta_used = beta_am
            prop_leak = torch.sigmoid(leak_logits) * cell_mask.unsqueeze(-1) * has_mol
            rate_mol = px_rate_cell * prop_leak
            rate_cell = px_rate_cell - rate_mol
            leak_base = px_rate_cell
            moved_cell = beta_cc @ rate_cell
            recon_cell = px_prop_cell * moved_cell + (1.0 - px_prop_cell) * rate_cell
            recon = recon_cell + beta_am @ rate_mol
            share = prop_leak.mean(dim=-1).unsqueeze(-2)
            beta_all = beta_cc * (1.0 - share) + beta_am * share
        elif self.beta_mix == 'retrain_leak':
            prop_leak = torch.sigmoid(leak_logits) * cell_mask.unsqueeze(-1) * has_mol
            rate_mol = px_rate_cell * prop_leak
            rate_cell = px_rate_cell - rate_mol
            recon = beta_ac @ rate_mol + rate_cell
            leak_base = px_rate_cell
            share = prop_leak.mean(dim=-1).unsqueeze(-2)
            beta_all = beta_ac * share
        elif self.beta_mix == 'retrain_prop':
            q_keep = (1.0 - px_prop_cell) * cm
            recon = ((1.0 - q_keep) * (beta_ac @ px_rate_cell) + q_keep * px_rate_cell)
            leak_base = px_rate_cell
            rate_cell = torch.zeros_like(px_rate_cell)
            rate_mol = px_rate_cell
            prop_leak = torch.zeros_like(px_rate_cell)
            beta_all = beta_ac
            beta_leak = None
            has_leak = False
        else:  # 'renorm'
            raw = beta_cc + beta_ac
            denom = raw.sum(dim=-2, keepdim=True).clamp_min(self.eps)
            beta_all = raw / denom
            recon = beta_all @ px_rate_cell
            share = beta_ac.sum(dim=-2, keepdim=True) / denom
            prop_leak = share.transpose(-1, -2).expand(B, share.shape[-1], G)
            leak_base = px_rate_cell

        px_rate = px_rate_cell
        if not self.train_beta:
            recon = px_rate
        px_proportion = px_prop_cell

        extra = {
            'cell_mask': cell_mask.detach(),
            'mask_c2a': m_a2c.detach(),
            # The trained latent is the molecule one.
            'valid_cell': mol_mask.detach(),
            'valid_mol': mol_mask.detach(),
            'valid_cell_token': cell_mask.detach(),
            'z': z_mol,
            'logvar': logvar_mol,
            'mu_nb': mu_nb_mol,
            'logvar_nb': logvar_nb_mol,
            'mu_cell': mu_c.detach(),
            'px_rate_cell': px_rate_cell.detach(),
            'beta_cell': beta_cc.detach(),
            'beta_transport': beta_ac.detach(),
            'beta_ac': beta_used,                 # with gradient, for the loss
            'beta_leak': None if beta_leak is None else beta_leak.detach(),
            'has_leak': has_leak,
            'leak': prop_leak,                    # with gradient, for loss6
            'leak_base': leak_base.detach(),
            'px_proportion': px_proportion.detach(),
        }
        extra.update(extra_beta)

        return recon, px_rate, px_r, px_dropout, mu_mol, beta_all, diffuse, advect, px_proportion, extra

    def forward_simple(self, genes, in_tissue, pos, mass=None):
        recon, px_rate, _, _, mu, _, _, _, _, extra = self.forward(genes, in_tissue, pos, mass=mass)
        cm = extra['cell_mask'].unsqueeze(-1).to(mu.dtype)
        latent = extra['mu_cell'] * cm + mu * (1.0 - cm)
        return recon, px_rate, latent

    def forward_posterior(self, genes, in_tissue, pos, mass=None):
        recon, px_rate, px_r, px_dropout, _, _, _, _, _, _ = self.forward(genes, in_tissue, pos, mass=mass)
        return recon, px_rate, px_r, px_dropout

    def forward_flow(self, genes, in_tissue, pos, mass=None):
        _, px_rate, _, _, _, beta, diffuse, advect, _, extra = self.forward(genes, in_tissue, pos, mass=mass)
        return px_rate, beta, extra['diffusive'], diffuse, advect, extra['px_proportion']

    def forward_flow_parts(self, genes, in_tissue, pos, mass=None):
        """The transport split into its cell -> cell and leaked (cell -> all) blocks."""
        (_, _, _, _, _, _, _, advect, _, extra) = self.forward(genes, in_tissue, pos, mass=mass)
        px_rate_cell = extra['px_rate_cell']
        leak_base = extra['leak_base']
        rate_mol = leak_base * extra['leak'].detach()
        rate_cell = leak_base - rate_mol
        if not extra['has_leak']:
            rate_cell = torch.zeros_like(px_rate_cell)
            rate_mol = px_rate_cell
        return {
            'rate_cell': rate_cell,                     # (B, Na, G) retained
            'rate_mol': rate_mol,                       # (B, Na, G) leaked
            'px_rate_cell': px_rate_cell,               # (B, Na, G)
            'beta_cc': extra['beta_cell'],              # (B, Na, Na) cell -> cell
            'beta_ac': extra['beta_transport'],         # (B, Na, Na) cell -> all
            'beta_leak': extra['beta_leak'],            # operator of the leaked half, or None
            'advect': advect.detach(),
            'px_proportion': extra['px_proportion'],
            'leak': extra['leak'].detach(),
            'has_leak': bool(extra['has_leak']),
            'beta_mix': self.beta_mix,
            'diffusive': extra['diffusive'],
            'cell_mask': extra['cell_mask'],
        }

    # ------------------------------------------------------------------- loss

    def kl_div(self, z, mu, logvar, valid=None):
        """KL of the molecule posterior to the (shared) mixture prior, over ``valid`` tokens."""
        from model.mutils import kl_mog_loss

        if valid is not None:
            if not bool(valid.any()):
                return z.new_zeros(())
            z, mu, logvar = (t[valid].unsqueeze(0) for t in (z, mu, logvar))

        mog_mu, mog_logvar, pi = self.prior()
        return kl_mog_loss(z, mu, logvar, mog_mu, mog_logvar, pi)

    def kl_div_std(self, mu, logvar, valid=None):
        from model.mutils import kl_std_loss

        return kl_std_loss(mu, logvar, valid=valid)
