import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import PiTLayer, PiTMultiheadLayer, PiTFlow, MoGPrior

SingleLayer = PiTLayer
MultiLayer = PiTMultiheadLayer


class Encoder(nn.Module):
    """Maps neighbourhood counts to two variational posteriors and the transport.

    * ``z``    -- expression latent (``enc_mu`` / ``enc_logvar``),
    * ``z_nb`` -- count latent feeding dispersion / dropout / proportion,
    * ``enc_beta`` -- the advection-diffusion transport operator (:class:`PiTFlow`).
    """

    def __init__(self, in_feats, hidden_dim, latent_dim, latent_dim_nb=5, n_heads=8,
                 qk_norm=False, advect_temp=None,
                 diff_kernel='exp', diff_min=0.0):
        super().__init__()
        emb_dim = 32
        self.tissue_emb = torch.nn.Embedding(3, emb_dim)

        self.enc_h1 = MultiLayer(in_feats + emb_dim, hidden_dim, n_heads, cond_dim=emb_dim)
        self.enc_h2 = MultiLayer(hidden_dim, hidden_dim, n_heads, cond_dim=emb_dim)

        self.enc_h1_nb = MultiLayer(in_feats, hidden_dim, n_heads)
        self.enc_h2_nb = MultiLayer(hidden_dim, hidden_dim, n_heads)

        self.enc_beta = PiTFlow(hidden_dim, qk_norm=qk_norm,
                                advect_temp=advect_temp, diff_kernel=diff_kernel,
                                diff_min=diff_min)

        self.enc_mu = SingleLayer(hidden_dim, latent_dim, cond_dim=emb_dim)
        self.enc_logvar = SingleLayer(hidden_dim, latent_dim, cond_dim=emb_dim)

        self.enc_logvar_nb = SingleLayer(hidden_dim, latent_dim_nb)
        self.enc_mu_nb = SingleLayer(hidden_dim, latent_dim_nb)

    def forward(self, genes, in_tissue, distance, mask):
        emb = self.tissue_emb(in_tissue.flatten().long())
        emb = emb.reshape(genes.shape[0], genes.shape[1], -1)
        x = torch.cat([genes, emb], dim=2)

        h1 = F.gelu(self.enc_h1(x, distance, mask))
        h = h1 + F.gelu(self.enc_h2(h1, distance, mask))
        mu = self.enc_mu(h, distance, mask)
        logvar = self.enc_logvar(h, distance, mask)

        h2 = F.gelu(self.enc_h1_nb(genes, distance, mask))
        h = h2 + F.gelu(self.enc_h2_nb(h2, distance, mask))
        mu_nb = self.enc_mu_nb(h, distance, mask)
        logvar_nb = self.enc_logvar_nb(h, distance, mask)

        beta, diffusive, diffuse, advect, raw_meta = self.enc_beta(h1, h1, distance, mask)

        return beta, diffusive, diffuse, advect, mu, logvar, mu_nb, logvar_nb, raw_meta


class Decoder(nn.Module):
    """Decodes the ZINB parameters and the transported proportion.

    The expression rate is one softmax over all (token, gene) entries of the
    neighbourhood, scaled by the neighbourhood's total counts (mass conservation).
    """

    def __init__(self, in_feats, hidden_dim, latent_dim, latent_dim_nb=5, n_heads=8,
                 gene_disper=False, gene_dropout=True, gene_proportion=True):
        super().__init__()
        self.dec_h1 = MultiLayer(latent_dim, hidden_dim, n_heads)
        self.dec_h2 = MultiLayer(hidden_dim, hidden_dim, n_heads)

        self.dec_h3 = MultiLayer(latent_dim_nb, hidden_dim, n_heads)
        self.dec_h4 = MultiLayer(hidden_dim, hidden_dim, n_heads)

        self.px_scale_lib = SingleLayer(hidden_dim, in_feats)
        self.px_scale_mass = SingleLayer(hidden_dim, in_feats)
        self.px_r_decoder = SingleLayer(hidden_dim, in_feats if gene_disper else 1)
        self.px_dropout_decoder = SingleLayer(hidden_dim, in_feats if gene_dropout else 1)
        self.px_proportion = SingleLayer(hidden_dim, in_feats if gene_proportion else 1)

        self.softmax = F.softmax

    def forward(self, z, z_nb, genes, distance, mask, total=None, valid=None):
        """``total`` overrides the mass budget read off ``genes``; ``valid`` marks padded tokens."""
        h = F.gelu(self.dec_h3(z_nb, distance, mask))
        h = h + F.gelu(self.dec_h4(h, distance, mask))
        px_r = F.softplus(self.px_r_decoder(h, distance, mask))
        px_r = torch.clamp(px_r, min=0.0, max=1e6)
        px_dropout = self.px_dropout_decoder(h, distance, mask)
        px_proportion = F.sigmoid(self.px_proportion(h, distance, mask))

        h = F.gelu(self.dec_h1(z, distance, mask))
        h = h + F.gelu(self.dec_h2(h, distance, mask))
        h_lib = self.px_scale_lib(h, distance, mask)
        h_mass = self.px_scale_mass(h, distance, mask)

        if not self.training:
            # Dropout correction of the observed counts at inference.
            dropout = torch.sigmoid(px_dropout).detach()
            genes = genes / torch.clamp(1 - dropout, min=1e-2, max=1.0)

        total = genes.sum(dim=(-1, -2), keepdims=True) if total is None else total
        B, N, G = h_lib.shape
        h_lib = h_lib.view(B, -1)
        if valid is not None:
            pad = ~valid.unsqueeze(-1).expand(B, N, G).reshape(B, -1)
            if pad.any():
                h_lib = h_lib.masked_fill(pad, -1e4)
        px_scale = self.softmax(h_lib, dim=-1).view(B, N, G)
        px_rate = px_scale * total
        px_scale_lib = px_scale_mass = 1

        return px_scale_mass, px_scale_lib, px_r, px_rate, px_dropout, px_proportion


class PinFuse_Cell(nn.Module):
    """PinFuse at the cell / spot level.

    Graph VAE over each node's spatial neighbourhood with a mixture-of-Gaussians
    latent prior. The denoised rate ``px_rate`` is redistributed by a learned
    transport ``beta = diffusion @ advection``::

        recon = prop * (beta^n_step @ px_rate) + (1 - prop) * px_rate

    and ``recon`` is fit to the observed counts with a ZINB likelihood.
    """
    VERSION = "v6.1"

    def __init__(self, in_feats, hidden_dim, latent_dim, latent_dim_nb=5, train_beta=True,
                 n_step=1, gene_disper=False, gene_dropout=True, locality=1.0,
                 qk_norm=False, advect_temp=None,
                 diff_kernel='exp', diff_min=0.0):
        super().__init__()

        self.locality = locality

        self.encoder = Encoder(
            in_feats, hidden_dim, latent_dim,
            latent_dim_nb=latent_dim_nb,
            qk_norm=qk_norm,
            advect_temp=advect_temp,
            diff_kernel=diff_kernel,
            diff_min=diff_min)
        self.decoder = Decoder(
            in_feats, hidden_dim, latent_dim,
            latent_dim_nb=latent_dim_nb,
            gene_disper=gene_disper,
            gene_dropout=gene_dropout)

        self.prior = MoGPrior(latent_dim)

        self.latent_dim = latent_dim
        self.latent_dim_nb = latent_dim_nb
        self.train_beta = train_beta
        self.n_step = max(1, int(n_step))
        self.gene_disper = gene_disper
        self.gene_dropout = gene_dropout
        self.eps = 1e-8

    def transport_operator(self, beta):
        """``beta ** n_step`` (Lie-Trotter splitting); stays column-stochastic."""
        beta_eff = beta
        for _ in range(self.n_step - 1):
            beta_eff = beta_eff @ beta
        return beta_eff

    def forward(self, genes, in_tissue, pos, mass):
        distance = torch.cdist(pos, pos, p=2)
        threshold = torch.quantile(distance, self.locality, dim=-1, keepdim=True)
        mask = (distance <= threshold) | (in_tissue <= 1.0).reshape(in_tissue.shape[0], 1, -1)

        beta, diffusive, diffuse, advect, mu, logvar, mu_nb, logvar_nb, raw_meta = self.encoder(
            genes, in_tissue, distance, mask,
        )
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        std_nb = torch.exp(0.5 * logvar_nb)
        z_nb = mu_nb + std_nb * torch.randn_like(std_nb)

        raw_meta['mu_nb'] = mu_nb
        raw_meta['logvar_nb'] = logvar_nb

        _, _, px_r, px_rate, px_dropout, px_proportion = self.decoder(
            z, z_nb, genes, distance, mask,
        )
        if self.train_beta:
            raw_meta['beta_step'] = beta.detach()
            beta = self.transport_operator(beta)
            recon = beta @ px_rate
            recon = px_proportion * recon + (1 - px_proportion) * px_rate
        else:
            recon = px_rate

        raw_meta['prop'] = px_proportion

        return recon, px_rate, px_r, px_dropout, z, mu, logvar, beta, diffusive, diffuse, advect, raw_meta

    def forward_simple(self, genes, in_tissue, pos, mass):
        recon, px_rate, _, _, _, mu, _, _, _, _, _, _ = self.forward(genes, in_tissue, pos, mass)
        return recon, px_rate, mu

    def forward_flow(self, genes, in_tissue, pos, mass):
        _, px_rate, _, _, _, _, _, beta, diffusive, diffuse, advect, raw_meta = self.forward(genes, in_tissue, pos, mass)
        return px_rate, beta, diffusive, diffuse, advect, raw_meta['prop']

    def forward_posterior(self, genes, in_tissue, pos, mass):
        recon, px_rate, px_r, px_dropout, _, _, _, _, _, _, _, _ = self.forward(genes, in_tissue, pos, mass)
        return recon, px_rate, px_r, px_dropout

    def kl_div(self, z, mu, logvar):
        from model.mutils import kl_mog_loss

        mog_mu, mog_logvar, pi = self.prior()
        return kl_mog_loss(z, mu, logvar, mog_mu, mog_logvar, pi)

    def kl_div_std(self, mu, logvar, valid=None):
        from model.mutils import kl_std_loss

        return kl_std_loss(mu, logvar, valid=valid)
