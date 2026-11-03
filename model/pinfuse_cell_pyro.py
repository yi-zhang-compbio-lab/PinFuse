import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from model.pinfuse_cell import PinFuse_Cell


class PinFuse_Cell_Pyro(PinFuse_Cell):
    """Pyro (SVI) formulation of :class:`PinFuse_Cell` with the same networks.

    The encoder is the guide; the transport is given Dirichlet priors (a
    Gaussian heat kernel for diffusion, uniform for advection) and ``z`` the
    mixture-of-Gaussians prior. ``lams = (lam1, lam3, lam4, lam5)`` weight the
    out-of-tissue rate, transport distance, transport entropy and
    transported-proportion penalties, on the same scale as the PyTorch loss.
    """
    PYRO_SUBVERSION = '2'

    def __init__(
        self,
        n_obs,
        in_feats,
        hidden_dim,
        latent_dim,
        latent_dim_nb=5,
        train_beta=True,
        n_step=1,
        gene_disper=False,
        gene_dropout=True,
        lams=(0, 0, 0, 0),
        locality=1.0,
        kl_z_scale=5.0,
        kl_nb_scale=5.0,
    ):
        super().__init__(
            in_feats,
            hidden_dim,
            latent_dim,
            latent_dim_nb=latent_dim_nb,
            train_beta=train_beta,
            n_step=n_step,
            gene_disper=gene_disper,
            gene_dropout=gene_dropout,
            locality=locality,
        )
        self.n_obs = n_obs
        self.eps = 1e-6
        self.lam1 = lams[0]
        self.lam2 = lams[1]
        self.lam3 = lams[2]
        self.lam4 = lams[3]
        self.kl_beta_scale = 1
        self.kl_z_scale = kl_z_scale
        # pyro.poutine.scale rejects exactly 0
        self.kl_nb_scale = max(float(kl_nb_scale), 1e-8)

    def _build_attention_inputs(self, pos, in_tissue):
        distance = torch.cdist(pos, pos, p=2)
        threshold = torch.quantile(distance, self.locality, dim=-1, keepdim=True)
        mask = (distance <= threshold) | (in_tissue <= 1.0).reshape(in_tissue.shape[0], 1, -1)
        return distance, mask

    def _recon_scale(self, B, N, G):
        """Lifts a per-element mean penalty to the scale of the ``genes`` site."""
        return 50.0 * (self.n_obs / B) * (B * N * G)

    def model_uncondition(self, genes, in_tissue, pos, mass):
        pyro.module("encoder", self.encoder)
        pyro.module("decoder", self.decoder)
        pyro.module("prior", self.prior)

        B, N, G = genes.shape
        distance, attn_mask = self._build_attention_inputs(pos, in_tissue)
        sigma2 = pyro.sample("sigma2_prior",
                             dist.InverseGamma(genes.new_ones([1]) * 3.0,
                                               genes.new_ones([1]) * 10.0).to_event(1))

        prior_mu, prior_logvar, prior_pi = self.prior()
        prior_sigma = (0.5 * prior_logvar).exp()

        with pyro.plate("obs_plate", size=self.n_obs * N, subsample_size=B * N, dim=-1):
            if self.train_beta:
                with pyro.poutine.scale(scale=self.kl_beta_scale):
                    diffuse_prior = self.encoder.enc_beta.prior(pos, sigma2).transpose(-1, -2).reshape(B * N, N)
                    diffuse = pyro.sample("diffuse.T", dist.Dirichlet(diffuse_prior, validate_args=False))
                    diffuse = diffuse.reshape(B, N, N).transpose(-1, -2)

                    advect_prior = torch.ones_like(diffuse_prior) / N
                    advect = pyro.sample("advect.T", dist.Dirichlet(advect_prior, validate_args=False))
                    advect = advect.reshape(B, N, N).transpose(-1, -2)

                    beta = self.transport_operator(diffuse @ advect)

            with pyro.poutine.scale(scale=self.kl_z_scale):
                cats = dist.Categorical(logits=prior_pi)
                normal_dists = dist.Independent(
                    dist.Normal(prior_mu.expand(B * N, -1, -1), prior_sigma.expand(B * N, -1, -1) + 1e-4),
                    reinterpreted_batch_ndims=1,
                )
                z = pyro.sample("z", dist.MixtureSameFamily(cats, normal_dists)).reshape(B, N, -1)
            with pyro.poutine.scale(scale=self.kl_nb_scale):
                z_nb = pyro.sample(
                    "z_nb",
                    dist.Normal(torch.zeros(B * N, self.latent_dim_nb, device=genes.device),
                                torch.ones(B * N, self.latent_dim_nb, device=genes.device)).to_event(1)
                ).reshape(B, N, -1)

            _, _, px_r, px_rate, px_dropout, px_proportion = self.decoder(
                z, z_nb, genes, distance, attn_mask,
            )
            if self.train_beta:
                recon = beta @ px_rate
                recon = px_proportion * recon + (1 - px_proportion) * px_rate
            else:
                recon = px_rate

            # Penalties are computed here but added outside the plate, so they
            # are not broadcast and rescaled by the subsample ratio.
            penalty_val = None
            penalty_prop_val = None
            if self.lam1 > 0:
                mask = (1.0 - in_tissue * 0.5)
                denom = mask.sum().clamp_min(1.0)
                penalty_val = -(
                    px_rate * mask - px_rate * (1 - mask) * mask
                ).square().sum() / denom
            if self.lam4 > 0:
                penalty_prop_val = -(1.0 - px_proportion).mean()

            probs = recon / (recon + px_r + self.eps)
            probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
            with pyro.poutine.scale(scale=50.0):
                pyro.sample(
                    "genes",
                    dist.ZeroInflatedNegativeBinomial(
                        total_count=px_r.reshape(B * N, -1),
                        gate_logits=px_dropout.reshape(B * N, -1),
                        probs=probs.reshape(B * N, G)
                    ).to_event(1)
                )

        recon_scale = self._recon_scale(B, N, G)
        if penalty_val is not None:
            with pyro.poutine.scale(scale=self.lam1 * recon_scale):
                pyro.factor("penalty", penalty_val)
        if penalty_prop_val is not None:
            with pyro.poutine.scale(scale=self.lam4 * recon_scale):
                pyro.factor("penalty_prop", penalty_prop_val)

    def guide(self, genes, in_tissue, pos, mass):
        pyro.module("encoder", self.encoder)
        pyro.module("decoder", self.decoder)
        pyro.module("prior", self.prior)

        B, N, G = genes.shape
        distance, attn_mask = self._build_attention_inputs(pos, in_tissue)

        sigma2_prior_est = pyro.param(
            "sigma2_prior_est",
            genes.new_ones([1]) * 10,
            constraint=constraints.greater_than(self.eps),
            event_dim=1,
        )
        pyro.sample("sigma2_prior", dist.Delta(sigma2_prior_est, event_dim=1))

        penalty2_val = None
        penalty3_val = None
        with pyro.plate("obs_plate", size=self.n_obs * N, subsample_size=B * N, dim=-1):
            _, _, diffuse_enc, advect_enc, mu, logvar, mu_nb, logvar_nb, _ = self.encoder(
                genes, in_tissue, distance, attn_mask,
            )

            with pyro.poutine.scale(scale=self.kl_z_scale):
                std = (0.5 * logvar).exp()
                pyro.sample("z", dist.Normal(mu.reshape(B * N, -1), std.reshape(B * N, -1)).to_event(1))
            with pyro.poutine.scale(scale=self.kl_nb_scale):
                std_nb = (0.5 * logvar_nb).exp() + 1e-4
                pyro.sample("z_nb", dist.Normal(mu_nb.reshape(B * N, -1), std_nb.reshape(B * N, -1)).to_event(1))

            if self.train_beta:
                with pyro.poutine.scale(scale=self.kl_beta_scale):
                    diffuse_enc = diffuse_enc.transpose(-1, -2).reshape(B * N, N)
                    pyro.sample("diffuse.T", dist.Delta(diffuse_enc).to_event(1))

                    advect_enc = advect_enc.transpose(-1, -2).reshape(B * N, N)
                    pyro.sample("advect.T", dist.Delta(advect_enc).to_event(1))

                # Guide-side factors are subtracted from the ELBO, hence positive values.
                if self.lam2 > 0:
                    dist_pen = torch.cdist(pos, pos, p=2)
                    penalty2_val = (dist_pen * advect_enc.reshape(B, N, N)).mean()
                if self.lam3 > 0:
                    penalty3_val = (advect_enc * advect_enc.clamp_min(self.eps).log()).mean()

        recon_scale = self._recon_scale(B, N, G)
        if penalty2_val is not None:
            with pyro.poutine.scale(scale=self.lam2 * recon_scale):
                pyro.factor("penalty2", penalty2_val, has_rsample=True)
        if penalty3_val is not None:
            with pyro.poutine.scale(scale=self.lam3 * recon_scale):
                pyro.factor("penalty3", penalty3_val, has_rsample=True)

    def model_train(self, genes, in_tissue, pos, mass):
        pyro.condition(self.model_uncondition, data={
            "genes": genes.reshape(-1, genes.shape[-1])
        })(genes, in_tissue, pos, mass)
