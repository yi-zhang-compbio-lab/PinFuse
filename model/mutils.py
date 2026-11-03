import numpy as np
import torch
import torch.nn.functional as F


def get_scheduler(warmup_steps, opt, platform='torch'):
    """Linear learning-rate warm-up from 0 to 1 over ``warmup_steps``."""
    def lr_lambda(current_step):
        return min((current_step + 1) / warmup_steps, 1.0)

    if platform == 'torch':
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(opt, lr_lambda=lr_lambda)
    elif platform == 'pyro':
        from pyro.optim import LambdaLR
        return LambdaLR({
            'optimizer': opt,
            'lr_lambda': lr_lambda,
            'optim_args': {"lr": 1e-3}})


def kl_std_loss(mu, logvar, valid=None):
    """KL of N(mu, exp(logvar)) to N(0, I), summed over latent dims, averaged over tokens."""
    MAX_LOGVAR = 10
    logvar = logvar.clamp(min=-MAX_LOGVAR, max=MAX_LOGVAR)
    if valid is not None:
        mu, logvar = mu[valid], logvar[valid]
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1)
    return kl.mean()


def gaussian_log_prob(z, mu, logvar):
    return -0.5 * (((z - mu) ** 2) / logvar.exp() + logvar + np.log(2 * np.pi)).sum(dim=-1)


def kl_mog_loss(z, mu_post, logvar_post, mog_mu, mog_logvar, pi):
    """Single-sample KL of the posterior to a mixture-of-Gaussians prior (``pi`` are logits)."""
    K = mog_mu.shape[0]
    q_log_prob = gaussian_log_prob(z, mu_post, logvar_post)                  # (B, N)
    log_probs = gaussian_log_prob(z.unsqueeze(2),
                                  mog_mu.unsqueeze(0).unsqueeze(0),
                                  mog_logvar.unsqueeze(0).unsqueeze(0))      # (B, N, K)
    log_pi = torch.log_softmax(pi, dim=-1).view(1, 1, K)
    log_pz = torch.logsumexp(log_probs + log_pi, dim=-1)                     # (B, N)
    return (q_log_prob - log_pz).mean()


# Adapted from scvi/distributions/_negative_binomial.py
def log_zinb_positive(x, mu, theta, pi, eps=1e-6):
    """Element-wise ZINB log-likelihood; ``pi`` is the dropout logit."""
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    # theta == 0 makes lgamma(theta) infinite.
    theta = theta.clamp_min(eps)

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    return mul_case_zero + mul_case_non_zero
