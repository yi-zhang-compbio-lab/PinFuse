"""Training + prediction of PinFuse (library; the CLI is ``main.py``)."""

import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataio import ARTIFACT_DIR, CHECKPOINT_DIR, cfg_get, load_data, save_config, subdir, to_numpy
from datasets import get_dataset, _process_batch_fast, filter_batch_by_cells
from model import pinfuse_cell, pinfuse_cell_pyro, pinfuse_spot
from model.mutils import get_scheduler, log_zinb_positive

CELL_MODEL = 'pinfuse_cell'
SPOT_MODEL = 'pinfuse_spot'


class NonFiniteLossError(RuntimeError):
    """Raised when the training loss becomes NaN or infinite."""


def set_seed(seed, use_pyro=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_pyro:
        import pyro
        pyro.set_rng_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_type(config):
    mtype = str(cfg_get(config, 'model.type', CELL_MODEL))
    if mtype not in (CELL_MODEL, SPOT_MODEL):
        raise ValueError(f"model.type must be '{CELL_MODEL}' or '{SPOT_MODEL}', got {mtype!r}")
    return mtype


def is_spot(config):
    return model_type(config) == SPOT_MODEL


def model_version(use_pyro=False, config=None):
    if config is not None and is_spot(config):
        return pinfuse_spot.PinFuse_Spot.VERSION
    if use_pyro:
        return f"{pinfuse_cell_pyro.PinFuse_Cell_Pyro.VERSION}.{pinfuse_cell_pyro.PinFuse_Cell_Pyro.PYRO_SUBVERSION}"
    return pinfuse_cell.PinFuse_Cell.VERSION


def build_model(config, n_gene, n_obs, use_pyro=False):
    kwargs = {
        'hidden_dim': cfg_get(config, 'model.hidden_dim', 128),
        'latent_dim': cfg_get(config, 'model.latent_dim', 10),
        'latent_dim_nb': cfg_get(config, 'model.latent_dim_nb', 5),
        'train_beta': cfg_get(config, 'model.train_beta', True),
        'n_step': cfg_get(config, 'model.n_step', 1),
        'gene_disper': cfg_get(config, 'model.gene_disper', False),
        'gene_dropout': cfg_get(config, 'model.gene_dropout', False),
        'locality': cfg_get(config, 'model.locality', 1.0),
    }

    if use_pyro:
        lams = (
            cfg_get(config, 'model.lam1', 0.0),
            cfg_get(config, 'model.lam3', 0.0),
            cfg_get(config, 'model.lam4', 0.0),
            cfg_get(config, 'model.lam5', 0.0),
        )
        return pinfuse_cell_pyro.PinFuse_Cell_Pyro(n_obs=n_obs, in_feats=n_gene, lams=lams, **kwargs).to(config.device)

    if is_spot(config):
        model = pinfuse_spot.PinFuse_Spot(
            in_feats=n_gene,
            beta_mix=cfg_get(config, 'model.beta_mix', 'leak'),
            gene_leak=cfg_get(config, 'model.gene_leak', True),
            share_prior=cfg_get(config, 'model.share_prior', True),
            freeze_gvae=cfg_get(config, 'model.freeze_gvae', True),
            stage1_locality=cfg_get(config, 'model.stage1_locality', None),
            flow_dst=cfg_get(config, 'model.stage2_flow_dst', True),
            mask_diffusion=cfg_get(config, 'model.stage2_mask_diffusion', True),
            qk_norm=cfg_get(config, 'model.stage2_qk_norm', True),
            diff_min=cfg_get(config, 'model.stage2_diff_min', 0.0),
            advect_temp=cfg_get(config, 'model.stage2_advect_temp', None),
            advect_temp_init=cfg_get(config, 'model.stage2_advect_temp_init', 1.0),
            advect_temp_min=cfg_get(config, 'model.stage2_advect_temp_min', 0.05),
            mol_latent=cfg_get(config, 'model.stage2_mol_latent', False),
            **kwargs,
        ).to(config.device)
        pretrain_ckpt = cfg_get(config, 'model.pretrain_ckpt', '')
        if pretrain_ckpt:
            model.load_pretrained(pretrain_ckpt)
        else:
            logging.warning(f"{SPOT_MODEL} built without model.pretrain_ckpt: the stage-1 model is untrained")
        return model

    return pinfuse_cell.PinFuse_Cell(
        in_feats=n_gene,
        qk_norm=cfg_get(config, 'model.stage1_qk_norm', False),
        advect_temp=cfg_get(config, 'model.stage1_advect_temp', None),
        diff_kernel=cfg_get(config, 'model.stage1_diff_kernel', 'exp'),
        diff_min=cfg_get(config, 'model.stage1_diff_min', 0.0),
        **kwargs,
    ).to(config.device)


def _total_loss(loss1, weighted_terms):
    # Zero-weight terms are left out: 0.0 * nan is still nan.
    loss = loss1
    for lam, term in weighted_terms:
        if lam:
            loss = loss + lam * term
    return loss


def _log_progress(epoch, num_epoch, idx, n_batch, terms):
    print(f"\r{epoch+1} / {num_epoch} | {idx+1} / {n_batch} : "
          + " : ".join(f"{t.item():.3e}" for t in terms) + "     ", end='', flush=True)


def _train_cell(config, model, train_loader, num_epoch):
    """Stage-1 loop (PinFuse_Cell).

    loss = loss1 (ZINB NLL) + lam1 * out-of-tissue rate + lam2 * latent KL
         + lam3 * transport distance + lam4 * transport entropy + lam5 * (1 - prop)
    """
    criterion = nn.MSELoss()
    n_nei = config.data.n_neighbors
    lams = [cfg_get(config, f'model.lam{i}', 0.0) for i in range(1, 6)]
    eps = 1e-6
    kl_type = str(cfg_get(config, 'model.kl_type', 'mog'))
    zinb_eps = float(cfg_get(config, 'model.zinb_eps', 1e-8))
    grad_clip = float(cfg_get(config, 'optim.grad_clip', 0.0) or 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_get(config, 'optim.lr', 1e-3))
    scheduler = get_scheduler(warmup_steps=100 * num_epoch, opt=opt)

    for i in range(num_epoch):
        for idx, batch in enumerate(train_loader):
            _, _, _, _batch = _process_batch_fast(batch, n_nei, mask_ratio=config.data.mask_ratio)
            genes = _batch['genes']
            pos = _batch['pos']
            in_tissue = _batch['in_tissue']

            recon, px_rate, px_r, px_dropout, z, mu, logvar, _, _, _, advect, raw_meta = model(**_batch)

            mask = (1.0 - in_tissue * 0.5)
            loss1 = -log_zinb_positive(x=genes, mu=recon, theta=px_r, pi=px_dropout, eps=zinb_eps).mean()
            loss2 = criterion(px_rate * mask, px_rate * (1 - mask) * mask) \
                / (mask.sum() if mask.sum() > 0 else 1.0) * recon.numel()
            if kl_type == 'std':
                loss3 = model.kl_div_std(mu, logvar) \
                    + model.kl_div_std(raw_meta['mu_nb'], raw_meta['logvar_nb'])
            else:
                loss3 = model.kl_div(z, mu, logvar)
            loss4 = (torch.cdist(pos, pos, p=2) * advect).mean()
            loss5 = (advect * advect.clamp_min(eps).log()).mean()
            loss6 = (1.0 - raw_meta['prop']).mean()

            terms = (loss1, loss2, loss3, loss4, loss5, loss6)
            loss = _total_loss(loss1, zip(lams, terms[1:]))
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"non-finite loss at epoch {i+1}, batch {idx+1}: "
                                         + " : ".join(f"{t.item():.3e}" for t in terms))

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            _log_progress(i, num_epoch, idx, len(train_loader), terms)
        print("")


def _train_spot(config, model, train_loader, num_epoch):
    """Stage-2 loop (PinFuse_Spot); only the wrapper's own parameters are trained.

    loss = loss1 (ZINB NLL, molecule tokens weighted by mol_loss_weight)
         + lam2 * molecule-latent KL + lam3 * distance of beta_ac
         + lam4 * entropy of the advection + lam5 * (1 - leak)
    """
    lams = [cfg_get(config, f'model.lam{i}', 0.0) for i in range(2, 6)]
    lam3, lam4 = lams[1], lams[2]
    kl_type = str(cfg_get(config, 'model.kl_type', 'mog'))
    grad_clip = float(cfg_get(config, 'optim.grad_clip', 0.0) or 0.0)
    min_cells = int(cfg_get(config, 'data.min_cell_tokens', 0))
    mol_weight = float(cfg_get(config, 'model.mol_loss_weight', 1.0) or 1.0)
    n_nei = config.data.n_neighbors
    eps = 1e-6

    if num_epoch <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=cfg_get(config, 'optim.lr', 1e-3))
    scheduler = get_scheduler(warmup_steps=100 * num_epoch, opt=opt)
    n_graph = n_dropped = 0

    for i in range(num_epoch):
        for idx, batch in enumerate(train_loader):
            _, _, _, _batch = _process_batch_fast(batch, n_nei, mask_ratio=config.data.mask_ratio)

            # Graphs without a cell token have nowhere to receive mass.
            n_graph += _batch['genes'].shape[0]
            _batch, keep = filter_batch_by_cells(_batch, min_cells=min_cells)
            if keep is not None:
                n_dropped += int((~keep).sum())
                if not bool(keep.any()):
                    continue

            recon, _, px_r, px_dropout, mu, _, _, advect, prop, extra = model(**_batch)
            zero = torch.zeros((), device=recon.device)

            log_prob = log_zinb_positive(x=_batch['genes'], mu=recon, theta=px_r, pi=px_dropout,
                                         eps=model.eps)
            if mol_weight == 1.0:
                loss1 = -log_prob.mean()
            else:
                is_cell_token = (_batch['in_tissue'].squeeze(-1) == 2.0)
                token_weight = torch.where(is_cell_token.unsqueeze(-1),
                                           torch.ones_like(log_prob),
                                           torch.full_like(log_prob, mol_weight))
                loss1 = -log_prob.sum() / token_weight.sum()

            if kl_type == 'std':
                loss3 = model.kl_div_std(mu, extra['logvar'], valid=extra['valid_cell']) \
                    + model.kl_div_std(extra['mu_nb'], extra['logvar_nb'], valid=extra.get('valid_mol'))
            else:
                loss3 = model.kl_div(extra['z'], mu, extra['logvar'], valid=extra['valid_cell'])

            beta_op = extra['beta_ac']
            loss4 = (torch.cdist(_batch['pos'], _batch['pos'], p=2) * beta_op).mean() if lam3 else zero
            loss5 = (advect * advect.clamp_min(eps).log()).mean() if lam4 else zero

            leak = extra['leak']
            if leak.requires_grad and extra['has_leak']:
                sel = extra['valid_cell_token'] & extra['valid_mol'].any(dim=-1, keepdim=True)
                loss6 = (1.0 - leak[sel]).mean() if bool(sel.any()) else zero
            else:
                loss6 = (1.0 - prop).mean()

            terms = (loss1, loss3, loss4, loss5, loss6)
            loss = _total_loss(loss1, zip(lams, terms[1:]))
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"non-finite loss at epoch {i+1}, batch {idx+1}: "
                                         + " : ".join(f"{t.item():.3e}" for t in terms))
            if not loss.requires_grad:
                raise RuntimeError(f"training loss has no gradient path (epoch {i+1}, batch {idx+1}): "
                                   "the batch holds no out-of-cell token")

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            _log_progress(i, num_epoch, idx, len(train_loader), terms)
        print("")

    if min_cells > 0:
        logging.info(f"cell-free graphs dropped (min_cell_tokens={min_cells}): {n_dropped} / {n_graph}")


def _train_pyro(config, model, train_loader, num_epoch):
    from pyro.infer import SVI, Trace_ELBO

    n_nei = config.data.n_neighbors
    scheduler = get_scheduler(warmup_steps=100 * num_epoch, opt=torch.optim.Adam, platform='pyro')
    svi = SVI(model.model_train, model.guide, scheduler, loss=Trace_ELBO())

    for i in range(num_epoch):
        for idx, batch in enumerate(train_loader):
            _, _, _, _batch = _process_batch_fast(batch, n_nei, mask_ratio=config.data.mask_ratio)
            loss = svi.step(**_batch)
            if not math.isfinite(loss):
                raise NonFiniteLossError(f"non-finite ELBO at epoch {i+1}, batch {idx+1}: {loss}")
            scheduler.step()
            print(f"\r{i+1} / {num_epoch} | {idx+1} / {len(train_loader)} : {loss:.3e}     ", end='', flush=True)
        print("")


def train_model(config, data, use_pyro=False, model=None):
    """Train a model on ``data``; a diverged run restarts from a fresh init with a new seed.

    ``model``: an untrained model from :func:`build_model` to train in place of a fresh one.
    """
    torch.set_float32_matmul_precision("high")

    n_obs, n_gene = data.shape
    n_nei = config.data.n_neighbors
    num_epoch = int(cfg_get(config, 'training.epochs', 3))
    logging.info(f"n_obs: {n_obs}, n_gene: {n_gene}, n_neighbors: {n_nei}, epochs: {num_epoch}")

    seed = cfg_get(config, 'seed', 1234)
    set_seed(seed, use_pyro=use_pyro)
    if model is None:
        model = build_model(config, n_gene, n_obs, use_pyro=use_pyro)

    if n_obs < n_nei:
        logging.info(f"Skipped training since there are only {n_obs} < {n_nei} spots.")
        return model

    train_loader, _ = get_dataset(config, data.adata)
    max_attempts = int(cfg_get(config, 'training.max_attempts', 3))
    for attempt in range(max_attempts):
        if attempt:
            logging.info(f"restarting training from a fresh init ({attempt+1}/{max_attempts})")
            if use_pyro:
                import pyro
                pyro.clear_param_store()
            set_seed(seed + attempt, use_pyro=use_pyro)
            model = build_model(config, n_gene, n_obs, use_pyro=use_pyro)
        try:
            if use_pyro:
                _train_pyro(config, model, train_loader, num_epoch)
            elif is_spot(config):
                _train_spot(config, model, train_loader, num_epoch)
            else:
                _train_cell(config, model, train_loader, num_epoch)
            return model
        except NonFiniteLossError as err:
            logging.warning(f"training diverged: {err}")

    raise NonFiniteLossError(f"training diverged in all {max_attempts} attempts")


def predict_data(config, data, model, do_extra=True, do_flow=True, num_sample=1, count_dist='poisson'):
    """Write latent / prediction / flow / source fields into ``data.adata``."""
    from predict import predict_all

    predict_all(config, data.adata, model, do_extra=do_extra, do_flow=do_flow,
                num_sample=num_sample, count_dist=count_dist)
    return data


def fit(config, data, model=None, use_pyro=False, dst_dir=None):
    """Train ``model`` (or a fresh one from ``config``) on ``data`` and return it.

    ``data`` is a :class:`dataio.SpatialData` from :func:`dataio.load_data`. With
    ``dst_dir`` the config and the checkpoint are written under it.
    """
    if dst_dir:
        save_config(config, dst_dir)
    model = train_model(config, data, use_pyro=use_pyro, model=model)
    if dst_dir:
        save_checkpoint(model, dst_dir, use_pyro=use_pyro)
    return model


def set_output_metadata(adata, config, model, use_pyro=False, num_sample=1, count_dist='poisson'):
    version = model_version(use_pyro=use_pyro, config=config)
    lam1 = cfg_get(config, 'model.lam1', 0.0)
    n_nei = cfg_get(config, 'data.n_neighbors', 40)
    locality = cfg_get(config, 'model.locality', 1.0)

    # Learned self-loop bias of the transport (the trained one lives on the frozen stage-1 model).
    flow = model.gvae.encoder.enc_beta if hasattr(model, 'gvae') else model.encoder.enc_beta
    beta_reg = F.softplus(flow.lam)

    adata.uns['ver'] = version
    adata.uns['lam1'] = lam1
    adata.uns['lam2'] = 'pyro' if use_pyro else cfg_get(config, 'model.lam2', 0.0)
    if is_spot(config):
        adata.uns['beta_mix'] = str(cfg_get(config, 'model.beta_mix', 'leak'))
    adata.uns['n_nei'] = n_nei
    adata.uns['locality'] = locality
    adata.uns['meta'] = {
        'version': version,
        'n_neighbor': n_nei,
        'locality': locality,
        'n_step': cfg_get(config, 'model.n_step', 1),
        'advection_constraint': f'{to_numpy(beta_reg).item():.2f}',
        'num_sample': num_sample,
        'count_distribution': count_dist,
        'model': 'pyro' if use_pyro else model_type(config),
    }


def save_checkpoint(model, dst_dir, use_pyro=False):
    ckpt_path = os.path.join(subdir(dst_dir, CHECKPOINT_DIR), 'checkpoint.pth')
    if use_pyro or hasattr(model, 'PYRO_SUBVERSION'):
        # Pyro-registered modules cannot be pickled whole.
        torch.save({'state_dict': model.state_dict(), 'model_type': 'pyro'}, ckpt_path)
    else:
        torch.save(model, ckpt_path)
    logging.info(f"model saved: {ckpt_path}")
    return ckpt_path


def checkpoint_gene_dim(state_dict):
    for key in ('encoder.enc_h1_nb.v_proj.weight', 'encoder_nb.enc_h1_nb.v_proj.weight'):
        weight = state_dict.get(key)
        if weight is not None:
            return int(weight.shape[1])
    return None


def load_checkpoint(config, ckpt_dir, n_obs=None, n_gene=None, use_pyro=False):
    ckpt_path = os.path.abspath(ckpt_dir)
    if os.path.isdir(ckpt_path):
        for candidate in (os.path.join(ckpt_path, CHECKPOINT_DIR, 'checkpoint.pth'),
                          os.path.join(ckpt_path, 'checkpoint.pth')):
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break
        else:
            raise FileNotFoundError(f"no checkpoint.pth found under {ckpt_dir}")
    loaded = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    is_state_dict = isinstance(loaded, dict) and 'state_dict' in loaded
    state_dict = loaded['state_dict'] if is_state_dict else loaded.state_dict()

    n_ckpt_gene = checkpoint_gene_dim(state_dict)
    if n_gene is not None and n_ckpt_gene not in (None, n_gene):
        raise ValueError(f"checkpoint {ckpt_path} was trained on {n_ckpt_gene} genes but the dataset has "
                         f"{n_gene}: the gene panel (and its order) must match")

    if is_state_dict:
        model = build_model(config, n_gene, n_obs, use_pyro=use_pyro)
        model.load_state_dict(state_dict)
    else:
        model = loaded
    logging.info(f"model loaded from: {ckpt_path}")
    return model


def mark_off_tissue(config, data, key='off_issue'):
    """Flag nodes whose graph holds no cell token; drop them with ``data.drop_cell_free``."""
    from predict import mark_graph

    min_cells = max(int(cfg_get(config, 'data.min_cell_tokens', 0)), 1)
    _, test_loader = get_dataset(config, data.adata)
    flagged = mark_graph(config, data.adata, test_loader=test_loader, min_cells=min_cells, key=key)

    if not bool(cfg_get(config, 'data.drop_cell_free', False)):
        return data

    keep = ~flagged
    if not keep.any():
        raise ValueError(f"drop_cell_free would remove every node: no graph holds {min_cells} cell tokens")
    logging.info(f"drop_cell_free: {int(keep.sum())} / {data.adata.n_obs} nodes kept")
    data.adata = data.adata[keep].copy()
    return data


def run(config, src_dir, dst_dir, ckpt_dir=None, save_ckpt=True, save_adata=True, use_pyro=False,
        do_predict=True, do_extra=True, do_flow=True, num_sample=1, count_dist='poisson'):
    """Train (or reload) a model, optionally predict, and write
    ``<dst_dir>/checkpoints/checkpoint.pth`` and ``<dst_dir>/artifacts/``."""
    os.makedirs(dst_dir, exist_ok=True)
    artifact_dir = subdir(dst_dir, ARTIFACT_DIR)
    save_config(config, dst_dir)
    data = load_data(config, src_dir)

    if is_spot(config):
        data = mark_off_tissue(config, data)

    if ckpt_dir:
        model = load_checkpoint(config, ckpt_dir, n_obs=data.shape[0], n_gene=data.shape[1],
                                use_pyro=use_pyro)
    else:
        model = train_model(config, data, use_pyro=use_pyro)
        if save_ckpt:
            save_checkpoint(model, dst_dir, use_pyro=use_pyro)

    if do_predict:
        predict_data(config, data, model, do_extra=do_extra, do_flow=do_flow,
                     num_sample=num_sample, count_dist=count_dist)
        set_output_metadata(data.adata, config, model, use_pyro=use_pyro,
                            num_sample=num_sample, count_dist=count_dist)
        data.adata.uns['dataset'] = data.name()
        if save_adata:
            descr = 'auto-denoised' if not ckpt_dir else f'denoised with the pretrained model from {ckpt_dir}'
            data.save(artifact_dir, descr=descr)

    return data, model
