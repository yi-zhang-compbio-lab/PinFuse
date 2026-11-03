"""Inference passes. Each writes its outputs into ``adata.obs`` / ``adata.obsm``."""
import logging
import time

import numpy as np
import scipy.sparse as sp_sparse
import torch

from datasets import get_dataset, graph_global_ids, _process_batch_fast


def _device(config):
    return 'cpu' if config.data.force_cpu else config.device


def predict_all(config, adata, model, do_extra=True, do_flow=True, num_sample=1, count_dist='poisson'):
    """Run every inference pass on ``adata`` (in place) with one shared loader.

    Writes ``obsm['latent' | 'predictions' | 'reconstruct']``, and with ``do_extra`` the
    sampled counts and ZINB parameters, with ``do_flow`` the flow field and source cells.
    Returns ``adata``.
    """
    torch.set_float32_matmul_precision("high")
    if adata.n_obs < config.data.n_neighbors:
        logging.info(f"Skipped prediction since there are only {adata.n_obs} spots.")
        return adata

    to_csc = bool(getattr(config.data, 'to_csc', False))
    _, test_loader = get_dataset(config, adata)

    predict(config, adata, model, test_loader=test_loader)
    if do_extra:
        predict_extra(config, adata, model, to_csc=to_csc, test_loader=test_loader,
                      num_sample=num_sample, count_dist=count_dist)
    if do_flow:
        predict_flow(config, adata, model, test_loader=test_loader)
        predict_source(config, adata, model, test_loader=test_loader)
    return adata


def mark_graph(config, adata, test_loader=None, min_cells=1, cell_value=2.0, key='off_issue'):
    """Flag nodes whose sampled graph holds fewer than ``min_cells`` cell tokens.

    Writes ``obs['n_cell_graph']`` and ``obs[key]``; returns the flag.
    """
    if test_loader is None:
        _, test_loader = get_dataset(config, adata)

    n_nei = config.data.n_neighbors
    device = _device(config)
    n_cell_graph = torch.zeros(adata.n_obs, dtype=torch.long).to(device)

    for idx, batch in enumerate(test_loader):
        _, output_nodes, _, _batch = _process_batch_fast(batch, n_nei)
        n_cells = (_batch['in_tissue'].squeeze(-1) == cell_value).sum(dim=-1)
        n_cell_graph[output_nodes.to(device)] = n_cells.to(device)
        print(f"\r{idx+1} / {len(test_loader)}    ", end='', flush=True)
    print("")

    counts = n_cell_graph.cpu().numpy()
    adata.obs['n_cell_graph'] = counts
    adata.obs[key] = counts < min_cells

    flagged = int(adata.obs[key].sum())
    logging.info(f"{key}: {flagged} / {adata.n_obs} nodes have fewer than {min_cells} "
                 f"cell tokens in their {n_nei}-neighbour graph")
    return adata.obs[key].to_numpy()


def predict(config, adata, model, test_loader=None, threshold=1e-3):
    """Latent and denoised expression of every node, read at the node's own (seed) token.

    ``obsm['latent']``, ``obsm['predictions']`` (rate before transport),
    ``obsm['reconstruct']`` (rate after transport).
    """
    model.eval()
    start = time.perf_counter()
    if test_loader is None:
        _, test_loader = get_dataset(config, adata)

    n_gene = adata.n_vars
    device = _device(config)
    latent = torch.zeros(adata.n_obs, model.latent_dim).to(device)
    predictions = torch.zeros(adata.n_obs, n_gene).to(device)
    reconstruct = torch.zeros(adata.n_obs, n_gene).to(device)
    for idx, batch in enumerate(test_loader):
        _, output_nodes, _, _batch = _process_batch_fast(batch, config.data.n_neighbors)
        output_nodes = output_nodes.to(device)
        with torch.no_grad():
            recon, px_rate, mu = model.forward_simple(**_batch)
        latent[output_nodes, :] = mu[:, 0, :].to(device)
        predictions[output_nodes, :] = px_rate[:, 0, :].to(device)
        reconstruct[output_nodes, :] = recon[:, 0, :].to(device)
        print(f"\r{idx+1} / {len(test_loader)}    ", end='', flush=True)
    print("")

    adata.obsm['latent'] = latent.cpu().numpy()
    adata.obsm['predictions'] = predictions.cpu().numpy()
    adata.obsm['reconstruct'] = reconstruct.cpu().numpy()
    adata.obsm['predictions'][adata.obsm['predictions'] < threshold] = 0
    adata.obsm['reconstruct'][adata.obsm['reconstruct'] < threshold] = 0

    adata.obs['n_counts'] = adata.X.sum(axis=1)
    adata.obs['n_counts_generated'] = adata.obsm["predictions"].sum(axis=1)
    adata.obs['n_counts_generated_constr'] = adata.obsm["reconstruct"].sum(axis=1)
    logging.info(f"predict: {time.perf_counter() - start:.1f}s")


def predict_extra(config, adata, model, to_csc=False, test_loader=None,
                  num_sample=5, count_dist="nb"):
    """Posterior means and sampled counts.

    ``obsm['predictions_count']`` (Poisson / NB sample of the rate),
    ``obsm['reconstruct_count']`` (ZINB sample), ``obsm['dropout']``, ``obsm['dispersion']``.
    """
    import pyro.distributions as dist

    model.eval()
    start = time.perf_counter()
    if test_loader is None:
        _, test_loader = get_dataset(config, adata)

    n_gene = adata.n_vars
    device = _device(config)
    dispersion = torch.zeros(adata.n_obs, n_gene if model.gene_disper else 1).to(device)
    dropout = torch.zeros(adata.n_obs, n_gene if model.gene_dropout else 1).to(device)
    predictions = torch.zeros(adata.n_obs, n_gene).to(device)
    reconstruct = torch.zeros(adata.n_obs, n_gene).to(device)
    predictions_count = torch.zeros(adata.n_obs, n_gene).to(device)
    reconstruct_count = torch.zeros(adata.n_obs, n_gene).to(device)

    def _repeat_batch(batch, repeats):
        if repeats <= 1:
            return batch
        return {k: v.repeat_interleave(repeats, dim=0) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    eps = 1e-6
    for idx, batch in enumerate(test_loader):
        _, output_nodes, _, _batch = _process_batch_fast(batch, config.data.n_neighbors)
        output_nodes = output_nodes.to(device)
        batch_size = _batch["genes"].shape[0]

        with torch.no_grad():
            recon, px_rate, px_r, px_dropout = model.forward_posterior(**_repeat_batch(_batch, num_sample))
        px_r = px_r.clamp_min(eps)

        recon = recon[:, 0, :].reshape(batch_size, num_sample, -1)
        px_rate = px_rate[:, 0, :].reshape(batch_size, num_sample, -1)
        px_r = px_r[:, 0, :].reshape(batch_size, num_sample, -1)
        px_dropout = px_dropout[:, 0, :].reshape(batch_size, num_sample, -1)

        if count_dist == "poisson":
            batch_pred_count = dist.Poisson(px_rate).sample().median(dim=1).values
        elif count_dist == "nb":
            probs = (px_rate / (px_rate + px_r + 1e-8)).clamp(0.0, 1.0 - eps)
            batch_pred_count = dist.NegativeBinomial(total_count=px_r, probs=probs).sample().median(dim=1).values
        else:
            raise ValueError(f"Unsupported count distribution: {count_dist}")

        probs = (recon / (recon + px_r + 1e-8)).clamp(0.0, 1.0 - eps)
        zinb = dist.ZeroInflatedNegativeBinomial(gate_logits=px_dropout, total_count=px_r, probs=probs)
        batch_recon_count = zinb.sample().median(dim=1).values

        predictions[output_nodes, :] = px_rate.mean(dim=1).to(device)
        reconstruct[output_nodes, :] = recon.mean(dim=1).to(device)
        dispersion[output_nodes, :] = px_r.mean(dim=1).to(device)
        dropout[output_nodes, :] = px_dropout.mean(dim=1).to(device)
        predictions_count[output_nodes, :] = batch_pred_count.to(device)
        reconstruct_count[output_nodes, :] = batch_recon_count.to(device)
        print(f"\r{idx+1} / {len(test_loader)}    ", end='', flush=True)
    print("")

    adata.obsm['predictions'] = predictions.cpu().numpy()
    adata.obsm['reconstruct'] = reconstruct.cpu().numpy()
    adata.obsm['predictions_count'] = predictions_count.cpu().numpy()
    adata.obsm['reconstruct_count'] = reconstruct_count.cpu().numpy()
    adata.obsm['dropout'] = torch.sigmoid(dropout).cpu().numpy()
    adata.obsm['dispersion'] = dispersion.cpu().numpy()
    adata.obs['n_counts_sampled'] = adata.obsm["predictions_count"].sum(axis=1)

    if to_csc:
        adata.obsm['predictions_count'] = sp_sparse.csc_matrix(adata.obsm['predictions_count'])
    logging.info(f"predict_extra: {time.perf_counter() - start:.1f}s")


def predict_flow(config, adata, model, test_loader=None):
    """Net molecular flow at every node.

    ``obsm['direction']`` (N, G, 2) per-gene flow vector, ``obsm['direction_tol']``
    its gene mean, ``obs['log_diffusivity']``.
    """
    model.eval()
    start = time.perf_counter()
    if test_loader is None:
        _, test_loader = get_dataset(config, adata)

    n_gene = adata.n_vars
    n_nei = config.data.n_neighbors
    device = _device(config)
    decomposed = hasattr(model, 'forward_flow_parts')
    direction = torch.zeros(adata.n_obs, n_gene, 2).to(device)
    diffusivity = torch.zeros(adata.n_obs).to(device)
    for idx, batch in enumerate(test_loader):
        _, output_nodes, _, _batch = _process_batch_fast(batch, n_nei)
        output_nodes = output_nodes.to(device)

        if decomposed:
            # PinFuse_Spot: the retained block is px_prop-weighted, the leaked block is not.
            with torch.no_grad():
                parts = model.forward_flow_parts(**_batch)
            diffusivity[output_nodes] = parts['diffusive'][:, 0].to(device)
            beta_cc, beta_ac = parts['beta_cc'], parts['beta_ac']
            rate_cell, rate_mol = parts['rate_cell'], parts['rate_mol']
            px_prop = parts['px_proportion']
            out_flow = (beta_cc[:, 1:, 0:1] * rate_cell[:, None, 0] * px_prop[:, 1:]
                        + beta_ac[:, 1:, 0:1] * rate_mol[:, None, 0])
            in_flow = (beta_cc[:, 0, 1:, None] * rate_cell[:, 1:] * px_prop[:, 0:1]
                       + beta_ac[:, 0, 1:, None] * rate_mol[:, 1:])
        else:
            with torch.no_grad():
                px_rate, beta, diffusive, _, _, px_prop = model.forward_flow(**_batch)
            diffusivity[output_nodes] = diffusive[:, 0].to(device)
            # Only the px_prop share of the mass is transported.
            out_flow = beta[:, 1:, 0:1] * px_rate[:, None, 0] * px_prop[:, 1:]
            in_flow = beta[:, 0, 1:, None] * px_rate[:, 1:] * px_prop[:, 0:1]

        diff = out_flow - in_flow
        dire = _batch['pos'][:, 1:, :] - _batch['pos'][:, 0:1, :]
        dire = dire / (torch.linalg.norm(dire + 1e-8, axis=-1, keepdims=True))
        flow = diff.reshape(-1, n_nei, n_gene, 1) * dire.reshape(-1, n_nei, 1, 2)
        direction[output_nodes, :] = flow.sum(axis=1).to(device)
        print(f"\r{idx+1} / {len(test_loader)}    ", end='', flush=True)
    print("")

    adata.obsm['direction'] = direction.cpu().numpy()
    adata.obsm['direction_tol'] = direction.mean(dim=-2).cpu().numpy()
    adata.obs['log_diffusivity'] = -diffusivity.cpu().numpy()
    logging.info(f"predict_flow: {time.perf_counter() - start:.1f}s")


def source_score(parts, mode=None):
    """Per-(destination, source, gene) origin score, ``(B, Na, G)``; destination 0 is the seed."""
    mode = str(mode if mode is not None else parts.get('beta_mix', 'leak'))
    beta_leak = parts.get('beta_leak')
    if beta_leak is None:
        beta_leak = parts['beta_ac']

    if mode == 'compose':
        # Leaked mass reaches a molecule directly or relayed through beta_cc;
        # both are attributed to the cell whose decoded rate it came from.
        prop_all = parts['px_proportion']
        rate_all = parts['px_rate_cell']
        v = beta_leak[:, 0, :].unsqueeze(-1) * parts['leak']
        relay_g = torch.einsum('bdg,bds->bsg', v * prop_all, parts['beta_cc'])
        direct_g = v * (1.0 - prop_all)
        return rate_all * (relay_g + direct_g)

    if mode == 'retrain_prop':
        return parts['beta_ac'][:, 0, :].unsqueeze(-1) * parts['px_rate_cell']

    # 'leak' / 'leak_molecule' / 'retrain_leak' / 'renorm' (approximate for the latter)
    return beta_leak[:, 0, :].unsqueeze(-1) * parts['rate_mol']


def predict_source(config, adata, model, test_loader=None, cell_value=2.0):
    """Most likely source cell of every detected gene in every out-of-cell token.

    Scores every candidate cell ``s`` for molecule ``m`` and gene ``g`` with
    :func:`source_score` (e.g. ``beta_ac[m->s] * leak[s, g] * rate_cell[s, g]``)
    and takes the argmax. Writes:

    * ``obsm['source_cell_gene_idx']``  (N, G) global obs index of the source, -1 = none
    * ``obsm['source_cell_gene_prob']`` (N, G) share of the winning score
    * ``obs['source_cell']`` / ``obs['source_cell_prob']`` -- the same for each
      molecule's dominant (highest-count) gene
    """
    if not hasattr(model, 'forward_flow_parts'):
        logging.info("skipped the source-cell pass (model exposes no decomposed transport)")
        return None

    model.eval()
    start = time.perf_counter()
    if test_loader is None:
        _, test_loader = get_dataset(config, adata)

    n_nei = config.data.n_neighbors
    device = _device(config)
    N, G = adata.n_obs, adata.n_vars
    X = adata.X.tocsr() if sp_sparse.issparse(adata.X) else np.asarray(adata.X)

    gene_src_idx = np.full((N, G), -1, dtype=np.int32)
    gene_src_prob = np.full((N, G), 0.0, dtype=np.float32)

    for batch_no, batch in enumerate(test_loader):
        _, output_nodes, blocks, _batch = _process_batch_fast(batch, n_nei)
        ids, valid = graph_global_ids(blocks, output_nodes, n_nei)

        with torch.no_grad():
            parts = model.forward_flow_parts(**_batch)

        in_tissue = _batch['in_tissue'].squeeze(-1)
        is_cell = (in_tissue == cell_value) & valid.to(in_tissue.device)
        seed_is_mol = in_tissue[:, 0] != cell_value

        # Candidates: cells whose advection column is distinguishable from uniform.
        advect = parts['advect']
        usable = advect.max(dim=-2).values > (1.0 / advect.shape[-2]) * 1.02
        cand = (is_cell & usable.to(is_cell.device)).to(torch.float32)

        score_all = source_score(parts) * cand.unsqueeze(-1)            # (B, Na, G)
        winner_na = score_all.argmax(dim=1)
        winner_top = score_all.max(dim=1).values
        winner_prob = winner_top / score_all.sum(dim=1).clamp_min(1e-12)
        ids_d = ids.to(device)
        winner_gidx = ids_d.gather(1, winner_na.to(device).clamp(0, ids_d.shape[1] - 1))

        mol_rows = np.where(seed_is_mol.cpu().numpy())[0]
        if len(mol_rows):
            obs_ids = output_nodes.cpu().numpy()[mol_rows]
            w_gidx = winner_gidx[mol_rows].cpu().numpy()
            w_prob = winner_prob[mol_rows].cpu().numpy()
            w_top = winner_top[mol_rows].cpu().numpy()
            n_cand = cand[mol_rows].sum(dim=-1).cpu().numpy()
            raw_mol = np.asarray(X[obs_ids].todense()) if sp_sparse.issparse(X) else X[obs_ids]

            ok = (raw_mol > 0) & (n_cand[:, None] > 0) & (w_top > 0)
            r, c = np.where(ok)
            gene_src_idx[obs_ids[r], c] = w_gidx[r, c].astype(np.int32)
            gene_src_prob[obs_ids[r], c] = w_prob[r, c].astype(np.float32)

        print(f"\r{batch_no+1} / {len(test_loader)}    ", end='', flush=True)
    print("")

    adata.obsm['source_cell_gene_idx'] = gene_src_idx
    adata.obsm['source_cell_gene_prob'] = gene_src_prob

    # Per-molecule summary on the dominant gene.
    source = np.full(N, -1, dtype=np.int64)
    prob = np.full(N, np.nan, dtype=np.float32)
    mol_obs = np.where(np.asarray(adata.obs['in_tissue'] != cell_value, dtype=bool))[0]
    if mol_obs.size:
        raw = np.asarray(X[mol_obs].todense()) if sp_sparse.issparse(X) else X[mol_obs]
        dom = raw.argmax(axis=1)
        src_dom = gene_src_idx[mol_obs, dom]
        ok = (raw[np.arange(len(mol_obs)), dom] > 0) & (src_dom >= 0)
        source[mol_obs[ok]] = src_dom[ok]
        prob[mol_obs[ok]] = gene_src_prob[mol_obs, dom][ok]
    adata.obs['source_cell'] = source
    adata.obs['source_cell_prob'] = prob

    logging.info(f"predict_source: {int((source >= 0).sum())} / {mol_obs.size} out-of-cell tokens "
                 f"assigned, {time.perf_counter() - start:.1f}s")
    return source
