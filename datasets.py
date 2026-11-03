import dgl
import numpy as np
import scipy.sparse as sp_sparse
import torch
from dgl.dataloading import NeighborSampler, DataLoader


def _prepare_data(adata, n_neighbors=10, spatial_key="spatial", method='sklearn',
                  thres1=0.8, thres2=0.9):
    """Write the spatial kNN index (``obsm['index_neighbor']``) and the token type (``obsm['in_tissue']``).

    ``in_tissue``: 2 = in tissue / cell, 1 = ambiguous, 0 = out of tissue / out-of-cell.
    When absent it is read from ``obs['cell_id']`` or thresholded off the H&E image.
    """
    index_neighbor = np.zeros([adata.n_obs, n_neighbors], dtype=int)
    index = np.arange(adata.n_obs)

    if method == 'sklearn':
        from sklearn.neighbors import NearestNeighbors

        coords = adata.obsm[spatial_key]
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        index_neighbor[index, :] = index[indices[:, 1:]]
    elif method == 'scanpy':
        import scanpy as sc
        from sklearn.neighbors._base import _kneighbors_from_graph

        sub_data = adata.copy()
        sc.pp.neighbors(sub_data, n_neighbors=n_neighbors + 5, use_rep=spatial_key)
        distances = sub_data.obsp["distances"] ** 2
        index_neighbor_batch = _kneighbors_from_graph(distances, n_neighbors, return_distance=False)
        index_neighbor[index, :] = index[index_neighbor_batch]
    else:
        raise NotImplementedError(f'unknown neighbor method {method!r}')

    adata.obsm["index_neighbor"] = index_neighbor

    if "in_tissue" in adata.obsm.keys():
        if adata.obsm["in_tissue"].ndim == 1:
            adata.obsm["in_tissue"] = adata.obsm["in_tissue"].reshape(-1, 1)
    elif 'cell_id' in adata.obs.keys():
        adata.obsm["in_tissue"] = np.where(adata.obs['cell_id'] == 'empty', 0, 2)[:, None].astype(float)
    else:
        img = None
        scl = None
        if "spatial" in adata.uns.keys():
            image_key = list(adata.uns["spatial"].keys())[0]
            images = adata.uns["spatial"][image_key].get("images", {})
            if "hires" in images:
                img = images["hires"]
            elif "lowres" in images:
                img = images["lowres"]
            if img is not None:
                scl = adata.uns["spatial"][image_key].get("scalefactors", {}).get("tissue_hires_scalef", 1.0)

        if img is not None:
            coords_img = np.round(adata.obsm[spatial_key] * scl).astype(int)
            h, w, _ = img.shape
            coords_img[:, 0] = np.clip(coords_img[:, 0], 0, w - 1)
            coords_img[:, 1] = np.clip(coords_img[:, 1], 0, h - 1)
            colors = img[coords_img[:, 1], coords_img[:, 0]] / 255.0
        else:
            # No image: every spot counts as in tissue.
            colors = np.zeros([adata.n_obs, 3])
            thres1 = thres2 = 1.0

        spot_gray = (colors[:, 0] + colors[:, 1] + colors[:, 2]) / 3
        adata.obsm["in_tissue"] = np.where(
            spot_gray < thres1, 2, np.where(spot_gray < thres2, 1, 0)
        )[:, None].astype(float)


def get_dataset(config, adata, force_gpu=True):
    """DGL neighbour-sampling loaders over the spatial kNN graph: ``(train_loader, test_loader)``."""
    keys = ['X', 'in_tissue', config.data.spatial_key]
    _prepare_data(adata,
                  n_neighbors=config.data.n_neighbors,
                  spatial_key=config.data.spatial_key,
                  method=config.data.neighbor_method,
                  thres1=config.data.thres1,
                  thres2=config.data.thres2)

    # Node features: [counts | in_tissue | x, y]
    xs = []
    for key in keys:
        if key == "X":
            arr = adata.X.toarray() if sp_sparse.issparse(adata.X) else adata.X
        else:
            arr = adata.obsm[key]
        xs.append(arr)
    x = torch.tensor(np.hstack(xs), dtype=torch.float)

    n = adata.n_obs
    k = config.data.n_neighbors
    rows = np.repeat(np.arange(n), k)
    cols = adata.obsm["index_neighbor"].reshape(-1)
    mask = rows != cols
    A = sp_sparse.coo_matrix((np.ones_like(rows[mask]), (rows[mask], cols[mask])), shape=(n, n))
    A.sum_duplicates()

    g = dgl.from_scipy(A.tocsr())
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)
    g = dgl.remove_self_loop(g)

    use_cpu = config.data.force_cpu and not force_gpu
    device = 'cpu' if use_cpu else config.device
    g = g.to(device)
    g.ndata['feat'] = x.to(device)

    loaders = []
    for shuffle in (True, False):
        loaders.append(DataLoader(
            g,
            torch.arange(n).to(device),
            NeighborSampler([k]),
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            num_workers=8 if use_cpu else 0,
        ))
    return loaders[0], loaders[1]


def _graph_local_ids(blocks, output_nodes, n_neighbors):
    """Local src indices of each seed and its sampled neighbours, plus the padding mask."""
    global_src = blocks[0].srcdata[dgl.NID]

    max_id = int(global_src.max().item()) + 1
    map_tensor = torch.full((max_id,), -1, dtype=torch.long, device=global_src.device)
    map_tensor[global_src] = torch.arange(global_src.size(0), device=global_src.device)
    seed_local = map_tensor[output_nodes]                       # [B]

    # DGL orders a block's dst nodes first among its src nodes, so seed_local == arange(B).
    src, dst = blocks[0].edges()
    mask = dst.unsqueeze(1) == seed_local.unsqueeze(0)          # [E, B]
    masked_src = src.unsqueeze(1).masked_fill(~mask, -1)

    topk, _ = masked_src.topk(n_neighbors, dim=0)               # [nei, B]
    neigh_valid = topk.t() >= 0
    neigh_local = topk.t().clamp(min=0)
    return seed_local, neigh_local, neigh_valid


def graph_global_ids(blocks, output_nodes, n_neighbors):
    """Global obs index of every token in ``_process_batch_fast`` order, and its validity."""
    global_src = blocks[0].srcdata[dgl.NID]
    seed_local, neigh_local, neigh_valid = _graph_local_ids(blocks, output_nodes, n_neighbors)

    neigh_ids = torch.flip(global_src[neigh_local], dims=[1])
    neigh_ok = torch.flip(neigh_valid, dims=[1])

    seed_ids = output_nodes.reshape(-1, 1).to(neigh_ids.dtype)
    ids = torch.cat([seed_ids, neigh_ids], dim=1)
    valid = torch.cat([torch.ones_like(seed_ids, dtype=torch.bool), neigh_ok], dim=1)
    return ids, valid


def _process_batch_fast(batch, n_neighbors, mask_ratio=0.0):
    """Turn a DGL block into dense neighbourhoods: seed token first, then its neighbours.

    Returns ``(input_nodes, output_nodes, blocks, batch)`` with ``batch`` holding
    ``genes`` (B, N, G), ``in_tissue`` (B, N, 1), ``pos`` (B, N, 2, normalized
    by the mean pairwise distance) and ``mass`` (B, 1, G).
    """
    input_nodes, output_nodes, blocks = batch
    x_src = blocks[0].srcdata['feat']

    seed_local, neigh_local, _ = _graph_local_ids(blocks, output_nodes, n_neighbors)
    batch_neighbors = torch.flip(x_src[neigh_local], dims=[1])
    batch_tensor = torch.cat([x_src[seed_local].unsqueeze(1), batch_neighbors], dim=1)

    if mask_ratio > 0.0:
        # Randomly drop neighbour tokens, keeping the seed.
        num_keep = int((1 - mask_ratio) * (n_neighbors + 1))
        ids_shuffle = torch.rand(batch_tensor.shape[0], n_neighbors, device=batch_tensor.device).argsort(dim=1)
        ids_pivot = torch.zeros(batch_tensor.shape[0], 1, device=batch_tensor.device, dtype=torch.int64)
        ids_keep = torch.cat([ids_pivot, ids_shuffle + 1], dim=-1)[:, :num_keep]
        batch_tensor = torch.gather(
            batch_tensor, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, batch_tensor.shape[2]),
        )

    pos = batch_tensor[:, :, -2:]
    D = torch.cdist(pos, pos, p=2)

    batch = {
        'genes': batch_tensor[:, :, :-3],
        'in_tissue': batch_tensor[:, :, -3:-2],
        'pos': pos / D.mean(),
        'mass': torch.sum(batch_tensor[:, :, :-3], dim=-2, keepdim=True),
    }
    return input_nodes, output_nodes, blocks, batch


def filter_batch_by_cells(batch, min_cells=1, cell_value=2.0):
    """Drop graphs holding fewer than ``min_cells`` cell tokens; returns ``(batch, keep)``."""
    if min_cells <= 0:
        return batch, None

    n_graph = batch['in_tissue'].shape[0]
    keep = (batch['in_tissue'].squeeze(-1) == cell_value).sum(dim=-1) >= min_cells
    if bool(keep.all()):
        return batch, keep

    filtered = {
        key: value[keep] if torch.is_tensor(value) and value.shape[0] == n_graph else value
        for key, value in batch.items()
    }
    return filtered, keep
