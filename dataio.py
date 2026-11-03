"""Dataset IO for PinFuse.

:class:`SpatialData` wraps the supported input formats:

* ``sthd``      : an STHD patch folder (needs the ``STHD`` package).
* ``adata``     : a plain ``.h5ad`` file, or a folder containing one.
* ``visiumhd``  : a 10x Visium HD output folder (``binned_outputs/``), read with
  ``spatialdata_io`` at ``data.bin_size`` and cached as ``.h5ad``.
* ``celllevel`` : a folder with a cell ``.h5ad`` and an out-of-cell ``.h5ad``
  (``data.cell_file`` / ``data.spot_file``), concatenated and tagged through
  ``obsm['in_tissue']`` (2.0 = cell, 0.0 = out-of-cell).
"""

import glob
import json
import logging
import os

import numpy as np

STHD_KIND = 'sthd'
ADATA_KIND = 'adata'
VISIUMHD_KIND = 'visiumhd'
CELLLEVEL_KIND = 'celllevel'

ADATA_FILENAME = 'adata_denoised.h5ad'
MARKER_FILENAME = 'pinfuse_data.json'
CONFIG_FILENAME = 'config.json'
BINNED_OUTPUTS_DIR = 'binned_outputs'

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
ARTIFACT_DIR = 'artifacts'


def subdir(root, name):
    """Return ``<root>/<name>``, creating it when needed."""
    path = os.path.join(os.path.abspath(root), name)
    os.makedirs(path, exist_ok=True)
    return path


def cfg_get(config_obj, path, default):
    """Read a dotted config path, falling back to ``default`` when missing."""
    current = config_obj
    for key in path.split('.'):
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def save_config(config, dst_dir, dirname=ARTIFACT_DIR, filename=CONFIG_FILENAME):
    """Write the resolved config of a run as JSON."""
    path = os.path.join(subdir(dst_dir, dirname), filename)
    payload = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2, sort_keys=True, default=str)
    logging.info(f"config saved: {path}")
    return path


def to_numpy(arr):
    import torch

    if hasattr(arr, 'toarray'):
        return np.asarray(arr.toarray())
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _split_gene_names(value):
    """Accept either a comma-separated string or a list of gene names."""
    if not value:
        return []
    items = value.split(',') if isinstance(value, str) else list(value)
    return [str(gene).strip() for gene in items if str(gene).strip()]


def pin_indexer(var_names, genes):
    """Column indices selecting ``genes`` from ``var_names`` in the list's order."""
    seen, dupes = set(), []
    for gene in genes:
        if gene in seen and gene not in dupes:
            dupes.append(gene)
        seen.add(gene)
    if dupes:
        raise ValueError(f"data.gene_list lists {len(dupes)} duplicate gene(s), e.g. {dupes[:10]}")
    return np.asarray(var_names.get_indexer(list(genes)), dtype=int)


def _read_marker(path):
    marker = os.path.join(path, MARKER_FILENAME)
    if os.path.isfile(marker):
        try:
            with open(marker, encoding='utf-8') as fp:
                return json.load(fp)
        except (OSError, ValueError) as err:
            logging.warning(f"Failed to read {marker}: {err}")
    return {}


def _find_h5ad(directory):
    for name in (ADATA_FILENAME, 'adata.h5ad.gzip', 'adata.h5ad'):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    matches = sorted(glob.glob(os.path.join(directory, '*.h5ad*')))
    return matches[0] if matches else None


def resolve_data_dir(src):
    """Point at the folder holding the data, following the ``artifacts/`` layout of a run."""
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        return src
    artifacts = os.path.join(src, ARTIFACT_DIR)
    if os.path.isdir(artifacts) and (_read_marker(artifacts) or _find_h5ad(artifacts)):
        return artifacts
    return src


def detect_kind(src):
    """Guess the dataset format from a path."""
    src = resolve_data_dir(src)
    if os.path.isfile(src):
        return ADATA_KIND
    if not os.path.isdir(src):
        raise FileNotFoundError(f"dataset not found: {src}")

    marker = _read_marker(src)
    if marker.get('kind') in (STHD_KIND, ADATA_KIND, VISIUMHD_KIND, CELLLEVEL_KIND):
        return marker['kind']
    if os.path.isfile(os.path.join(src, 'fullresimg_path.json')):
        return STHD_KIND
    if os.path.isdir(os.path.join(src, BINNED_OUTPUTS_DIR)):
        return VISIUMHD_KIND
    return ADATA_KIND


class SpatialData:
    """Uniform view over an STHD patch or a plain AnnData object."""

    def __init__(self, adata, kind, sthdata=None, source=None):
        self.adata = adata
        self.kind = kind
        self.sthdata = sthdata
        self.source = source

    @property
    def shape(self):
        return self.adata.X.shape

    def name(self):
        base = os.path.basename(os.path.abspath(str(self.source).rstrip('/')))
        return os.path.splitext(base)[0] if base else self.kind

    def save(self, dst_dir, descr=None):
        os.makedirs(dst_dir, exist_ok=True)
        dst_dir = os.path.abspath(dst_dir)
        if descr is not None:
            self.adata.uns['descr'] = descr

        if self.kind == STHD_KIND and self.sthdata is not None:
            self.sthdata.adata = self.adata
            self.sthdata.save(dst_dir)
            saved, filename, saved_kind = dst_dir, None, STHD_KIND
        else:
            saved = os.path.join(dst_dir, ADATA_FILENAME)
            self.adata.write(saved)
            filename, saved_kind = ADATA_FILENAME, ADATA_KIND

        with open(os.path.join(dst_dir, MARKER_FILENAME), 'w', encoding='utf-8') as fp:
            json.dump({'kind': saved_kind, 'file': filename, 'spatial_key': 'spatial'}, fp, indent=2)

        logging.info(f"data saved: {saved}")
        return saved


def _read_adata(src):
    import anndata

    path = src if os.path.isfile(src) else _find_h5ad(src)
    if path is None:
        raise FileNotFoundError(f"no .h5ad file found under {src}")
    logging.info(f"reading AnnData: {path}")
    return anndata.read_h5ad(path)


def _read_sthd(src):
    from STHD import train as STHDtrain

    logging.info(f"reading STHD patch: {src}")
    return STHDtrain.load_data(src)


def _read_visium_hd(config, src):
    """Read one bin size of a 10x Visium HD run, cached as ``.h5ad`` next to the data."""
    import anndata

    bin_size = int(cfg_get(config, 'data.bin_size', 8))
    cache = os.path.join(src, f"pinfuse_bin{bin_size:03d}um.h5ad")
    if os.path.isfile(cache):
        logging.info(f"reading cached Visium HD table: {cache}")
        return anndata.read_h5ad(cache)

    import spatialdata_io

    logging.info(f"reading Visium HD ({bin_size}um bins): {src}")
    sdata = spatialdata_io.visium_hd(src, bin_size=bin_size, load_all_images=False)
    adata = sdata.tables[f"square_{bin_size:03d}um"].copy()
    del sdata

    adata.uns.pop('spatialdata_attrs', None)
    adata.obs.drop(columns=[c for c in ('region',) if c in adata.obs], inplace=True)
    adata.var_names_make_unique()

    if 'in_tissue' in adata.obs:
        adata.obsm['in_tissue'] = (np.asarray(adata.obs['in_tissue'], dtype=float) * 2.0).reshape(-1, 1)
    else:
        adata.obsm['in_tissue'] = np.full((adata.n_obs, 1), 2.0)

    adata.write_h5ad(cache)
    logging.info(f"cached Visium HD table: {cache}")
    return adata


def _crop_box(adata, box, spatial_key='spatial', tag=''):
    """Keep the spots inside the ``(x1, x2, y1, y2)`` window, if one is given."""
    if not box:
        return adata
    if len(box) != 4:
        raise ValueError(f"data.crop_box must hold 4 numbers (x1, x2, y1, y2), got {list(box)}")
    x1, x2, y1, y2 = (float(v) for v in box)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    keep = ((coords[:, 0] > x1) & (coords[:, 0] < x2)
            & (coords[:, 1] > y1) & (coords[:, 1] < y2))
    logging.info(f"crop_box{f' [{tag}]' if tag else ''} "
                 f"x=({x1:g}, {x2:g}) y=({y1:g}, {y2:g}): {int(keep.sum())} / {adata.n_obs} spots")
    return adata[keep].copy()


def _read_celllevel(config, src):
    """Concatenate the cell and out-of-cell halves of a cell-level dataset (cropped first)."""
    import anndata

    cell_file = str(cfg_get(config, 'data.cell_file', 'adata_cell.h5ad'))
    spot_file = str(cfg_get(config, 'data.spot_file', ''))
    if not spot_file:
        raise ValueError("celllevel data needs data.spot_file (the out-of-cell .h5ad)")

    spatial_key = str(cfg_get(config, 'data.spatial_key', 'spatial'))
    box = list(cfg_get(config, 'data.crop_box', []) or [])
    counts_layer = str(cfg_get(config, 'data.counts_layer', ''))

    parts = []
    for path, in_tissue, tag in ((cell_file, 2.0, 'cell'), (spot_file, 0.0, 'out-of-cell')):
        full = path if os.path.isabs(path) else os.path.join(src, path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"celllevel {tag} file not found: {full}")
        logging.info(f"reading {tag} AnnData: {full}")
        adata = anndata.read_h5ad(full)
        if counts_layer and counts_layer in adata.layers:
            adata.X = adata.layers[counts_layer]
        adata.layers.clear(keep_x=True)
        adata = _crop_box(adata, box, spatial_key=spatial_key, tag=tag)
        adata.obs['in_tissue'] = in_tissue
        parts.append(adata)

    merged = anndata.concat(parts, axis=0, join='outer', label='level',
                            keys=['cell', 'out'], index_unique='-')
    merged.obsm['in_tissue'] = np.asarray(merged.obs['in_tissue'], dtype=float).reshape(-1, 1)
    if counts_layer:
        merged.layers[counts_layer] = merged.X
    n_cell, n_spot = (part.n_obs for part in parts)
    logging.info(f"celllevel merge: {n_cell} cells + {n_spot} out-of-cell spots "
                 f"= {merged.n_obs} obs x {merged.n_vars} genes")
    return merged


def _apply_preprocessing(config, adata, is_saved_run=False):
    """Apply the config's preprocessing to a freshly read ``adata``.

    Row filters are skipped on a saved run (``is_saved_run``): it is already
    filtered and carries obs-index-valued fields that re-filtering would break.
    """
    counts_layer = cfg_get(config, 'data.counts_layer', '')
    if counts_layer:
        if counts_layer not in adata.layers:
            raise KeyError(f"layer '{counts_layer}' not found in adata.layers")
        adata.X = adata.layers[counts_layer]
        logging.info(f"using counts from layer '{counts_layer}'")

    spatial_key = cfg_get(config, 'data.spatial_key', 'spatial')
    if spatial_key not in adata.obsm:
        raise KeyError(f"spatial key '{spatial_key}' not found in adata.obsm")

    adata = _crop_box(adata, list(cfg_get(config, 'data.crop_box', []) or []),
                      spatial_key=spatial_key)

    subset_key = cfg_get(config, 'data.subset_key', '')
    subset_value = cfg_get(config, 'data.subset_value', '')
    if subset_key and not is_saved_run:
        if subset_key not in adata.obs:
            raise KeyError(f"obs column '{subset_key}' not found for subsetting")
        keep = adata.obs[subset_key].astype(str) == str(subset_value)
        logging.info(f"subset {subset_key}=={subset_value}: {int(keep.sum())} / {adata.n_obs} spots")
        adata = adata[np.asarray(keep)].copy()

    min_counts = float(cfg_get(config, 'data.min_counts', 5))
    if min_counts > 0 and not is_saved_run:
        totals = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        keep = totals >= min_counts
        logging.info(f"min_counts={min_counts:g}: {int(keep.sum())} / {adata.n_obs} spots")
        adata = adata[keep].copy()

    # Pinned gene axis: stage 2 must see exactly stage 1's genes, in order.
    pinned = _split_gene_names(cfg_get(config, 'data.gene_list', ''))
    if pinned:
        missing = [gene for gene in pinned if gene not in adata.var_names]
        if missing:
            raise KeyError(f"data.gene_list: {len(missing)} of {len(pinned)} genes are absent "
                           f"from the data, e.g. {missing[:10]}")
        n_gene = adata.n_vars
        adata = adata[:, pin_indexer(adata.var_names, pinned)].copy()
        logging.info(f"data.gene_list: pinned axis to {adata.n_vars} / {n_gene} genes")

    min_cells = float(cfg_get(config, 'data.min_cells', 0))
    if min_cells > 0 and not pinned:
        import scanpy as sc

        n_gene = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells)
        logging.info(f"min_cells={min_cells:g}: {adata.n_vars} / {n_gene} genes")

    n_top_genes = int(cfg_get(config, 'data.n_top_genes', 0))
    if not pinned and 0 < n_top_genes < adata.n_vars and 'hvg' not in adata.uns:
        import scanpy as sc

        n_gene = adata.n_vars
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        keep_genes = np.asarray(adata.var['highly_variable'])
        forced = [g for g in dict.fromkeys(_split_gene_names(cfg_get(config, 'data.keep_genes', '')))
                  if g in adata.var_names]
        if forced:
            keep_genes = keep_genes | adata.var_names.isin(forced)
        adata = adata[:, keep_genes].copy()
        logging.info(f"n_top_genes={n_top_genes}: {adata.n_vars} / {n_gene} genes "
                     f"(incl. {len(forced)} forced data.keep_genes)")

    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = np.asarray(adata.obsm[spatial_key])

    return adata


def from_anndata(config, adata, preprocess=True, name='adata'):
    """Wrap an in-memory AnnData as :class:`SpatialData`, applying the config's preprocessing.

    The input is not modified; preprocessing works on a copy.
    """
    adata = adata.copy()
    if preprocess:
        adata = _apply_preprocessing(config, adata)
    logging.info(f"prepared in-memory AnnData: n_obs={adata.n_obs}, n_gene={adata.n_vars}")
    return SpatialData(adata, ADATA_KIND, source=name)


def load_data(config, src, preprocess=True):
    """Load ``src`` into a :class:`SpatialData` wrapper."""
    src = os.path.abspath(src)
    resolved = resolve_data_dir(src)

    kind = _read_marker(resolved).get('kind', '')
    if kind not in (STHD_KIND, ADATA_KIND, VISIUMHD_KIND, CELLLEVEL_KIND):
        kind = str(cfg_get(config, 'data.type', 'auto'))
        if kind == 'auto':
            kind = detect_kind(src)

    sthdata = None
    if kind == STHD_KIND:
        try:
            sthdata = _read_sthd(resolved)
            adata = sthdata.adata
        except ImportError as err:
            logging.warning(f"STHD package unavailable ({err}); falling back to plain AnnData loading")
            kind = ADATA_KIND
            adata = _read_adata(resolved)
    elif kind == VISIUMHD_KIND:
        adata = _read_visium_hd(config, resolved)
    elif kind == CELLLEVEL_KIND:
        adata = _read_celllevel(config, resolved)
    else:
        adata = _read_adata(resolved)

    if preprocess:
        is_saved_run = (
            os.path.abspath(resolved) != os.path.abspath(src)
            or os.path.isfile(os.path.join(resolved, MARKER_FILENAME))
        )
        adata = _apply_preprocessing(config, adata, is_saved_run=is_saved_run)
        if sthdata is not None:
            sthdata.adata = adata

    data = SpatialData(adata, kind, sthdata=sthdata, source=resolved)
    logging.info(f"loaded {kind} data from {resolved}: n_obs={adata.n_obs}, n_gene={adata.n_vars}")
    return data
