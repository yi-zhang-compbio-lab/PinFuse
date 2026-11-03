from config.xenium_configs import get_config as _get_xenium_config


def get_config():
    """Stage 1 (PinFuse_Cell) on the cells of the mouse_brain_xenium crop.

    ``data/testdata/mouse_brain_xenium`` holds the whole-slide cells and the
    out-of-cell molecules; this config reads the cell file only.
    """
    config = _get_xenium_config()

    # data
    config.data.type = 'adata'
    config.data.cell_file = 'adata_cell_with_celltype.h5ad'
    config.data.spot_file = ''
    config.data.spatial_key = 'spatial'
    config.data.counts_layer = ''
    config.data.crop_box = (1000.0, 5000.0, 500.0, 4500.0)
    config.data.min_counts = 0.0
    config.data.drop_cell_free = False
    config.data.min_cell_tokens = 0
    config.data.to_csc = True

    # model
    config.model.type = 'pinfuse_cell'
    config.model.hidden_dim = 256
    config.model.latent_dim = 10
    config.model.locality = 0.5
    config.model.lam1 = 1e-4
    config.model.lam2 = 0.0
    config.model.lam3 = 1e-2
    config.model.lam4 = 0.0
    config.model.lam5 = 1e-1

    # training
    config.training.batch_size = 32
    config.training.epochs = 3
    config.optim.lr = 1e-3

    return config
