from config.mousebrain_xenium_stage1_configs import get_config as _get_stage1_config


def get_config():
    """Stage 2 (PinFuse_Spot) on the joint cell + out-of-cell molecule mouse_brain_xenium crop."""
    config = _get_stage1_config()

    # data: the joint cell + out-of-cell object
    config.data.type = 'celllevel'
    config.data.spot_file = 'adata_out_raw_outside_cell_molecules.h5ad'
    config.data.min_counts = 1.0
    config.data.drop_cell_free = True
    config.data.min_cell_tokens = 1

    # model
    config.model.type = 'pinfuse_spot'
    config.model.pretrain_ckpt = 'results/mousebrain_cellonly/checkpoints/checkpoint.pth'
    config.model.freeze_gvae = True
    config.model.lam1 = 0.0
    config.model.lam2 = 1.0
    config.model.lam3 = 0.0
    config.model.lam5 = 0.1

    return config
