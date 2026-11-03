from config.colon_b2c_configs import get_config as _get_colon_b2c_config


def get_config():
    """Stage 2 (PinFuse_Spot) on crop5_large: segmented cells + out-of-cell 2 um bins.

    ``data/testdata/crop5_large_b2c`` holds the cell file and the out-of-cell bin
    file. The frozen stage-1 model is the colon_b2c run; its gene axis is pinned
    through ``data.gene_list``.
    """
    config = _get_colon_b2c_config()

    # data
    config.data.type = 'celllevel'
    config.data.cell_file = 'adata_cell_with_sthd_celltype_coarse.h5ad'
    config.data.spot_file = 'adata_out_outside_cell_2um_bins.h5ad'
    config.data.spatial_key = 'spatial'
    config.data.counts_layer = 'counts'
    config.data.crop_box = ()
    config.data.min_counts = 5.0
    config.data.drop_cell_free = True
    config.data.min_cell_tokens = 1
    config.data.to_csc = True
    config.data.n_neighbors = 120
    config.data.n_top_genes = 0
    config.data.min_cells = 1

    # model
    config.model.type = 'pinfuse_spot'
    config.model.pretrain_ckpt = 'results/colon_b2c_lam5e-2_rerun/checkpoints/checkpoint.pth'
    config.model.freeze_gvae = True
    config.model.hidden_dim = 256
    config.model.latent_dim = 10
    config.model.locality = 0.5
    config.model.stage2_advect_temp = 'learn'
    config.model.stage2_advect_temp_init = 1.0
    config.model.stage2_advect_temp_min = 0.05
    config.model.stage2_diff_min = 0.8
    config.model.lam2 = 1.0
    config.model.lam3 = 0.1
    config.model.lam5 = 0.1
    config.model.flow_dst = True

    # training
    config.training.batch_size = 32
    config.training.epochs = 3

    config.seed = 3035

    return config
