from config.xenium_configs import get_config as _get_xenium_config


def get_config():
    """10x Visium HD mouse brain, bin2cell whole-slide cells (data/testdata/mousebrain_hd_b2c)."""
    config = _get_xenium_config()

    # data
    config.data.type = 'adata'
    config.data.spatial_key = 'spatial'
    config.data.counts_layer = ''
    config.data.min_counts = 0.0

    config.data.to_csc = True
    config.data.force_cpu = True
    config.data.n_top_genes = 2000
    config.data.min_cells = 1
    config.data.keep_genes = ['Slc17a7', 'Gad1', 'Mbp', 'C1qa', 'Gfap', 'Aqp4', 'Aldh1l1', 'Hexb',
                              'Csf1r', 'Ctss', 'Trem2', 'Slc17a6', 'Gad2', 'Snap25', 'Sst', 'Pvalb',
                              'Vip', 'Mobp', 'Cldn11', 'Gjc3', 'Olig2', 'Cldn5', 'Flt1']
    config.data.n_neighbors = 80

    # training
    config.training.batch_size = 32
    config.training.epochs = 4

    config.model.lam5 = 5e-2
    return config
