from config.default_configs import get_default_configs


def get_config():
    """10x Visium HD mouse brain, 8 um bins (data/testdata/mousebrain_hd)."""
    config = get_default_configs()

    # data
    config.data.type = 'visiumhd'
    config.data.bin_size = 8
    config.data.spatial_key = 'spatial'
    config.data.n_neighbors = 40
    config.data.min_counts = 10.0
    config.data.min_cells = 1
    config.data.n_top_genes = 2000
    config.data.keep_genes = ['Slc17a7', 'Gad1', 'Mbp', 'C1qa', 'Gfap', 'Aqp4', 'Aldh1l1', 'Hexb',
                              'Csf1r', 'Ctss', 'Trem2', 'Slc17a6', 'Gad2', 'Snap25', 'Sst', 'Pvalb',
                              'Vip', 'Mobp', 'Cldn11', 'Gjc3', 'Olig2', 'Cldn5', 'Flt1']
    config.data.force_cpu = True
    config.data.to_csc = True

    # training
    config.training.batch_size = 64
    config.training.epochs = 3

    # model
    config.model.lam1 = 0.0
    config.model.lam2 = 0.0
    config.model.lam3 = 0.0
    config.model.lam4 = 0.0
    config.model.lam5 = 5e-3

    return config
