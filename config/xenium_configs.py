from config.default_configs import get_default_configs


def get_config():
    """Xenium mouse-brain tutorial dataset (data/xenium_tutorial.h5ad), left hemisphere.

        python main.py --config config/xenium_configs.py \\
            --src_dir data/xenium_tutorial.h5ad --dst_dir results/xenium
    """
    config = get_default_configs()

    # data
    config.data.type = 'adata'
    config.data.spatial_key = 'X_spatial'
    config.data.n_neighbors = 80
    config.data.counts_layer = 'counts'
    config.data.min_counts = 0.0
    config.data.crop_box = (129.0, 5000.0, 328.0, 7033.0)   # left hemisphere (x < 5000)
    config.data.to_csc = False

    # training
    config.training.batch_size = 32
    config.training.epochs = 4

    # model
    config.model.lam1 = 0.0
    config.model.lam2 = 0.0
    config.model.lam3 = 0.0
    config.model.lam4 = 0.0
    config.model.lam5 = 5e-2

    return config
