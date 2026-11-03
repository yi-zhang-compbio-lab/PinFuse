from config.mousebrain_hd_b2c_configs import get_config as _get_hd_b2c_config


def get_config():
    """Visium HD mouse brain bin2cell cells restricted to the Xenium 5k panel.

        python main.py --config config/mousebrain_hd_stage1_configs.py \\
            --src_dir data/testdata/mousebrain_hd_b2c_5k/adata_cell_xenium5kgenes_celltype.h5ad \\
            --dst_dir results/mousebrain_hd_stage1
    """
    config = _get_hd_b2c_config()
    config.data.keep_genes = ['Slc17a7', 'Gad1', 'Mog', 'Csf1r', 'Gfap', 'Aqp4', 'Aldh1l1', 'Gja1',
                              'Tmem119', 'Trem2', 'Aif1', 'Fcrls', 'Slc17a6', 'Gad2', 'Snap25', 'Sst',
                              'Pvalb', 'Vip', 'Mag', 'Opalin', 'Olig2', 'Sox10', 'Cdh5', 'Pecam1',
                              'Kdr', 'Flt1']
    return config
