from config.mousebrain_hd_b2c_configs import get_config as _get_b2c_config


def get_config():
    """Visium HD colon cancer bin2cell whole-slide cells (data/testdata/colon_cancer_b2c).

    Stage 1 of the crop5_large two-stage run (see crop5_large_stage2_configs).
    """
    config = _get_b2c_config()

    config.data.counts_layer = 'counts'
    config.data.min_counts = 5.0
    config.data.n_top_genes = 2000
    config.data.keep_genes = ['EPCAM', 'IGKC', 'CD74', 'TMSB10', 'MS4A1', 'CD79A', 'CD19', 'CD37',
                              'PECAM1', 'VWF', 'PLVAP', 'CDH5', 'ENG', 'COL1A1', 'COL1A2', 'COL3A1',
                              'DCN', 'LUM', 'THY1', 'LYZ', 'C1QA', 'C1QB', 'AIF1', 'TYROBP', 'FCER1G',
                              'CD68', 'IGHG1', 'IGHA1', 'JCHAIN', 'MZB1', 'ACTA2', 'MYH11', 'TAGLN',
                              'DES', 'CD3D', 'CD3E', 'CD2', 'TRAC', 'IL7R', 'CD8A', 'KRT8', 'KRT18',
                              'CEACAM5', 'LGALS4', 'PHGR1', 'OLFM4', 'REG4', 'TFF3', 'MUC2']

    config.training.epochs = 2

    return config
