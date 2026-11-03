import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data
    config.data = data = ml_collections.ConfigDict()
    data.type = 'auto'              # 'auto' | 'sthd' | 'adata' | 'visiumhd' | 'celllevel'
    data.spatial_key = 'spatial'
    data.n_neighbors = 80           # graph size: seed + n_neighbors tokens
    data.neighbor_method = 'sklearn'
    data.mask_ratio = 0.0           # random neighbour dropout during training
    data.thres1 = 0.8               # H&E gray level: < thres1 in tissue,
    data.thres2 = 0.9               #   < thres2 ambiguous, else out of tissue
    data.force_cpu = False          # keep prediction buffers in host RAM

    # preprocessing
    data.counts_layer = ''          # if set, adata.X = adata.layers[counts_layer]
    data.min_counts = 1.0           # drop spots below this total count
    data.subset_key = ''            # obs column used to subset the dataset
    data.subset_value = ''
    data.crop_box = ()              # (x1, x2, y1, y2) rectangular window; () disables
    data.min_cells = 0              # drop genes seen in fewer spots
    data.n_top_genes = 0            # keep the top HVGs (seurat_v3); 0 disables
    data.keep_genes = []            # genes always kept by the HVG selection
    data.gene_list = ''             # pinned gene axis (overrides min_cells / n_top_genes)
    data.to_csc = True              # store sampled counts sparsely

    # cell-level sources (data.type == 'celllevel')
    data.cell_file = 'adata_cell.h5ad'
    data.spot_file = ''
    data.min_cell_tokens = 0        # PinFuse_Spot: skip graphs with fewer cell tokens
    data.drop_cell_free = False     # PinFuse_Spot: drop nodes whose graph holds no cell

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.epochs = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.type = 'pinfuse_cell'     # 'pinfuse_cell' (stage 1) | 'pinfuse_spot' (stage 2)
    model.hidden_dim = 256
    model.latent_dim = 10
    model.latent_dim_nb = 5
    model.locality = 0.5            # distance quantile defining each token's attention graph
    model.train_beta = True
    model.n_step = 1                # Lie-Trotter steps of the transport (beta ** n_step)
    model.gene_disper = False
    model.gene_dropout = False

    # loss weights
    model.lam1 = 1e-4               # out-of-tissue rate penalty (PinFuse_Cell only)
    model.lam2 = 0.0                # latent KL
    model.lam3 = 1e-2               # transport distance penalty
    model.lam4 = 0.0                # transport entropy penalty (> 0 flattens, < 0 sharpens)
    model.lam5 = 1e-4               # pushes the transported share (prop / leak) toward 1
    model.kl_type = 'mog'           # 'mog' mixture prior | 'std' N(0, I)
    model.zinb_eps = 1e-8           # PinFuse_Cell ZINB stability constant
    model.mol_loss_weight = 1.0     # PinFuse_Spot: weight of molecule tokens in the likelihood

    # PinFuse_Cell (stage 1) transport operator (flags also restored from the checkpoint by PinFuse_Spot)
    model.stage1_qk_norm = False    # layer-norm the advection query / key
    model.stage1_advect_temp = None # advection temperature; None: 1 with qk_norm, else 10
    model.stage1_diff_kernel = 'exp'      # diffusivity map: 'softplus' | 'exp'
    model.stage1_diff_min = 0.0     # additive floor on the diffusivity

    # PinFuse_Spot (stage 2)
    model.pretrain_ckpt = ''        # stage-1 checkpoint (file or run folder)
    model.freeze_gvae = True
    model.stage1_locality = None    # stage-1 locality, only needed for bare state dicts
    model.beta_mix = 'leak'         # 'leak' | 'compose' | 'leak_molecule' | 'retrain_prop' | 'retrain_leak' | 'renorm'
    model.gene_leak = True          # per-gene (vs per-cell) leak
    model.share_prior = True        # molecule latent shares the stage-1 mixture prior
    model.stage2_flow_dst = True    # transport reads destination features + joint latent
    model.stage2_mask_diffusion = True    # restrict the diffusion kernel to the locality graph
    model.stage2_qk_norm = True
    model.stage2_advect_temp = None       # float, None (derived from qk_norm) or 'learn'
    model.stage2_advect_temp_init = 1.0
    model.stage2_advect_temp_min = 0.05
    model.stage2_diff_min = 0.0     # diffusivity floor, in normalized local D^2 units (~0.2-1.0)
    model.stage2_mol_latent = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 1e-3
    optim.grad_clip = 5.

    return config


def get_config():
    return get_default_configs()
