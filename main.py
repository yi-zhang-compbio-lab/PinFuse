"""PinFuse command line: train (or load) a model, predict, and save the denoised data.

    python main.py --config config/xenium_configs.py \
        --src_dir data/xenium_tutorial.h5ad --dst_dir results/xenium
"""
import logging
import os

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from dataio import LOG_DIR, subdir

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration file.", lock_config=True)
flags.DEFINE_string("src_dir", None, "Input data: .h5ad file, STHD patch, Visium HD or cell-level folder.")
flags.DEFINE_string("dst_dir", None, "Output root; receives checkpoints/, artifacts/ and logs/.")
flags.DEFINE_string("ckpt_dir", None, "Pretrained checkpoint (file or run folder); skips training.")
flags.DEFINE_bool("use_pyro", False, "Train the Pyro (SVI) variant of PinFuse_Cell.")
flags.DEFINE_bool("predict", True, "Run prediction after training.")
flags.DEFINE_bool("extra", True, "Run the posterior / count-sampling pass.")
flags.DEFINE_bool("flow", True, "Run the transport-field and source-cell passes.")
flags.DEFINE_bool("save_ckpt", True, "Save the model checkpoint.")
flags.DEFINE_bool("save_adata", True, "Save the predicted AnnData.")
flags.DEFINE_integer("num_sample", 1, "Posterior samples per node used for count sampling.")
flags.DEFINE_enum("count_dist", "poisson", ["poisson", "nb"], "Count distribution used for sampling.")
flags.mark_flags_as_required(["src_dir", "dst_dir", "config"])


def main(argv):
    del argv
    os.makedirs(FLAGS.dst_dir, exist_ok=True)

    handler = logging.StreamHandler(open(os.path.join(subdir(FLAGS.dst_dir, LOG_DIR), 'stdout.txt'), 'w'))
    handler.setFormatter(logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    import train

    train.run(
        FLAGS.config,
        FLAGS.src_dir,
        FLAGS.dst_dir,
        ckpt_dir=FLAGS.ckpt_dir,
        save_ckpt=FLAGS.save_ckpt,
        save_adata=FLAGS.save_adata,
        use_pyro=FLAGS.use_pyro,
        do_predict=FLAGS.predict,
        do_extra=FLAGS.extra,
        do_flow=FLAGS.flow,
        num_sample=FLAGS.num_sample,
        count_dist=FLAGS.count_dist,
    )


if __name__ == "__main__":
    app.run(main)
