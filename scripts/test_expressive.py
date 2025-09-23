import os
import sys
import torch

from data_loader.lmdb_data_loader_expressive import default_collate_fn
from utils.common import setup_logger
from utils.test_utils import restore_experiment, load_mean_vectors, load_language_model
from scripts.utils.pipelines.pipeline_factory import PipelineFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main(mode, checkpoint_path):
    setup_logger()
    args, diffusion, lang_model, speaker_model, pose_dim = restore_experiment(checkpoint_path, device)
    mean_pose, mean_dir_vec = load_mean_vectors(args)
    vocab_cache_path = os.path.join("data/ted_expressive_dataset", "vocab_cache.pkl")
    lang_model = load_language_model(vocab_cache_path)
    collate_fn = default_collate_fn
    pipeline = PipelineFactory.create(
        mode, args, diffusion, lang_model, speaker_model,
        mean_pose, mean_dir_vec, pose_dim, collate_fn, device
    )
    pipeline.run()


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ["eval", "short", "long"]
    ckpt_path = 'output/train_diffusion_expressive/pose_diffusion_checkpoint_499.bin'
    main(mode, ckpt_path)
