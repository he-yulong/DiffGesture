# DiffGesture/scripts/train.py
import torch
import lightning as L
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from utils.vocab_utils import build_vocab
from utils.common import set_random_seed, setup_logger
from data_loader.lmdb_data_loader_expressive import SpeechMotionDataset, default_collate_fn
from train_module import DiffGestureModule
from parse_args_diffusion import parse_args
from lightning.pytorch.loggers import TensorBoardLogger
from scripts.checkpoints import val_checkpoint, train_checkpoint, last_checkpoint


def main(args):
    setup_logger(__file__)

    # Reproducibility
    if getattr(args, "random_seed", -1) >= 0:
        set_random_seed(args.random_seed)

    # Dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(
        args.train_data_path[0],
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        mean_dir_vec=mean_dir_vec,
        mean_pose=args.mean_pose,
        remove_word_timing=(args.input_context == "text"),
    )
    val_dataset = SpeechMotionDataset(  # reserved for FINAL evaluation
        args.val_data_path[0],
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        speaker_model=train_dataset.speaker_model,
        mean_dir_vec=mean_dir_vec,
        mean_pose=args.mean_pose,
        remove_word_timing=(args.input_context == "text"),
    )
    test_dataset = SpeechMotionDataset(  # will be used during training validation
        args.test_data_path[0],
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        speaker_model=train_dataset.speaker_model,
        mean_dir_vec=mean_dir_vec,
        mean_pose=args.mean_pose,
    )

    # build vocab
    vocab_cache_path = Path(args.train_data_path[0]).parent / "vocab_cache.pkl"
    lang_model = build_vocab("words", [train_dataset, val_dataset, test_dataset], vocab_cache_path,
                             args.wordembed_path, args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)
    test_dataset.set_lang_model(lang_model)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0, collate_fn=default_collate_fn)
    # NOTE: using test set for validation during training
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=0, collate_fn=default_collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             drop_last=True, num_workers=0, collate_fn=default_collate_fn)

    # LightningModule
    module = DiffGestureModule(args, lang_model, train_dataset.speaker_model, args.pose_dim)

    # Logger with custom name
    logger = TensorBoardLogger(
        save_dir=args.model_save_path,
        name=args.log_name
    )

    # Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=1,
        default_root_dir=args.model_save_path,
        logger=logger,
        callbacks=[val_checkpoint, train_checkpoint, last_checkpoint],
    )

    trainer.fit(module, train_loader, val_loader)
    # Final evaluation (on val_dataset)
    trainer.test(module, test_loader)


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
