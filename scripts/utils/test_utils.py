# DiffGesture/scripts/utils/test_utils.py
import os
import logging
import torch
import pprint
import pickle
import time
import numpy as np
from scripts.model.pose_diffusion import PoseDiffusion
from scripts.utils.common import set_random_seed
from scripts.data_loader.lmdb_data_loader import SpeechMotionDataset, default_collate_fn
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
from textwrap import wrap
from scripts.utils.data_utils_expressive import convert_dir_vec_to_pose, dir_vec_pairs


def create_video_and_save(save_path, iter_idx, prefix, target, output, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
    """Render and save a side-by-side video (target vs generated)."""
    print("rendering a video...")
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d")]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title
    if aux_str:
        fig_title += "\n" + aux_str
    fig.suptitle("\n".join(wrap(fig_title, 75)), fontsize="medium")

    # unnormalize + convert
    mean_data = mean_data.flatten()
    output = output + mean_data
    output_poses = convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        target_poses = convert_dir_vec_to_pose(target)

    def animate(i):
        for k, name in enumerate(["human", "generated"]):
            pose = None
            if name == "human" and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == "generated" and i < len(output):
                pose = output_poses[i]
            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(dir_vec_pairs):
                    axes[k].plot(
                        [pose[pair[0], 0], pose[pair[1], 0]],
                        [pose[pair[0], 2], pose[pair[1], 2]],
                        [pose[pair[0], 1], pose[pair[1], 1]],
                        zdir="z", linewidth=1.5
                    )
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_title(f"{name} ({i + 1}/{len(output)})")

    num_frames = max(len(target) if target is not None else 0, len(output))
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # audio
    audio_path = None
    if audio is not None:
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = f"{save_path}/{iter_idx}.wav"
        sf.write(audio_path, audio, sr)

    # video
    video_path = f"{save_path}/temp_{iter_idx}.mp4"
    ani.save(video_path, fps=15, dpi=80)
    plt.close(fig)

    if audio is not None:
        merged_video_path = f"{save_path}/{prefix}_{iter_idx}.mp4"
        cmd = ["ffmpeg", "-loglevel", "panic", "-y", "-i", video_path, "-i", audio_path, "-strict", "-2",
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, "-shortest")
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print(f"done, took {time.time() - start:.1f} seconds")
    return output_poses, target_poses


def restore_experiment(checkpoint_path, device="cpu", resume_training=False, manual_seed=123):
    """Restore a PoseDiffusion experiment from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract fields
    args, state_dict = _validate_checkpoint(checkpoint)
    epoch = checkpoint.get("epoch")
    lang_model = checkpoint.get("lang_model")
    speaker_model = checkpoint.get("speaker_model")
    pose_dim = checkpoint.get("pose_dim")

    # Reproducibility
    if getattr(args, "random_seed", -1) >= 0 and resume_training:
        set_random_seed(args.random_seed)
    elif manual_seed is not None:
        set_random_seed(manual_seed)

    # Logging
    _log_environment_info(device, args, checkpoint_path, epoch)

    # Initialize model
    diffusion_model = _initialize_model(args, state_dict, device)

    return args, diffusion_model, lang_model, speaker_model, pose_dim


def load_mean_vectors(args):
    """Extract mean pose and mean direction vectors from args."""
    mean_pose = np.array(args.mean_pose).squeeze()
    mean_dir_vec = np.array(args.mean_dir_vec).squeeze()
    return mean_pose, mean_dir_vec


def build_dataset(path, args, speaker_model, mean_pose, mean_dir_vec):
    """Helper to construct a SpeechMotionDataset with consistent settings."""
    dataset = SpeechMotionDataset(
        path,
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        speaker_model=speaker_model,
        mean_pose=mean_pose,
        mean_dir_vec=mean_dir_vec,
    )
    logging.info(f"Loaded dataset from {path} with {len(dataset)} samples")
    return dataset


def load_language_model(vocab_cache_path):
    """Load a cached language model from disk."""
    with open(vocab_cache_path, "rb") as f:
        lang_model = pickle.load(f)
    return lang_model


def _log_environment_info(device, args, checkpoint_path, epoch):
    """Log environment and experiment details."""
    logging.debug(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"{torch.cuda.device_count()} GPUs, default device: {device}")
    logging.info("Experiment arguments:\n" + pprint.pformat(vars(args)))
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    logging.info(f"Checkpoint epoch: {epoch}")


def _validate_checkpoint(checkpoint):
    """Validate checkpoint structure and extract required fields."""
    args = checkpoint.get("args")
    state_dict = checkpoint.get("state_dict")
    if args is None or state_dict is None:
        raise ValueError("Checkpoint is missing required keys: 'args' or 'state_dict'")
    return args, state_dict


def _initialize_model(args, state_dict, device):
    """Initialize PoseDiffusion model and load weights."""
    model = PoseDiffusion(args).to(device)
    model.load_state_dict(state_dict)
    logging.info("PoseDiffusion model initialized and weights loaded.")
    return model
