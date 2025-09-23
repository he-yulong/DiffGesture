# DiffGesture/scripts/utils/long_pipeline.py
import os
import time
import math
import random
import datetime
import lmdb
import pickle

import numpy as np
import torch

from scripts.utils.pipelines.base_pipeline import BasePipeline
from scripts.utils.test_utils import create_video_and_save
from scripts.utils.data_utils_expressive import convert_pose_seq_to_dir_vec, resample_pose_seq


def generate_gestures(args, diffusion, lang_model, audio, words, pose_dim, audio_sr=16000,
                      seed_seq=None, fade_out=False, device="cpu"):
    """Generate gesture sequences from audio and words."""
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1

    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    num_subdivision = 1 if clip_length < unit_time else math.ceil((clip_length - unit_time) / stride_time) + 1
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    print(f"{num_subdivision}, {unit_time}, {clip_length}, {stride_time}, {audio_sample_length}")

    out_dir_vec = None
    start = time.time()
    for i in range(num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare audio
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), "constant")
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        # prepare text
        word_seq = []
        if words:
            word_seq = [w for w in words if start_time <= w[1] < end_time]
        extended_word_indices = np.zeros(n_frames)
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            print(word[0], end=", ")
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        print(" ")

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1
        pre_seq = pre_seq.float().to(device)

        if args.model == "pose_diffusion":
            out_dir_vec = diffusion.sample(pose_dim, pre_seq, in_audio)

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smooth transition
        if out_list:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev, nxt = last_poses[j], out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + nxt * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print(f"generation took {(time.time() - start) / num_subdivision:.2} s")

    out_dir_vec = np.vstack(out_list)

    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode="constant")
        out_dir_vec[end_frame - n_smooth:] = np.zeros((len(args.mean_dir_vec)))
        # quadratic interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.arange(y.shape[0])
        w = np.ones(len(y))
        w[0] = w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(y.shape[1])]
        interpolated_y = np.transpose([f(x) for f in fit_functions])
        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec


class LongPipeline(BasePipeline):
    """Run long video generation pipeline with random clip sampling."""

    def run(self):
        clip_duration_range = [50, 90]
        n_generations = 5
        n_saved = 0

        with lmdb.open(self.val_data_path, readonly=True, lock=False).begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            while n_saved < n_generations:
                vid, clip = self._sample_clip(txn, keys)
                if not clip:
                    continue
                result = self._process_clip(clip, clip_duration_range)
                if not result:
                    continue
                target_dir_vec, out_dir_vec, clip_audio, clip_time = result
                self._save_result(vid, n_saved, target_dir_vec, out_dir_vec, clip_audio, clip_time)
                n_saved += 1

    # ---------------- private helpers ----------------

    def _sample_clip(self, txn, keys):
        """Pick a random video and clip from LMDB."""
        key = random.choice(keys)
        buf = txn.get(key)
        video = pickle.loads(buf)
        vid, clips = video["vid"], video["clips"]
        if not clips:
            return vid, None
        return vid, random.choice(clips)

    def _process_clip(self, clip, clip_duration_range):
        """Resample poses, filter by duration, normalize words, and generate gestures."""
        clip_poses = clip["skeletons_3d"]
        clip_audio = clip["audio_raw"]
        clip_words = clip["words"]
        clip_time = [clip["start_time"], clip["end_time"]]

        # resample and normalize
        clip_poses = resample_pose_seq(
            clip_poses, clip_time[1] - clip_time[0], self.args.motion_resampling_framerate
        )
        target_dir_vec = convert_pose_seq_to_dir_vec(clip_poses).reshape(clip_poses.shape[0], -1)
        target_dir_vec -= self.mean_dir_vec

        # duration filter
        clip_duration = clip_time[1] - clip_time[0]
        if not (clip_duration_range[0] <= clip_duration <= clip_duration_range[1]):
            return None

        # normalize word times
        for w in clip_words:
            w[1] -= clip_time[0]
            w[2] -= clip_time[0]

        # generate gestures
        out_dir_vec = generate_gestures(
            self.args, self.diffusion, self.lang_model,
            clip_audio, clip_words, self.pose_dim,
            seed_seq=target_dir_vec[: self.args.n_pre_poses],
            fade_out=False,
            device=self.device,
        )
        return target_dir_vec, out_dir_vec, clip_audio, clip_time

    def _save_result(self, vid, idx, target_dir_vec, out_dir_vec, clip_audio, clip_time):
        """Render and save the generated video."""
        aux_str = f"({vid}, time: {datetime.timedelta(seconds=clip_time[0])}" \
                  f"-{datetime.timedelta(seconds=clip_time[1])})"
        mean_data = np.array(self.args.mean_dir_vec).reshape(-1, 3)
        save_path = self.args.model_save_path

        create_video_and_save(
            save_path, idx, "long",
            target_dir_vec, out_dir_vec, mean_data,
            "", audio=clip_audio, aux_str=aux_str,
        )
