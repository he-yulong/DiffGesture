# DiffGesture/scripts/utils/short_pipeline.py
import logging
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.utils.test_utils import build_dataset, create_video_and_save
from scripts.utils.pipelines.base_pipeline import BasePipeline


class ShortPipeline(BasePipeline):
    """Run short video generation pipeline."""

    def run(self):
        dataset = build_dataset(self.val_data_path, self.args, self.speaker_model, self.mean_pose, self.mean_dir_vec)
        dataset.set_lang_model(self.lang_model)
        loader = DataLoader(
            dataset=dataset,
            batch_size=32,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        self._save_video(loader)

    def _save_video(self, test_data_loader, n_save=5):
        """Generate short evaluation videos from test set."""
        with torch.no_grad():
            for iter_idx, data in enumerate(test_data_loader, 0):
                if iter_idx >= n_save:
                    break
                logging.info(f"testing {iter_idx}/{len(test_data_loader)}")
                # Step 1. Prepare inputs
                in_text_padded, in_audio, in_spec, target_dir_vec, aux_info = \
                    self._prepare_inputs(data)
                # Step 2. Generate gestures
                out_dir_vec = self._generate_output(in_audio, target_dir_vec)
                # Step 3. Postprocess results
                audio_npy, target_dir_vec_np, out_dir_vec_np = \
                    self._postprocess_results(in_audio, target_dir_vec, out_dir_vec)
                # Step 4. Rebuild sentence and metadata
                sentence = self._build_sentence(in_text_padded)
                aux_str = self._build_aux_info(aux_info)
                # Step 5. Save video
                self._render_video(iter_idx, target_dir_vec_np, out_dir_vec_np,
                                   sentence, audio_npy, aux_str)

    def _prepare_inputs(self, data):
        """Extract and prepare input tensors from a single batch."""
        _, _, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
        select_index = 0

        in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(self.device)
        in_audio = in_audio[select_index, :].unsqueeze(0).to(self.device)
        in_spec = in_spec[select_index, :].unsqueeze(0).to(self.device)
        target_dir_vec = target_vec[select_index, :].unsqueeze(0).to(self.device)

        return in_text_padded, in_audio, in_spec, target_dir_vec, aux_info

    def _generate_output(self, in_audio, target_dir_vec):
        """Run diffusion model to generate gesture outputs."""
        pre_seq = target_dir_vec.new_zeros(
            (target_dir_vec.shape[0], target_dir_vec.shape[1], target_dir_vec.shape[2] + 1)
        )
        pre_seq[:, 0:self.args.n_pre_poses, :-1] = target_dir_vec[:, 0:self.args.n_pre_poses]
        pre_seq[:, 0:self.args.n_pre_poses, -1] = 1  # indicating bit for constraints

        assert self.args.model == "pose_diffusion"
        return self.diffusion.sample(self.pose_dim, pre_seq, in_audio)

    def _postprocess_results(self, in_audio, target_dir_vec, out_dir_vec):
        """Convert tensors to numpy arrays."""
        audio_npy = np.squeeze(in_audio.cpu().numpy())
        target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
        out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())
        return audio_npy, target_dir_vec, out_dir_vec

    def _build_sentence(self, in_text_padded):
        """Convert padded text indices into a readable sentence."""
        words = []
        for i in range(in_text_padded.shape[1]):
            word_idx = int(in_text_padded.data[0, i])
            if word_idx > 0:
                words.append(self.lang_model.index2word[word_idx])
        return " ".join(words)

    def _build_aux_info(self, aux_info):
        """Format video metadata string."""
        return "({}, time: {}-{})".format(
            aux_info["vid"][0],
            str(datetime.timedelta(seconds=aux_info["start_time"][0].item())),
            str(datetime.timedelta(seconds=aux_info["end_time"][0].item())),
        )

    def _render_video(self, iter_idx, target_dir_vec, out_dir_vec, sentence, audio_npy, aux_str):
        """Call video renderer to save output with audio."""
        mean_data = np.array(self.args.mean_dir_vec).reshape(-1, 3)
        save_path = self.args.model_save_path
        create_video_and_save(
            save_path, iter_idx, "short",
            target_dir_vec, out_dir_vec, mean_data,
            sentence, audio=audio_npy, aux_str=aux_str,
        )
