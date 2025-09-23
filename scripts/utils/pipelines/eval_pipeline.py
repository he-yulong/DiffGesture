# DiffGesture/scripts/utils/eval_pipeline.py
import logging
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
from scripts.utils.pipelines.base_pipeline import BasePipeline
from scripts.utils.average_meter import AverageMeter
from scripts.model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from scripts.utils.data_utils_expressive import convert_dir_vec_to_pose
from scripts.utils.test_utils import build_dataset

class EvalPipeline(BasePipeline):
    """Run evaluation with metrics and embedding space evaluator."""

    angle_pair = [
        (0, 1), (0, 2), (1, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
        (11, 12), (12, 13), (14, 15), (15, 16), (17, 18), (18, 19),
        (17, 5), (5, 8), (8, 14), (14, 11), (2, 20), (20, 21), (22, 23),
        (23, 24), (25, 26), (26, 27), (28, 29), (29, 30), (31, 32),
        (32, 33), (34, 35), (35, 36), (34, 22), (22, 25), (25, 31),
        (31, 28), (0, 37), (37, 38), (37, 39), (38, 40), (39, 41),
        (4, 42), (21, 43)
    ]

    change_angle = [0.0027804733254015446, 0.002761547453701496, 0.005953566171228886, 0.013764726929366589,
                    0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386,
                    0.0485558956861496,
                    0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707,
                    0.01640886254608631,
                    0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088,
                    0.005681346170604229,
                    0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383,
                    0.03378918394446373,
                    0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834,
                    0.02628341130912304,
                    0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141,
                    0.0014634154504165053,
                    0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285,
                    0.0012279648799449205,
                    0.047324635088443756, 0.04472292214632034]

    sigma = 0.1
    thres = 0.001

    def run(self):
        dataset = build_dataset(self.val_data_path, self.args, self.speaker_model, self.mean_pose, self.mean_dir_vec)
        dataset.set_lang_model(self.lang_model)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        eval_net_path = "output/TED_Expressive_output/AE-cos1e-3/checkpoint_best.bin"
        evaluator = EmbeddingSpaceEvaluator(self.args, eval_net_path, self.lang_model, self.device)
        return self._evaluate(loader, evaluator)

    def _evaluate(self, test_data_loader, embed_space_evaluator):
        if embed_space_evaluator:
            embed_space_evaluator.reset()
        # (don't delete this comment): losses = AverageMeter('loss')
        joint_mae = AverageMeter("mae_on_joint")
        accel = AverageMeter("accel")
        bc = AverageMeter("bc")
        start = time.time()

        self._evaluate_batches(test_data_loader, embed_space_evaluator,
                               joint_mae, accel, bc)
        elapsed_time = time.time() - start
        return self._collect_metrics(joint_mae, accel, bc,
                                     embed_space_evaluator, elapsed_time)

    def _collect_metrics(self, joint_mae, accel, bc, embed_space_evaluator, elapsed_time):
        """Aggregate metrics into a dictionary and log them if available."""
        ret_dict = {"joint_mae": joint_mae.avg}

        if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
            frechet_dist, feat_dist = embed_space_evaluator.get_scores()
            diversity_score = embed_space_evaluator.get_diversity_scores()

            self._log_metrics(joint_mae, accel, frechet_dist,
                              diversity_score, bc, feat_dist, elapsed_time)

            ret_dict.update({
                "frechet": frechet_dist,
                "feat_dist": feat_dist,
                "diversity_score": diversity_score,
                "bc": bc.avg,
            })
        return ret_dict

    def _log_metrics(self, joint_mae, accel, frechet_dist, diversity_score, bc, feat_dist, elapsed_time):
        """Pretty-print evaluation metrics."""
        logging.info(
            f"[VAL] joint mae: {joint_mae.avg:.5f}, "
            f"accel diff: {accel.avg:.5f}, "
            f"FGD: {frechet_dist:.3f}, "
            f"diversity_score: {diversity_score:.3f}, "
            f"BC: {bc.avg:.3f}, "
            f"feat_D: {feat_dist:.3f}, {elapsed_time:.1f}s"
        )

    def _evaluate_batches(self, test_data_loader, embed_space_evaluator,
                          joint_mae, accel, bc):
        """Iterate over batches and update evaluation metrics."""
        with torch.no_grad():
            for iter_idx, data in enumerate(test_data_loader, 0):
                if iter_idx == 2:  # early stop (debug only?)
                    break
                logging.info(f"testing {iter_idx}/{len(test_data_loader)}")

                in_text, in_text_padded, in_audio, in_spec, target, pre_seq, batch_size = \
                    self._prepare_batch(data)

                out_dir_vec = self._generate_output(pre_seq, in_audio)
                self._compute_beat_consistency(out_dir_vec, in_audio, bc, batch_size)
                self._update_metrics(out_dir_vec, target, joint_mae, accel,
                                     embed_space_evaluator, in_text_padded, in_audio)

    # ---------------- Helper methods ----------------
    def _prepare_batch(self, data):
        in_text, _, in_text_padded, _, target_vec, in_audio, in_spec, _ = data
        batch_size = target_vec.size(0)

        in_text = in_text.to(self.device)
        in_text_padded = in_text_padded.to(self.device)
        in_audio = in_audio.to(self.device)
        in_spec = in_spec.to(self.device)
        target = target_vec.to(self.device)

        pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
        pre_seq[:, 0:self.args.n_pre_poses, :-1] = target[:, 0:self.args.n_pre_poses]
        pre_seq[:, 0:self.args.n_pre_poses, -1] = 1  # constraints

        return in_text, in_text_padded, in_audio, in_spec, target, pre_seq, batch_size

    def _generate_output(self, pre_seq, in_audio):
        assert self.args.model == "pose_diffusion"
        return self.diffusion.sample(self.pose_dim, pre_seq, in_audio)

    def _compute_beat_consistency(self, out_dir_vec, in_audio, bc, batch_size):
        out_dir_vec_bc = out_dir_vec + torch.tensor(self.args.mean_dir_vec).squeeze(1).unsqueeze(0).unsqueeze(0).to(
            self.device)
        left_palm = torch.cross(out_dir_vec_bc[:, :, 11 * 3:12 * 3], out_dir_vec_bc[:, :, 17 * 3:18 * 3], dim=2)
        right_palm = torch.cross(out_dir_vec_bc[:, :, 28 * 3:29 * 3], out_dir_vec_bc[:, :, 34 * 3:35 * 3], dim=2)
        beat_vec = torch.cat((out_dir_vec_bc, left_palm, right_palm), dim=2)
        beat_vec = F.normalize(beat_vec, dim=-1)
        all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)

        # angles
        for idx, pair in enumerate(self.angle_pair):
            vec1 = all_vec[:, pair[0]]
            vec2 = all_vec[:, pair[1]]
            inner_product = torch.einsum("ij,ij->i", [vec1, vec2])
            inner_product = torch.clamp(inner_product, -1, 1)
            angle = torch.acos(inner_product) / math.pi
            angle_time = angle.reshape(batch_size, -1)
            if idx == 0:
                angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / self.change_angle[idx] / len(
                    self.change_angle)
            else:
                angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / self.change_angle[idx] / len(
                    self.change_angle)
        angle_diff = torch.cat((torch.zeros(batch_size, 1).to(self.device), angle_diff), dim=-1)

        # beat consistency vs audio
        for b in range(batch_size):
            motion_beat_time = []
            for t in range(2, 33):
                if angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]:
                    if ((angle_diff[b][t - 1] - angle_diff[b][t] >= self.thres) or
                            (angle_diff[b][t + 1] - angle_diff[b][t] >= self.thres)):
                        motion_beat_time.append(float(t) / 15.0)
            if not motion_beat_time:
                continue

            audio = in_audio[b].cpu().numpy()
            audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units="time")
            score = 0
            for a in audio_beat_time:
                score += np.exp(-np.min(np.square(a - motion_beat_time)) / (2 * self.sigma * self.sigma))
            bc.update(score / len(audio_beat_time), len(audio_beat_time))

    def _update_metrics(self, out_dir_vec, target, joint_mae, accel,
                        embed_space_evaluator, in_text_padded, in_audio):
        if self.args.model != "gesture_autoencoder":
            if embed_space_evaluator:
                embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

            # MAE
            out_dir_vec = out_dir_vec.cpu().numpy()
            out_dir_vec += np.array(self.args.mean_dir_vec).squeeze()
            out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)

            target_vec = target.cpu().numpy()
            target_vec += np.array(self.args.mean_dir_vec).squeeze()
            target_poses = convert_dir_vec_to_pose(target_vec)

            if out_joint_poses.shape[1] == self.args.n_poses:
                diff = out_joint_poses[:, self.args.n_pre_poses:] - target_poses[:, self.args.n_pre_poses:]
            else:
                diff = out_joint_poses - target_poses[:, self.args.n_pre_poses:]
            mae_val = np.mean(np.abs(diff))
            joint_mae.update(mae_val, target.shape[0])

            # acceleration
            target_acc = np.diff(target_poses, n=2, axis=1)
            out_acc = np.diff(out_joint_poses, n=2, axis=1)
            accel.update(np.mean(np.abs(target_acc - out_acc)), target.shape[0])
