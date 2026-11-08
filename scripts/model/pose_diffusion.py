# Diffgesture/scripts/model/pose_diffusion.py
import torch
import torch.nn as nn
import logging

from .diffusion_util import *


class PoseDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context

        # add attribute args for sampling
        self.args = args
        pose_dim = args.pose_dim
        diff_hidden_dim = args.diff_hidden_dim
        block_depth = args.block_depth

        self.in_size = 32 + pose_dim + 1
        self.audio_encoder = WavEncoder()

        self.classifier_free = args.classifier_free
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))

        if args.log_name == 'baseline':
            from .diffusion_net import DiffusionNet
            self.diffusion_net = DiffusionNet(
                net=TransformerModel(num_pose=args.n_poses,
                                     pose_dim=pose_dim,
                                     embed_dim=pose_dim + 3 + self.in_size,
                                     hidden_dim=diff_hidden_dim,
                                     depth=block_depth // 2,
                                     decoder_depth=block_depth // 2
                                     ),
                var_sched=VarianceSchedule(
                    num_steps=500,
                    beta_1=1e-4,
                    beta_T=0.02,
                    mode='linear'
                )
            )
        else:
            from .diffusion_net_v2 import DiffusionNet
            self.diffusion_net = DiffusionNet(
                net=TransformerModel(num_pose=args.n_poses,
                                     pose_dim=pose_dim,
                                     embed_dim=pose_dim + 3 + self.in_size,
                                     hidden_dim=diff_hidden_dim,
                                     depth=block_depth // 2,
                                     decoder_depth=block_depth // 2
                                     ),
                var_sched=VarianceSchedule(
                    num_steps=500,
                    beta_1=1e-4,
                    beta_T=0.02,
                    mode='linear'
                )
            )

    def get_loss(self, x, pre_seq, in_audio):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        else:
            raise ValueError(f"Unsupported input_context: {self.input_context}")

        if self.classifier_free:
            mask = torch.zeros((x.shape[0],), device=x.device).float().uniform_(0, 1) < self.null_cond_prob
            in_data = torch.where(mask.unsqueeze(1).unsqueeze(2),
                                  self.null_cond_emb.repeat(in_data.shape[1], 1).unsqueeze(0), in_data)

        neg_elbo = self.diffusion_net.get_loss(x, in_data)

        return neg_elbo

    def sample(self, pose_dim, pre_seq, in_audio):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            # 🔧 Force match along time dimension
            T = pre_seq.shape[1]
            if audio_feat_seq.shape[1] != T:
                logging.warning(f"Audio feature length {audio_feat_seq.shape[1]} != target length {T}, interpolating.")
                audio_feat_seq = torch.nn.functional.interpolate(
                    audio_feat_seq.transpose(1, 2),  # (B, D, Ta)
                    size=T,  # target length
                    mode="linear", align_corners=False
                ).transpose(1, 2)  # back to (B, T, D)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)

        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1], 1).unsqueeze(0)
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim,
                                                uncondition_embedding=uncondition_embedding)
        else:
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim)
        return samples


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)
