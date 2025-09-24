# DiffGesture/scripts/train_module.py
import torch
import lightning as L
from torch import optim


class DiffGestureModule(L.LightningModule):
    def __init__(self, args, lang_model, speaker_model, pose_dim):
        super().__init__()
        # Save hyperparameters (Lightning will write them into hparams.yaml)
        self.save_hyperparameters(ignore=["lang_model", "speaker_model"])
        # ignore these because they are not serializable
        self.args = args
        self.lang_model = lang_model
        self.speaker_model = speaker_model
        self.pose_dim = pose_dim

        from model.pose_diffusion import PoseDiffusion
        self.model = PoseDiffusion(args)

    def forward(self, pre_seq, in_audio):
        # used for validation/inference
        return self.model.sample(self.pose_dim, pre_seq, in_audio)

    def _train_iter_diffusion(self, args, in_audio, target_poses, pose_diffusion):
        # make pre seq input
        pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
        pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
        pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pose_diffusion.train()
        loss = pose_diffusion.get_loss(target_poses, pre_seq, in_audio)
        return {"mse_loss": loss}

    def training_step(self, batch, batch_idx):
        _, _, _, _, target_vec, in_audio, _, _ = batch
        in_audio, target_vec = in_audio.to(self.device), target_vec.to(self.device)

        loss_dict = self._train_iter_diffusion(self.args, in_audio, target_vec, self.model)

        # log all losses
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict["mse_loss"]  # main optimization target

    def validation_step(self, batch, batch_idx):
        _, _, _, _, target_vec, in_audio, _, _ = batch
        in_audio, target_vec = in_audio.to(self.device), target_vec.to(self.device)

        loss_dict = self._train_iter_diffusion(self.args, in_audio, target_vec, self.model)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True)

        return loss_dict["mse_loss"]

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
