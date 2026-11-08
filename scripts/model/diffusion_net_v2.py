# Diffgesture/scripts/model/diffusion_net_v2.py
import logging
import math
import torch
import torch.nn.functional as F
import tqdm
from torch.nn import Module
from .diffusion_util import VarianceSchedule, TransformerModel


class DiffusionNet(Module):

    def __init__(self, net: TransformerModel, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):

        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim),
                          reduction='mean')

        return loss

    @staticmethod
    def _initialize_noise(batch_size, num_pose, pose_dim, device):
        """Initialize Gaussian noise at T."""
        return torch.randn([batch_size, num_pose, pose_dim], device=device)

    def _get_coeffs(self, t):
        """Return diffusion coefficients at step t."""
        alpha = self.var_sched.alphas[t]
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.get_sigmas(t, flexibility=0.0)
        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        return alpha, alpha_bar, sigma, c0, c1

    def _predict_noise(self, x_t, beta, context, uncondition_embedding=None):
        """Predict noise (eps_theta) with optional classifier-free guidance."""
        if uncondition_embedding is not None:
            x_in = torch.cat([x_t] * 2)
            beta_in = torch.cat([beta] * 2)
            uncond_emb = uncondition_embedding.repeat(x_t.shape[0], 1, 1)
            context_in = torch.cat([uncond_emb, context])
            e_theta_uncond, e_theta = self.net(x_in, beta=beta_in, context=context_in).chunk(2)
            return e_theta_uncond + 1.15 * (e_theta - e_theta_uncond)
        return self.net(x_t, beta=beta, context=context)

    def _apply_extra_noise(self, z, x_T, t, num_pose, t0=25):
        """Optionally apply structured noise for small t."""
        if t < t0 and t > 1:
            sigma_a = 1 / (t0 * t0) * (t - t0) * (t - t0)
            z0 = sigma_a * torch.randn_like(x_T[:, 0, :].unsqueeze(1))

            res = torch.zeros_like(z)
            for n in range(num_pose):
                zn = math.sqrt((1 - sigma_a * sigma_a)) * torch.randn_like(z0)
                res[:, n:n + 1, :] = zn + z0
            return res
        return z

    def sample(self, num_pose, context, pose_dim, flexibility=0.0, ret_traj=False, uncondition_embedding=None):
        batch_size = context.size(0)
        # 1. Add noise ONCE
        x_t = self._initialize_noise(batch_size, num_pose, pose_dim, context.device)
        traj = {self.var_sched.num_steps: x_t}
        logging.info(f"Sampling (one-time noise) with {self.var_sched.num_steps} steps.")
        # 2. Deterministic denoising loop
        for t in tqdm.tqdm(range(self.var_sched.num_steps, 0, -1)):
            alpha, alpha_bar, sigma, c0, c1 = self._get_coeffs(t)
            beta = self.var_sched.betas[[t] * batch_size]
            # predict noise
            e_theta = self._predict_noise(x_t, beta, context, uncondition_embedding)

            # (optional structured noise: disable or keep depending on your need)
            # z = self._apply_extra_noise(torch.zeros_like(x_t), x_t, t, num_pose)
            # 3. No fresh Gaussian noise added, only deterministic update
            z = torch.zeros_like(x_t)

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        return traj if ret_traj else traj[0]
