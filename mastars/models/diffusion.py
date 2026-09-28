from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from .helpers import WeightedStateL2, apply_conditioning, mlp


class GaussianDiffusion(nn.Module):
    """Return-conditioned multi-agent trajectory diffusion plus the auxiliary
    models used by MASTARS:

        inv_model     : (o_t, o_{t+1})     -> a_t          (shared across agents)
        dynamic_model : (o_t, a_t)         -> o_{t+1}      (shared across agents)
        reward_model  : (o_t, a_t) of all agents -> team reward
        value_model   : o_t                -> V(o_t)       (shared across agents)

    The diffusion model predicts epsilon over the observation trajectory of all
    agents [batch x horizon x agent x obs_dim]; actions are recovered with the
    inverse-dynamics model.
    """

    def __init__(
        self,
        model: nn.Module,
        n_agents: int,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        discrete_action: bool = False,
        num_actions: int = 0,
        n_timesteps: int = 200,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        returns_condition: bool = True,
        condition_guidance_w: float = 1.2,
        value_discount: float = 0.99,
    ):
        super().__init__()
        assert action_dim > 0
        self.model = model
        self.n_agents = n_agents
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.value_discount = value_discount
        self.n_timesteps = int(n_timesteps)

        # discrete actions enter the dynamics / reward models one-hot encoded
        action_in_dim = num_actions if discrete_action else action_dim
        action_out_dim = num_actions if discrete_action else action_dim

        self.inv_model = mlp(2 * observation_dim, hidden_dim, action_out_dim)
        self.dynamic_model = mlp(observation_dim + action_in_dim, hidden_dim, observation_dim)
        self.reward_model = mlp(n_agents * (observation_dim + action_in_dim), hidden_dim, 1)
        self.value_model = mlp(observation_dim, hidden_dim, 1)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )

        self.loss_fn = WeightedStateL2(self._loss_weights(loss_discount))

    def _loss_weights(self, discount: float) -> torch.Tensor:
        """Per-timestep weights discount**t (mean-normalized), shared over agents / features."""
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        weights = discounts[:, None] * torch.ones(self.observation_dim)
        return weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.discrete_action:
            return F.one_hot(actions.to(torch.long), num_classes=self.num_actions).squeeze(-2)
        return actions

    # ---------------------------------------------------------------- sampling

    def get_model_output(self, x, t, returns=None):
        """Classifier-free guided epsilon prediction."""
        if not self.returns_condition:
            return self.model(x, t)
        epsilon_cond = self.model(x, t, returns=returns, use_dropout=False)
        epsilon_uncond = self.model(x, t, returns=returns, force_dropout=True)
        return epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)

    @torch.no_grad()
    def repaint_sample(
        self,
        cond: Dict[str, torch.Tensor],
        repaint_cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """RePaint-style inpainting.

        cond         : the standard conditioning (initial observation), enforced
                       on the denoised sample at every step
        repaint_cond : {"x": known trajectory, "masks": bool, True = keep}
                       the known region is re-noised to the current step and
                       pasted in, so only masks == False is generated
        """
        batch_size = cond["x"].shape[0]
        shape = (batch_size, self.horizon, self.n_agents, self.observation_dim)
        device = cond["x"].device
        scheduler = self.noise_scheduler

        x = 0.5 * torch.randn(shape, device=device)  # 0.5 for low-temperature sampling
        x_known = repaint_cond["x"]
        keep = repaint_cond["masks"].to(x.dtype)

        for t in scheduler.timesteps:
            x = apply_conditioning(x, cond)
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_output = self.get_model_output(x, ts, returns)
            x = scheduler.step(model_output, t, x).prev_sample

            noise = torch.zeros_like(x_known) if t == 0 else torch.randn_like(x_known)
            x_known_noisy = scheduler.add_noise(x_known, noise, ts)
            x = (1 - keep) * x + keep * x_known_noisy

        return apply_conditioning(x, cond)

    # ---------------------------------------------------------------- training

    def p_losses(self, x_start, cond, t, loss_masks, returns=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = apply_conditioning(x_noisy, cond)

        epsilon = self.model(x_noisy, t, returns=returns)
        loss, info = self.loss_fn(epsilon, noise)

        # mask out the conditioned step, normalizing by the number of kept entries
        loss = ((loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])).mean()
        return loss, info

    def compute_inv_loss(self, x, loss_masks, legal_actions=None):
        obs, act = x[:, :-1, :, self.action_dim :], x[:, :-1, :, : self.action_dim]
        next_obs = x[:, 1:, :, self.action_dim :]
        inp = torch.cat([obs, next_obs], dim=-1).reshape(-1, self.n_agents, 2 * self.observation_dim)
        act = act.reshape(-1, self.n_agents, self.action_dim)
        masks = loss_masks[:, 1:].reshape(-1, self.n_agents)

        pred = self.inv_model(inp)
        info = {}
        if self.discrete_action:
            if legal_actions is not None:
                legal = legal_actions[:, :-1].reshape(-1, *legal_actions.shape[2:])
                pred[legal == 0] = -1e10
            loss = F.cross_entropy(
                pred.reshape(-1, pred.shape[-1]), act.reshape(-1).long(), reduction="none"
            )
            loss = (loss * masks.reshape(-1)).mean() / masks.mean()
            acc = (pred.argmax(dim=-1, keepdim=True) == act).float().squeeze(-1)
            info["inv_acc"] = (acc * masks).mean() / masks.mean()
        else:
            loss = (F.mse_loss(pred, act, reduction="none") * masks.unsqueeze(-1)).mean()
            loss = loss / masks.mean()
        return loss, info

    def compute_dynamic_loss(self, x, loss_masks):
        obs, act = x[:, :-1, :, self.action_dim :], x[:, :-1, :, : self.action_dim]
        next_obs = x[:, 1:, :, self.action_dim :].reshape(-1, self.n_agents, self.observation_dim)
        inp = torch.cat([obs, self._encode_actions(act)], dim=-1)
        inp = inp.reshape(-1, self.n_agents, inp.shape[-1])
        masks = loss_masks[:, 1:].reshape(-1, self.n_agents)

        pred = self.dynamic_model(inp)
        loss = (F.mse_loss(pred, next_obs, reduction="none") * masks.unsqueeze(-1)).mean()
        return loss / masks.mean()

    def predict_reward(self, obs_act: torch.Tensor) -> torch.Tensor:
        """obs_act: [batch x T x agent x (obs_dim + action_in_dim)] -> [batch x T x 1]"""
        B, T = obs_act.shape[:2]
        return self.reward_model(obs_act.reshape(B * T, -1)).reshape(B, T, 1)

    def compute_reward_loss(self, x, rewards):
        obs, act = x[..., self.action_dim :], x[..., : self.action_dim]
        pred = self.predict_reward(torch.cat([obs, self._encode_actions(act)], dim=-1))
        team_reward = rewards.mean(dim=-2)  # [batch x T x 1]
        return F.mse_loss(pred, team_reward)

    def compute_value_loss(self, x, rewards, loss_masks):
        obs = x[:, :-1, :, self.action_dim :].reshape(-1, self.n_agents, self.observation_dim)
        next_obs = x[:, 1:, :, self.action_dim :].reshape(-1, self.n_agents, self.observation_dim)
        rewards = rewards[:, :-1].reshape(-1, self.n_agents, 1)
        masks = loss_masks[:, 1:].reshape(-1, self.n_agents)

        value = self.value_model(obs)
        target = rewards + self.value_discount * self.value_model(next_obs)
        loss = (F.mse_loss(value, target, reduction="none") * masks.unsqueeze(-1)).mean()
        return loss / masks.mean()

    def loss(self, x, cond, loss_masks, rewards, returns=None, legal_actions=None):
        """x: [batch x horizon x agent x (action_dim + obs_dim)]"""
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        diffusion_loss, info = self.p_losses(
            x[..., self.action_dim :], cond, t, loss_masks, returns
        )
        inv_loss, inv_info = self.compute_inv_loss(x, loss_masks, legal_actions)
        dynamic_loss = self.compute_dynamic_loss(x, loss_masks)
        reward_loss = self.compute_reward_loss(x, rewards)
        value_loss = self.compute_value_loss(x, rewards, loss_masks)

        info.update(inv_info)
        info.update(
            inv_loss=inv_loss,
            dynamic_loss=dynamic_loss,
            reward_loss=reward_loss,
            value_loss=value_loss,
        )
        loss = (diffusion_loss + inv_loss + dynamic_loss + reward_loss + value_loss) / 5
        return loss, info
