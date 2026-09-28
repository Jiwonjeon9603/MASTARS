import torch

from .arrays import batch_to_device
from .training import Trainer, cycle


class EpisodeBuffer:
    """Accepted generated episodes (s, a, r, done)."""

    def __init__(self):
        self.s, self.a, self.r, self.d = [], [], [], []

    def add(self, s, a, r, d):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)

    def __len__(self):
        return len(self.s)


class Generator(Trainer):
    """MASTARS data augmentation.

    Subgoal-based agent-wise sequential trajectory generation with RePaint
    inpainting, followed by transition-model consistency filtering.
    """

    def __init__(
        self,
        *args,
        generate_batch_size: int = 256,
        adapt_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adapt_threshold = adapt_threshold
        self.buffer = EpisodeBuffer()
        self.generate_dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset, batch_size=generate_batch_size, num_workers=0, shuffle=True, pin_memory=True
            )
        )

    @torch.no_grad()
    def generate_sample(self, batch, returns):
        """Agent-wise sequential trajectory generation (MASTARS).

        1) Value-based subgoal: for each agent, the timestep with the highest
           estimated value; the trajectory prefix up to it is preserved.
        2) Subgoal-based ordering: agents with later subgoals are generated first.
        3) Each agent's suffix is regenerated via RePaint inpainting, conditioned
           on previously generated agents and all preserved prefixes.
        4) Post-processing: actions from the inverse-dynamics model, rewards from
           the reward model, and a transition-model consistency error used for
           acceptance filtering.

        Returns (dynamics_error [B x T-1 x A], obs [B x T x A x D], actions,
                 rewards [B x T-1 x 1], kept_mask [B x T x A x D]).
        """
        model = self.model
        obs = batch["cond"]["x"]
        B, T, A, D = obs.shape

        # standard conditioning: the initial observation of every agent
        cond_x = torch.zeros_like(obs)
        cond_x[:, :1] = obs[:, :1]
        cond = {"x": cond_x, "masks": batch["cond"]["masks"]}

        # (1) value-based subgoal: highest-value timestep per agent
        values = model.value_model(obs).squeeze(-1)  # [B x T x A]
        subgoal_time = values.argmax(dim=1)  # [B x A]
        time_range = torch.arange(T, device=obs.device).view(1, T, 1, 1).expand_as(obs)
        prefix_mask = time_range <= subgoal_time.view(B, 1, A, 1)

        # (2) subgoal-based ordering: later subgoal -> generated earlier
        ordering = torch.argsort(subgoal_time, dim=-1, descending=True)  # [B x A]

        # (3) agent-wise sequential generation via RePaint inpainting
        samples = obs
        generated = torch.zeros(B, A, dtype=torch.bool, device=obs.device)
        kept_mask = torch.ones_like(obs, dtype=torch.bool)
        for i in range(A):
            # True = known region (kept), False = region to generate
            keep = generated.view(B, 1, A, 1).expand_as(obs).clone()
            keep |= prefix_mask
            keep[:, 0] = True
            kept_mask &= keep

            samples = model.repaint_sample(
                cond, {"x": samples, "masks": keep}, returns=returns
            )
            generated.scatter_(1, ordering[:, i : i + 1], True)

        # (4) post-processing: actions, rewards, transition consistency
        pred_act = model.inv_model(torch.cat([samples[:, :-1], samples[:, 1:]], dim=-1))
        if model.discrete_action:
            pred_act = torch.nn.functional.one_hot(
                pred_act.argmax(dim=-1), num_classes=model.num_actions
            )
        obs_act = torch.cat([samples[:, :-1], pred_act], dim=-1)
        pred_reward = model.predict_reward(obs_act)
        pred_next = model.dynamic_model(obs_act)
        dynamics_error = torch.nn.functional.mse_loss(
            samples[:, 1:], pred_next, reduction="none"
        ).mean(dim=-1)

        return dynamics_error, samples, pred_act, pred_reward, kept_mask

    def generate_episodes(self):
        """Generate one batch and keep the episodes whose mean transition-model
        error on the regenerated region is below `adapt_threshold`."""
        batch = batch_to_device(next(self.generate_dataloader), self.device)
        returns = torch.ones_like(batch["returns"])  # condition on the maximum return

        dynamics_error, s, a, r, kept = self.generate_sample(batch, returns)
        d = torch.zeros_like(r)

        regenerated = ~kept[:, 1:, :, 0]  # [B x T-1 x A]
        error = (regenerated.float() * dynamics_error).mean(dim=[1, 2])
        for i in torch.nonzero(error <= self.adapt_threshold).flatten().tolist():
            self.buffer.add(s[i], a[i], r[i], d[i])
        return len(self.buffer)
