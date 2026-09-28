from typing import List, Optional

import numpy as np
import torch

from mastars.datasets import mpe, smac
from mastars.datasets.buffer import ReplayBuffer
from mastars.datasets.normalization import DatasetNormalizer


class SequenceDataset(torch.utils.data.Dataset):
    """Fixed-horizon multi-agent trajectory segments for diffusion training.

    Each item is a dict with
        x            : (H, A, action_dim + obs_dim)  actions ++ normalized observations
        cond         : {"x": (H, A, obs_dim), "masks": (H, A, obs_dim) bool}
                       the first observation of every agent is the known region
        loss_masks   : (H, A, 1)   1 everywhere except the conditioned first step
        rewards      : (H, A, 1)
        returns      : (1, A)      discounted return-to-go of the segment start
        legal_actions: (H, A, num_actions)  (SMAC only)
    """

    def __init__(
        self,
        env_type: str,
        dataset: str,
        data_dir: str,
        n_agents: int,
        horizon: int,
        max_path_length: int,
        discrete_action: bool = False,
        normalizer: str = "CDFNormalizer",
        discount: float = 0.99,
        returns_scale: float = 1.0,
        dataset_seed: Optional[int] = None,
    ):
        assert env_type in ("mpe", "smac"), env_type
        self.env_type = env_type
        self.n_agents = n_agents
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discrete_action = discrete_action
        self.discount = discount
        self.returns_scale = returns_scale
        self.discounts = discount ** np.arange(max_path_length)[:, None, None]

        if env_type == "mpe":
            episodes = list(
                mpe.load_episodes(data_dir, dataset, n_agents, max_path_length, seed=dataset_seed)
            )
        else:
            episodes = list(smac.load_episodes(data_dir, dataset))

        self.fields = ReplayBuffer(episodes, max_path_length)
        assert self.fields.n_agents == n_agents, (
            f"Config says n_agents={n_agents} but the dataset has {self.fields.n_agents}"
        )
        self.path_lengths = self.fields.path_lengths
        self.n_episodes = self.fields.n_episodes

        self.observation_dim = self.fields["observations"].shape[-1]
        self.action_dim = self.fields["actions"].shape[-1]
        self.num_actions = (
            self.fields["legal_actions"].shape[-1] if "legal_actions" in self.fields else 0
        )

        # discrete actions (SMAC) are kept as indices; only observations are normalized
        norm_keys = ["observations"] if discrete_action else ["observations", "actions"]
        self.normalizer = DatasetNormalizer(self.fields, norm_keys, normalizer, self.path_lengths)
        for key in norm_keys:
            self.fields[f"normed_{key}"] = self.normalizer(self.fields[key], key)

        self.indices = self._make_indices()
        self._pad_future()

    # ------------------------------------------------------------------ setup

    def _make_indices(self) -> np.ndarray:
        """(episode, start) pairs; segments may run past the episode end."""
        indices = [
            (i, start)
            for i, path_length in enumerate(self.path_lengths)
            for start in range(path_length - 1)
        ]
        return np.array(indices)

    def _pad_future(self):
        """Append horizon-1 copies of the last step so every segment is full length."""
        keys = ["normed_observations", "rewards", "terminals"]
        keys.append("actions" if self.discrete_action else "normed_actions")
        if "legal_actions" in self.fields:
            keys.append("legal_actions")
        for key in keys:
            values = self.fields[key]
            pad = np.repeat(values[:, -1:], self.horizon - 1, axis=1)
            self.fields[key] = np.concatenate([values, pad], axis=1)

    # ---------------------------------------------------------------- dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        path_ind, start = self.indices[idx]
        end = start + self.horizon

        observations = self.fields["normed_observations"][path_ind, start:end]
        if self.discrete_action:
            actions = self.fields["actions"][path_ind, start:end]
        else:
            actions = self.fields["normed_actions"][path_ind, start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)

        cond_masks = np.zeros_like(observations, dtype=bool)
        cond_masks[0] = True
        cond = {"x": observations.copy(), "masks": cond_masks}

        loss_masks = np.ones((self.horizon, self.n_agents, 1), dtype=np.float32)
        loss_masks[0] = 0.0

        rewards = self.fields["rewards"][path_ind, start : self.max_path_length]
        returns = (self.discounts[: len(rewards)] * rewards).sum(axis=0).squeeze(-1)
        returns = np.array([returns / self.returns_scale], dtype=np.float32)

        batch = {
            "x": trajectories,
            "cond": cond,
            "loss_masks": loss_masks,
            "rewards": self.fields["rewards"][path_ind, start:end],
            "returns": returns,
        }
        if "legal_actions" in self.fields:
            batch["legal_actions"] = self.fields["legal_actions"][path_ind, start:end]
        return batch
