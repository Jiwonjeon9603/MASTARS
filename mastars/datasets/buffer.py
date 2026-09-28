from typing import Dict, List

import numpy as np


def atleast_nd(x: np.ndarray, n: int) -> np.ndarray:
    while x.ndim < n:
        x = np.expand_dims(x, axis=-1)
    return x


class ReplayBuffer:
    """Stacks variable-length episodes into fixed-size arrays.

    Every field has shape (n_episodes, max_path_length, n_agents, dim).
    Steps beyond an episode's length are filled with that episode's last
    step (rewards are zero-filled instead).
    """

    def __init__(self, episodes: List[Dict[str, np.ndarray]], max_path_length: int):
        assert len(episodes) > 0, "No episodes were loaded"
        self.max_path_length = max_path_length
        self.n_episodes = len(episodes)
        self.keys = list(episodes[0].keys())
        self.path_lengths = np.array([len(ep["observations"]) for ep in episodes], dtype=int)
        assert self.path_lengths.max() <= max_path_length, (
            f"Found an episode of length {self.path_lengths.max()} > "
            f"max_path_length={max_path_length}"
        )

        first = episodes[0]
        self.n_agents = first["observations"].shape[1]
        self._fields: Dict[str, np.ndarray] = {}
        for key in self.keys:
            dim = atleast_nd(first[key], 3).shape[-1]
            self._fields[key] = np.zeros(
                (self.n_episodes, max_path_length, self.n_agents, dim), dtype=np.float32
            )

        for i, episode in enumerate(episodes):
            length = self.path_lengths[i]
            for key in self.keys:
                array = atleast_nd(episode[key], 3)
                if key != "rewards":
                    self._fields[key][i] = array[-1]
                self._fields[key][i, :length] = array

        print(f"[ datasets/buffer ] Loaded {self.n_episodes} episodes\n{self}")

    def __repr__(self) -> str:
        return "\n".join(f"    {k}: {v.shape}" for k, v in self._fields.items())

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __getitem__(self, key: str) -> np.ndarray:
        return self._fields[key]

    def __setitem__(self, key: str, value: np.ndarray):
        self._fields[key] = value

    def items(self):
        return self._fields.items()
