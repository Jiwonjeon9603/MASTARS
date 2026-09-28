"""Multi-agent Particle Environment (MPE) offline datasets (OMAR format).

Expected layout (per dataset, one sub-directory per data seed):

    <data_dir>/mpe/<scenario>/<quality>/seed_<k>_data/
        obs_<i>.npy   acs_<i>.npy   rews_<i>.npy   dones_<i>.npy   (i = agent index)

Each file is a flat array of transitions; episodes are recovered by splitting
on `done` or on `max_path_length`.
"""

import collections
import os
from typing import Iterator, Optional

import numpy as np


def split_dataset_name(name: str):
    """'simple_spread-medium' -> ('simple_spread', 'medium')."""
    idx = name.find("-")
    if idx < 0:
        raise ValueError(f"Expected '<scenario>-<quality>', got '{name}'")
    return name[:idx], name[idx + 1 :]


def load_episodes(
    data_dir: str,
    dataset: str,
    n_agents: int,
    max_path_length: int,
    seed: Optional[int] = None,
) -> Iterator[dict]:
    """Yield episodes as dicts of arrays shaped (T, n_agents, dim).

    seed=None loads every `seed_*_data` sub-directory; an int loads only
    `seed_<seed>_data`.
    """
    scenario, quality = split_dataset_name(dataset)
    dataset_path = os.path.join(data_dir, "mpe", scenario, quality)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if seed is None:
        seed_dirs = sorted(
            d for d in os.listdir(dataset_path) if d.startswith("seed_") and d.endswith("_data")
        )
        print(f"[ datasets/mpe ] Using all seed directories: {seed_dirs}")
    else:
        seed_dirs = [f"seed_{seed}_data"]
        print(f"[ datasets/mpe ] Using seed directory: {seed_dirs[0]}")

    for seed_dir in seed_dirs:
        seed_path = os.path.join(dataset_path, seed_dir)
        if not os.path.isdir(seed_path):
            raise FileNotFoundError(f"Seed directory not found: {seed_path}")

        def stack(prefix):
            return np.stack(
                [np.load(os.path.join(seed_path, f"{prefix}_{i}.npy")) for i in range(n_agents)],
                axis=1,
            )

        observations = stack("obs")
        actions = stack("acs")
        rewards = stack("rews")
        dones = stack("dones")

        episode = collections.defaultdict(list)
        for obs, act, rew, done in zip(observations, actions, rewards, dones):
            episode["observations"].append(obs)
            episode["actions"].append(act)
            episode["rewards"].append(rew)
            episode["terminals"].append(done)

            if done.all() or len(episode["observations"]) == max_path_length:
                yield {k: np.asarray(v) for k, v in episode.items()}
                episode = collections.defaultdict(list)
