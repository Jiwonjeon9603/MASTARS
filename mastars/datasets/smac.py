"""StarCraft Multi-Agent Challenge (SMAC) offline datasets (og-marl, preprocessed).

Expected layout, produced by `scripts/transform_og_marl_dataset.py`:

    <data_dir>/smac/<map>/<quality>/
        obs.npy  actions.npy  rewards.npy  legals.npy  states.npy  path_lengths.npy

Observations fed to the model are the concatenation of each agent's local
observation and the global state.
"""

import os
from typing import Iterator

import numpy as np


def split_dataset_name(name: str):
    """'3m-Good' -> ('3m', 'Good')."""
    idx = name.find("-")
    if idx < 0:
        raise ValueError(f"Expected '<map>-<quality>', got '{name}'")
    return name[:idx], name[idx + 1 :]


def load_episodes(data_dir: str, dataset: str) -> Iterator[dict]:
    """Yield episodes as dicts of arrays shaped (T, n_agents, dim)."""
    map_name, quality = split_dataset_name(dataset)
    dataset_path = os.path.join(data_dir, "smac", map_name, quality)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    def load(name):
        return np.load(os.path.join(dataset_path, f"{name}.npy"))

    observations = load("obs")
    legal_actions = load("legals")
    rewards = load("rewards")
    actions = load("actions")
    path_lengths = load("path_lengths")
    states = load("states")

    n_agents = observations.shape[1]
    states = np.repeat(states[:, None], n_agents, axis=1)
    observations = np.concatenate([observations, states], axis=-1)

    start = 0
    for path_length in path_lengths:
        end = start + path_length
        terminals = np.zeros((path_length, n_agents), dtype=bool)
        terminals[-1] = True
        yield {
            "observations": observations[start:end],
            "legal_actions": legal_actions[start:end],
            "rewards": rewards[start:end],
            "actions": actions[start:end],
            "terminals": terminals,
        }
        start = end
