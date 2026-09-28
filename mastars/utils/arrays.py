import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    raise TypeError(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch: dict, device) -> dict:
    return {k: to_device(v, device) for k, v in batch.items()}


def _to_str(num: int) -> str:
    if num >= 1e6:
        return f"{num / 1e6:.2f} M"
    return f"{num / 1e3:.2f} k"


def report_parameters(model: torch.nn.Module, topk: int = 10) -> int:
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    sorted_keys = sorted(counts, key=lambda k: -counts[k])
    for key in sorted_keys[:topk]:
        print(" " * 8, f"{key}: {_to_str(counts[key])}")
    remaining = sum(counts[k] for k in sorted_keys[topk:])
    print(
        " " * 8,
        f"... and {max(len(counts) - topk, 0)} others accounting for "
        f"{_to_str(remaining)} parameters",
    )
    return n_parameters
