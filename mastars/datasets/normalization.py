from typing import Dict, List

import numpy as np
import scipy.interpolate as interpolate


class DatasetNormalizer:
    """Per-key normalizer fitted on the valid (non-padded) steps of every agent.

    Parameters are shared across agents, matching the parameter-shared
    diffusion model.
    """

    def __init__(self, fields, keys: List[str], normalizer: str, path_lengths: np.ndarray):
        self.normalizers: Dict[str, Normalizer] = {}
        normalizer_cls = NORMALIZERS[normalizer]
        for key in keys:
            values = np.concatenate(
                [x[:length] for x, length in zip(fields[key], path_lengths)], axis=0
            )
            self.normalizers[key] = normalizer_cls(values.reshape(-1, values.shape[-1]))

    def normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x: np.ndarray, key: str) -> np.ndarray:
        return self.normalizers[key].unnormalize(x)

    def __call__(self, x: np.ndarray, key: str) -> np.ndarray:
        return self.normalize(x, key)

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.normalizers.items())


class Normalizer:
    def __init__(self, X: np.ndarray):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def normalize(self, x):
        raise NotImplementedError

    def unnormalize(self, x):
        raise NotImplementedError


class LimitsNormalizer(Normalizer):
    """Maps [min, max] to [-1, 1]."""

    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins + 1e-8)
        return 2 * x - 1

    def unnormalize(self, x, eps=1e-4):
        x = np.clip(x, -1 - eps, 1 + eps)
        x = (x + 1) / 2.0
        return x * (self.maxs - self.mins) + self.mins

    def __repr__(self):
        return f"[ LimitsNormalizer ] dim: {self.mins.size}"


class CDFNormalizer(Normalizer):
    """Makes each dimension uniform on [-1, 1] via its empirical marginal CDF."""

    def __init__(self, X: np.ndarray):
        super().__init__(X)
        self.dim = X.shape[1]
        self.cdfs = [CDFNormalizer1d(X[:, i]) for i in range(self.dim)]

    def _wrap(self, fn_name: str, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            out[:, i] = getattr(cdf, fn_name)(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self._wrap("normalize", x)

    def unnormalize(self, x):
        return self._wrap("unnormalize", x)

    def __repr__(self):
        return f"[ CDFNormalizer ] dim: {self.dim}"


class CDFNormalizer1d:
    def __init__(self, X: np.ndarray):
        assert X.ndim == 1
        X = X.astype(np.float32)
        self.constant = X.max() == X.min()
        if not self.constant:
            quantiles, cumprob = empirical_cdf(X)
            self.fn = interpolate.interp1d(quantiles, cumprob)
            self.inv = interpolate.interp1d(cumprob, quantiles)
            self.xmin, self.xmax = quantiles.min(), quantiles.max()
            self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def normalize(self, x):
        if self.constant:
            return x
        x = np.clip(x, self.xmin, self.xmax)
        return 2 * self.fn(x) - 1  # [0, 1] -> [-1, 1]

    def unnormalize(self, x):
        if self.constant:
            return x
        x = (x + 1) / 2.0
        x = np.clip(x, self.ymin, self.ymax)
        return self.inv(x)


def empirical_cdf(sample: np.ndarray):
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob


NORMALIZERS = {
    "CDFNormalizer": CDFNormalizer,
    "LimitsNormalizer": LimitsNormalizer,
}
