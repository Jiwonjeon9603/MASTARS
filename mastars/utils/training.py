import copy
import os
import time

import torch

from .arrays import batch_to_device


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, beta: float):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = ma_params.data * self.beta + (1 - self.beta) * current_params.data


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        log_dir: str,
        device,
        train_batch_size: int = 32,
        train_lr: float = 2e-4,
        gradient_accumulate_every: int = 2,
        ema_decay: float = 0.995,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        log_freq: int = 1000,
        save_freq: int = 20000,
        wandb_run=None,
    ):
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.wandb_run = wandb_run

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.dataset = dataset
        self.device = device
        self.step = 0

        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
                )
            )
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.reset_ema()

    # ------------------------------------------------------------------- ema

    def reset_ema(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_ema()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    # ----------------------------------------------------------------- train

    def train(self, n_train_steps: int):
        start = time.time()
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = batch_to_device(next(self.dataloader), self.device)
                loss, infos = self.model.loss(**batch)
                (loss / self.gradient_accumulate_every).backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                metrics = {k: float(v) for k, v in infos.items()}
                metrics["total_loss"] = float(loss)
                infos_str = " | ".join(f"{k}: {v:8.4f}" for k, v in metrics.items())
                print(f"[ {self.step} ] {infos_str} | t: {time.time() - start:8.2f}s")
                start = time.time()
                if self.wandb_run is not None:
                    self.wandb_run.log(metrics, step=self.step)

            if self.step % self.save_freq == 0:
                self.save()

    # ------------------------------------------------------------ checkpoint

    def save(self) -> str:
        savedir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, f"state_{self.step}.pt")
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema_model.state_dict(),
            },
            savepath,
        )
        print(f"[ utils/training ] Saved model to {savepath}")
        return savepath

    def load(self, path: str):
        """Load the model weights from a checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.step = state["step"]
        self.model.load_state_dict(_remap_legacy_keys(state["model"]))
        self.ema_model.load_state_dict(_remap_legacy_keys(state["ema"]))
        print(f"[ utils/training ] Loaded weights from {path} (step {self.step})")


def _remap_legacy_keys(state_dict: dict) -> dict:
    """Accept checkpoints produced by the research codebase, where the reward
    model was stored as a one-element ensemble."""
    prefix = "reward_model_ensembles.0."
    return {
        (("reward_model." + k[len(prefix) :]) if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
