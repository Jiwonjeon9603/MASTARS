"""Train the MASTARS diffusion model and its auxiliary models on an offline
multi-agent dataset.

    python train.py --config configs/mpe/simple_spread_medium.yaml --gpu 0
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to a YAML config")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--dataset_seed",
        type=str,
        default=None,
        help="MPE only: 'all' or an int selecting data/mpe/<scenario>/<quality>/seed_<k>_data",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="override config data_dir")
    parser.add_argument("--log_dir", type=str, default=None, help="override the default log dir")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return parser.parse_args()


def default_log_dir(config) -> str:
    seed_tag = "all" if config.dataset_seed is None else str(config.dataset_seed)
    return os.path.join(
        "logs", config.env_type, config.dataset, f"dseed_{seed_tag}_h{config.horizon}", f"seed_{config.seed}"
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch

    import mastars.utils as utils
    from mastars.datasets import SequenceDataset
    from mastars.models import GaussianDiffusion, SharedConvAttentionDeconv

    overrides = {"seed": args.seed, "data_dir": args.data_dir}
    if args.dataset_seed is not None:
        overrides["dataset_seed"] = None if args.dataset_seed == "all" else int(args.dataset_seed)
    config = utils.load_config(args.config, overrides)
    config.dataset_seed = config.get("dataset_seed", None)
    log_dir = args.log_dir or default_log_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)

    utils.set_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(log_dir, exist_ok=True)
    config.save(os.path.join(log_dir, "config.yaml"))

    # ------------------------------------------------------------- dataset
    dataset = SequenceDataset(
        env_type=config.env_type,
        dataset=config.dataset,
        data_dir=config.data_dir,
        n_agents=config.n_agents,
        horizon=config.horizon,
        max_path_length=config.max_path_length,
        discrete_action=config.discrete_action,
        normalizer=config.normalizer,
        discount=config.discount,
        returns_scale=config.returns_scale,
        dataset_seed=config.dataset_seed,
    )

    # --------------------------------------------------------------- model
    model = SharedConvAttentionDeconv(
        horizon=config.horizon,
        transition_dim=dataset.observation_dim,
        n_agents=config.n_agents,
        dim=config.dim,
        dim_mults=tuple(config.dim_mults),
        returns_condition=config.returns_condition,
        condition_dropout=config.condition_dropout,
        residual_attn=config.residual_attn,
    )
    diffusion = GaussianDiffusion(
        model,
        n_agents=config.n_agents,
        horizon=config.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        discrete_action=config.discrete_action,
        num_actions=dataset.num_actions,
        n_timesteps=config.n_diffusion_steps,
        hidden_dim=config.hidden_dim,
        loss_discount=config.loss_discount,
        returns_condition=config.returns_condition,
        condition_guidance_w=config.condition_guidance_w,
        value_discount=config.discount,
    ).to(device)
    utils.report_parameters(diffusion)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="mastars", name=f"{config.dataset}-seed{config.seed}", config=config.to_dict())

    trainer = utils.Trainer(
        diffusion,
        dataset,
        log_dir=log_dir,
        device=device,
        train_batch_size=config.batch_size,
        train_lr=config.learning_rate,
        gradient_accumulate_every=config.gradient_accumulate_every,
        ema_decay=config.ema_decay,
        log_freq=config.log_freq,
        save_freq=config.save_freq,
        wandb_run=wandb_run,
    )
    if args.resume:
        trainer.load(args.resume)

    # sanity check: a single forward / backward pass
    batch = utils.batch_to_device(
        torch.utils.data.default_collate([dataset[0]]), device
    )
    loss, _ = diffusion.loss(**batch)
    loss.backward()
    trainer.optimizer.zero_grad()
    print("[ train ] forward / backward check passed")

    # ------------------------------------------------------------ training
    remaining = config.n_train_steps - trainer.step
    print(f"[ train ] Training for {remaining} steps | log_dir: {log_dir}")
    trainer.train(remaining)
    if trainer.step % config.save_freq != 0:
        trainer.save()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
