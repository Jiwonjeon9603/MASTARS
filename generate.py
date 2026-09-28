"""MASTARS data augmentation: generate new episodes with a trained diffusion
checkpoint via subgoal-based agent-wise sequential RePaint sampling, keep the
episodes that pass the transition-model consistency check, and save them in
the same format as the source dataset.

    python generate.py --config configs/mpe/simple_spread_medium.yaml \
        --checkpoint logs/.../checkpoint/state_100000.pt --gpu 0
"""

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to the YAML config used for training")
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint (state_<step>.pt)")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--dataset_seed",
        type=str,
        default=None,
        help="MPE only: 'all' or an int selecting data/mpe/<scenario>/<quality>/seed_<k>_data",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="override config data_dir")
    parser.add_argument("--adapt_threshold", type=float, default=None, help="acceptance threshold on the transition-model error")
    parser.add_argument("--generate_episode_nums", type=int, default=None, help="stop once this many episodes are accepted")
    parser.add_argument("--max_generate_epochs", type=int, default=None, help="maximum number of generation batches")
    parser.add_argument("--save_dir", type=str, default=None, help="output directory (default: generated_data/<dataset>/...)")
    return parser.parse_args()


def default_save_dir(config) -> str:
    seed_tag = "all" if config.dataset_seed is None else str(config.dataset_seed)
    return os.path.join(
        "generated_data", config.env_type, config.dataset, f"dseed_{seed_tag}_thresh_{config.adapt_threshold}"
    )


def save_generated_dataset(config, buffer, save_dir: str):
    """Write accepted episodes in the source dataset's layout."""
    import torch

    os.makedirs(save_dir, exist_ok=True)
    n_agents = config.n_agents

    s = torch.stack(buffer.s)  # [N x T x A x D]
    a = torch.stack(buffer.a)  # [N x T-1 x A x act]
    r = torch.stack(buffer.r)  # [N x T-1 x 1]
    d = torch.stack(buffer.d)

    obs = s[:, :-1].reshape(-1, n_agents, s.shape[-1]).cpu().numpy()
    next_obs = s[:, 1:].reshape(-1, n_agents, s.shape[-1]).cpu().numpy()
    acts = a.reshape(-1, n_agents, a.shape[-1]).cpu().numpy()
    rews = r.reshape(-1).cpu().numpy()
    dones = d.reshape(-1).cpu().numpy()

    if config.env_type == "smac":
        n_episodes, steps = s.shape[0], s.shape[1] - 1
        discounts = np.ones((n_episodes, steps, n_agents), dtype=np.float32)
        discounts[:, -1] = 0.0
        np.save(os.path.join(save_dir, "obs.npy"), obs)
        np.save(os.path.join(save_dir, "actions.npy"), acts)
        np.save(os.path.join(save_dir, "rewards.npy"), np.repeat(rews[:, None], n_agents, axis=-1))
        np.save(os.path.join(save_dir, "discounts.npy"), discounts.reshape(-1, n_agents))
        np.save(os.path.join(save_dir, "path_lengths.npy"), np.full(n_episodes, steps))
    else:
        for i in range(n_agents):
            np.save(os.path.join(save_dir, f"obs_{i}.npy"), obs[:, i])
            np.save(os.path.join(save_dir, f"next_obs_{i}.npy"), next_obs[:, i])
            np.save(os.path.join(save_dir, f"acs_{i}.npy"), acts[:, i])
            np.save(os.path.join(save_dir, f"rews_{i}.npy"), rews)
            np.save(os.path.join(save_dir, f"dones_{i}.npy"), dones)

    print(f"[ generate ] Saved {len(buffer)} generated episodes to {save_dir}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from tqdm import tqdm

    import mastars.utils as utils
    from mastars.datasets import SequenceDataset
    from mastars.models import GaussianDiffusion, SharedConvAttentionDeconv

    overrides = {
        "seed": args.seed,
        "data_dir": args.data_dir,
        "adapt_threshold": args.adapt_threshold,
        "generate_episode_nums": args.generate_episode_nums,
        "max_generate_epochs": args.max_generate_epochs,
    }
    if args.dataset_seed is not None:
        overrides["dataset_seed"] = None if args.dataset_seed == "all" else int(args.dataset_seed)
    config = utils.load_config(args.config, overrides)
    config.dataset_seed = config.get("dataset_seed", None)
    save_dir = args.save_dir or default_save_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)

    utils.set_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    diffusion.eval()

    generator = utils.Generator(
        diffusion,
        dataset,
        log_dir=save_dir,
        device=device,
        generate_batch_size=config.generate_batch_size,
        adapt_threshold=config.adapt_threshold,
    )
    generator.load(args.checkpoint)

    # ---------------------------------------------------------- generation
    target = int(config.generate_episode_nums)
    progress = tqdm(range(config.max_generate_epochs), desc="Generating")
    for epoch in progress:
        n_accepted = generator.generate_episodes()
        progress.set_postfix(accepted=n_accepted)
        if n_accepted >= target:
            break
        if epoch + 1 >= config.bailout_epochs and n_accepted < config.min_episodes:
            print(f"[ generate ] Only {n_accepted} episodes accepted after {epoch + 1} epochs; giving up.")
            break

    if len(generator.buffer) < config.min_episodes:
        print(f"[ generate ] {len(generator.buffer)} episodes accepted (< {config.min_episodes}); nothing saved.")
        return

    config.save(os.path.join(save_dir, "config.yaml"))
    save_generated_dataset(config, generator.buffer, save_dir)


if __name__ == "__main__":
    main()
