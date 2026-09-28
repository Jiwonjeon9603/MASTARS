# MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This is the official code for the paper **"MASTARS: Multi-Agent Sequential
Trajectory Augmentation with Return-Conditioned Subgoals"**. MASTARS is a
data-augmentation method for offline multi-agent reinforcement learning:
a return-conditioned multi-agent diffusion
model is trained on the offline dataset together with small inverse-dynamics,
transition, reward and value models. New episodes are then synthesized by

1. **Value-based subgoals** – for every agent, the timestep of highest estimated
   value splits its trajectory into a preserved prefix and a suffix to regenerate;
2. **Subgoal-based agent ordering** – agents whose subgoal comes later are
   regenerated first, so that earlier agents can react to them;
3. **Agent-wise sequential RePaint inpainting** – each agent's suffix is sampled
   from the diffusion model while all previously generated agents and every
   preserved prefix are kept fixed;
4. **Transition-model filtering** – episodes whose regenerated region is not
   consistent with the learned dynamics model are discarded.

The repository contains exactly two entry points: `train.py` (diffusion model +
auxiliary models) and `generate.py` (data augmentation).

## Installation

```bash
conda create -n mastars python=3.8
conda activate mastars
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
```

Training and generation only read `.npy` files; no simulator has to be
installed for them. StarCraft II is needed only if you convert the raw OG-MARL
SMAC datasets yourself (see below).

## Datasets

All datasets used in the paper are hosted on the Hugging Face Hub at
[jwjeonn/mastars-dataset](https://huggingface.co/datasets/jwjeonn/mastars-dataset).
Download them into `data/` (configurable with `data_dir` in the config or
`--data_dir` on the command line):

```bash
pip install -U huggingface_hub
hf download jwjeonn/mastars-dataset --repo-type dataset --local-dir data
```

This gives the following layout, which the loaders read directly:

```
data/mpe/<scenario>/<quality>/seed_2000_data/{obs,acs,rews,dones}_<agent>.npy
data/smac/<map>/<quality>/{obs,actions,rewards,discounts,legals,states,path_lengths}.npy
data/smacv2/<task>/<quality>/...                                      (same as smac)
```

| Env | Scenarios | Qualities |
|---|---|---|
| `mpe` | `simple_spread`, `simple_tag`, `simple_world` | `medium`, `medium-replay`, `random` |
| `smac` | `3m`, `2s3z`, `8m` | `Good`, `Medium` |
| `smac` | `25m`, `2c_vs_64zg` | `Medium` |
| `smacv2` | `protoss_3_vs_3`, `protoss_5_vs_5`, `terran_3_vs_3`, `terran_5_vs_5`, `zerg_3_vs_3` | `Good`, `Medium` |

MPE data comes from [OMAR](https://github.com/ling-pan/OMAR) and contains only
the `seed_2000_data` subset (2,000 episodes per task), matching the small-data
setting studied in the paper. SMAC data comes from
[Off-the-Grid MARL](https://sites.google.com/view/og-marl); SMAC-v2 data was
collected by us with QMIX (see the paper's Appendix G).

### Converting raw OG-MARL datasets (optional)

If you want to build the SMAC `.npy` files yourself from the raw OG-MARL
tfrecords instead of downloading them:

```bash
bash scripts/install_sc2.sh                                   # StarCraft II + SMAC maps
pip install git+https://github.com/oxwhirl/smac.git
pip install -r third_party/og-marl/install_environments/requirements/smacv1.txt
pip install -e third_party/og-marl

# raw tfrecords under data/smac/<map>/<quality>/ , then
python scripts/transform_og_marl_dataset.py --map_name 3m --quality Good
```

## Training

```bash
python train.py --config configs/mpe/simple_spread_medium.yaml --gpu 0
python train.py --config configs/smac/3m_good.yaml --gpu 0
```

Checkpoints and a copy of the config are written to
`logs/<env>/<dataset>/dseed_<seed>_h<horizon>/seed_<seed>/checkpoint/state_<step>.pt`.
Useful flags: `--dataset_seed {all,<k>}`, `--seed`, `--resume <ckpt>`,
`--log_dir`, `--wandb`.

## Generation (data augmentation)

```bash
python generate.py --config configs/mpe/simple_spread_medium.yaml \
    --checkpoint logs/mpe/simple_spread-medium/dseed_2000_h8/seed_100/checkpoint/state_100000.pt \
    --gpu 0 --adapt_threshold 0.01 --generate_episode_nums 1000
```

Generation loops over batches of dataset segments, regenerates them with
MASTARS and keeps the episodes whose mean transition-model error on the
regenerated region is at most `adapt_threshold`, until
`generate_episode_nums` episodes have been accepted. The accepted episodes are
saved in the same layout as the source dataset (MPE: per-agent
`obs/next_obs/acs/rews/dones` files; SMAC: `obs/actions/rewards/discounts/path_lengths`)
under `generated_data/<env>/<dataset>/dseed_<seed>_thresh_<eps>/`, ready to be
mixed with the original data for offline MARL training.

Useful flags: `--adapt_threshold`, `--generate_episode_nums`, `--save_dir`,
`--weights {model,ema}`, `--no_subgoal` (regenerate whole trajectories).

Observations and continuous actions are stored in the model's normalized
space (`CDFNormalizer`, range `[-1, 1]`). To recover physical values, fit the
same normalizer on the source dataset (`SequenceDataset(...).normalizer`) and
call `unnormalize`.

## Configuration

Configs are flat YAML files (`configs/mpe/*.yaml`, `configs/smac/*.yaml`)
grouped into dataset, model, training and generation sections. The most
relevant knobs are `horizon` (segment length), `returns_scale` (return
normalization), `condition_guidance_w` (classifier-free guidance weight),
`n_train_steps`, and the generation settings `generate_batch_size`,
`adapt_threshold` and `generate_episode_nums`.

## Repository layout

```
train.py / generate.py        entry points
configs/                      MPE and SMAC hyper-parameters
mastars/datasets/             dataset loaders, replay buffer, normalizers
mastars/models/               multi-agent temporal U-Net and Gaussian diffusion (+ auxiliary models)
mastars/utils/                trainer, MASTARS generator, config helpers
scripts/                      run scripts and SMAC dataset conversion
third_party/og-marl           dataset utilities used by the SMAC conversion script
```

## Acknowledgements

The diffusion backbone follows [MADiff](https://github.com/zbzhu99/madiff),
which in turn builds on
[decision-diffuser](https://github.com/anuragajay/decision-diffuser).
Inpainting follows [RePaint](https://github.com/andreas128/RePaint).

## License

MIT. See [LICENSE](LICENSE).
