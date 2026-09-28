<div align="center">

# MASTARS

**Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals**

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-mastars--dataset-FFD21E)](https://huggingface.co/datasets/jwjeonn/mastars-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

</div>

This is the official code for the paper **"MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals"**.

MASTARS is a diffusion-based data-augmentation method for **offline multi-agent RL**. It synthesizes new episodes by:

| | Component | What it does |
|---|---|---|
| 🎯 | **Value-based subgoals** | Each agent's trajectory is split at its highest-value timestep; the prefix is kept, the suffix is regenerated. |
| 🔢 | **Subgoal-based ordering** | Agents with later subgoals are generated first so that later-generated agents can adapt to them. |
| 🎨 | **Sequential RePaint inpainting** | Each agent's suffix is sampled while previously generated agents and all prefixes stay fixed. |
| ✅ | **Transition-model filtering** | Episodes inconsistent with the learned dynamics model are discarded. |

---

## 🛠️ Installation

```bash
conda create -n mastars python=3.8 && conda activate mastars
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
```

> No simulator is needed for training or generation; only `.npy` files are read.

## 📦 Datasets

All datasets used in the paper are on the Hugging Face Hub:
**[jwjeonn/mastars-dataset](https://huggingface.co/datasets/jwjeonn/mastars-dataset)**

```bash
pip install -U huggingface_hub
hf download jwjeonn/mastars-dataset --repo-type dataset --local-dir data
```

| Env | Scenarios | Qualities |
|---|---|---|
| `mpe` | `simple_spread`, `simple_tag`, `simple_world` | `medium`, `medium-replay`, `random` |
| `smac` | `3m`, `2s3z`, `8m` | `Good`, `Medium` |
| `smac` | `25m`, `2c_vs_64zg` | `Medium` |
| `smacv2` | `protoss_{3_vs_3,5_vs_5}`, `terran_{3_vs_3,5_vs_5}`, `zerg_3_vs_3` | `Good`, `Medium` |

Layout after download (`data_dir` in the config or `--data_dir` changes the root):

```
data/mpe/<scenario>/<quality>/seed_2000_data/{obs,acs,rews,dones}_<agent>.npy
data/smac/<map>/<quality>/{obs,actions,rewards,discounts,legals,states,path_lengths}.npy
data/smacv2/<task>/<quality>/...            # same layout as smac
```

Sources: MPE from [OMAR](https://github.com/ling-pan/OMAR) (`seed_2000_data` subset, 2,000 episodes per task), SMAC from [OG-MARL](https://sites.google.com/view/og-marl), SMAC-v2 collected by us with QMIX (paper Appendix G).

<details>
<summary>Convert raw OG-MARL tfrecords yourself (optional, needs StarCraft II)</summary>

```bash
bash scripts/install_sc2.sh
pip install git+https://github.com/oxwhirl/smac.git
pip install -r third_party/og-marl/install_environments/requirements/smacv1.txt
pip install -e third_party/og-marl

# raw tfrecords under data/smac/<map>/<quality>/, then
python scripts/transform_og_marl_dataset.py --map_name 3m --quality Good
```
</details>

## 🚀 Training

```bash
python train.py --config configs/mpe/simple_spread_medium.yaml --gpu 0
python train.py --config configs/smac/3m_good.yaml --gpu 0
```

Checkpoints go to `logs/<env>/<dataset>/dseed_<seed>_h<horizon>/seed_<seed>/checkpoint/state_<step>.pt`.

| Flag | Meaning |
|---|---|
| `--dataset_seed {all,<k>}` | which `seed_<k>_data` subset to load (MPE) |
| `--seed` | training seed |
| `--resume <ckpt>` | continue from a checkpoint |
| `--wandb` | log to Weights & Biases |

## ✨ Generation

```bash
python generate.py --config configs/mpe/simple_spread_medium.yaml \
    --checkpoint logs/mpe/simple_spread-medium/dseed_2000_h8/seed_100/checkpoint/state_100000.pt \
    --gpu 0 --adapt_threshold 0.01 --generate_episode_nums 1000
```

Segments are regenerated with MASTARS and kept only if the mean transition-model error on the regenerated region is at most `adapt_threshold`, until `generate_episode_nums` episodes are accepted. Output is written in the same layout as the source dataset under `generated_data/<env>/<dataset>/dseed_<seed>_thresh_<eps>/`, ready to be mixed with the original data.

| Flag | Meaning |
|---|---|
| `--adapt_threshold` | acceptance threshold ε (Eq. 6 in the paper) |
| `--generate_episode_nums` | number of episodes to accept |
| `--save_dir` | output root |

> Observations and continuous actions are saved in the normalized space (`CDFNormalizer`, `[-1, 1]`). To recover raw values, fit the same normalizer on the source dataset (`SequenceDataset(...).normalizer`) and call `unnormalize`.

## ⚙️ Configuration

Configs are flat YAML files in `configs/mpe/` and `configs/smac/`. Key knobs: `horizon` (segment length), `returns_scale`, `condition_guidance_w` (classifier-free guidance), `n_train_steps`, and the generation settings `generate_batch_size`, `adapt_threshold`, `generate_episode_nums`.

## 🗂️ Repository layout

```
train.py / generate.py   entry points
configs/                 MPE and SMAC hyper-parameters
mastars/datasets/        dataset loaders, replay buffer, normalizers
mastars/models/          multi-agent temporal U-Net, Gaussian diffusion, auxiliary models
mastars/utils/           trainer, MASTARS generator, config helpers
scripts/                 run scripts and SMAC dataset conversion
third_party/og-marl      dataset utilities used by the conversion script
```

## 🙏 Acknowledgements

The diffusion backbone follows [MADiff](https://github.com/zbzhu99/madiff) and [decision-diffuser](https://github.com/anuragajay/decision-diffuser); inpainting follows [RePaint](https://github.com/andreas128/RePaint).

## 📄 License

MIT. See [LICENSE](LICENSE).
