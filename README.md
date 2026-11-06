# BehaviorBench

This is the companion code for the benchmarking study reported in the paper "Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving", accepted at NeurIPS2026. The paper can be found at https://arxiv.org/pdf/2605.10034. The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.

This software is a research prototype, solely developed for and published as part of the publication above.

The companion code is a fork of [PufferDrive](https://github.com/Emerge-Lab/PufferDrive/). Below are instructions in setting up PufferDrive as well as the BehaviorBench additions.

# PufferDrive

<img align="left" style="width:260px" src="https://github.com/Emerge-Lab/PufferDrive/blob/main/pufferlib/resources/drive/pufferdrive_20fps_long.gif" width="288px">

**PufferDrive is a fast and friendly driving simulator to train and test RL-based models.**

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

---

**Docs**: https://emerge-lab.github.io/PufferDrive

---

### See our 2.0 release video

<a href="https://www.youtube.com/watch?v=LfQ324R-cbE">
  <img src="https://img.youtube.com/vi/LfQ324R-cbE/0.jpg" alt="PufferDrive 2.0" width="300">
</a>

## Installation

Clone the repo
```bash
https://github.com/Emerge-Lab/PufferDrive.git
```

Make a venv (`uv venv`), activate the venv
```
source .venv/bin/activate
```

Inside the venv, install the dependencies
```
uv pip install -e .
```

Compile the C code
```
python setup.py build_ext --inplace --force
```
Run this while your virtual environment is active so the extension is built against the right interpreter.

To test your setup, you can run
```
puffer train puffer_drive
```
See also the [puffer docs](https://puffer.ai/docs.html).


## Quick start

Start a training run
```
puffer train puffer_drive
```

## Documentation

| Document | Description |
|----------|-------------|
| [EVALUATION.md](EVALUATION.md) | Planner evaluation framework (eval.py) - ego planners vs traffic controllers |
| [REALISM_EVALUATION.md](REALISM_EVALUATION.md) | WOSAC realism evaluation (eval_realism.py) - distributional realism metrics |
| [BENCHMARK.md](BENCHMARK.md) | Interactive benchmark extraction from Waymo data |
| [TRAINING.md](TRAINING.md) | PufferRL PPO training configuration and usage |
| [PREDICTION.md](PREDICTION.md) | SMART prediction model - architecture, training, evaluation |
| [NUPLAN.md](NUPLAN.md) | nuPlan integration - evaluate PufferDrive planners on nuPlan |

---

## Planner Evaluation

PufferDrive includes a config-driven evaluation framework for testing different planning algorithms as ego planners and traffic controllers. See [EVALUATION.md](EVALUATION.md) for comprehensive documentation.

### Pre-trained Weights

The repository includes pre-trained weights in `weights/`:

| File | Description |
|------|-------------|
| `weights/simple_ppo.pt` | PPO policy trained with self-play and simple reward on WOMD |
| `weights/conditioned_ppo.pt` | Conditioned PPO policy trained with self-play on WOMD |
| `weights/smart_epoch_030.pt` | SMART prediction model (1M params, 30 epochs) |

### Benchmark Splits

We provide two curated evaluation splits:

| Split | Name | Description |
|-------|------|-------------|
| `interactive1k` | **Interactive1k** | 1,000 most interactive WOMD validation scenarios |
| `random1k` | **Random1k** | 1,000 random WOMD validation scenarios |

See [BENCHMARK.md](BENCHMARK.md) for data preprocessing, download instructions and benchmark details.

### Available Planners

| Planner | Description |
|---------|-------------|
| `pdm` | Predictive Driver Model. Proposes trajectory candidates via IDM with different velocities/offsets, selects best. |
| `ppo` | Pre-trained RL policy with LSTM. Requires checkpoint weights. |
| `smart` | SMART autoregressive trajectory prediction. Requires trained weights. |
| `idm` | Intelligent Driver Model. Rule-based lane-following. |
| `hybrid` | PPO + PDM |
| `conditioned_aggr` / `conditioned_normal` / `conditioned_caut` | Reward-conditioned PPO variants. |
| `constant_velocity` | Baseline that maintains current velocity. |

### Available Traffic Agents

| Traffic Agent | Description |
|---------------|-------------|
| `idm` | Intelligent Driver Model. Default traffic controller, rule-based lane-following. |
| `pdm` | Predictive Driver Model used as traffic. |
| `ppo` | Pre-trained RL policy used for traffic agents. Requires checkpoint weights. |
| `smart` | SMART autoregressive trajectory prediction for traffic. Requires trained weights. |
| `conditioned_mix` / `conditioned_aggr` / `conditioned_normal` / `conditioned_caut` | Reward-conditioned PPO traffic variants. |
| `expert` | Ground truth trajectory replay from the Waymo dataset (traffic only). |
| `constant_velocity` | Baseline that maintains current velocity. |

### Quick Start

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries

# PDM ego vs IDM traffic (default)
python pufferlib/ocean/benchmark/eval.py --map-ids 0-10

# SMART ego planner (using provided weights)
python pufferlib/ocean/benchmark/eval.py --planner.type smart \
    --planner.smart.weights-path weights/smart_1M_epoch_029.pt --map-ids 0-10

# PPO ego vs SMART traffic (using provided weights)
python pufferlib/ocean/benchmark/eval.py --planner.type ppo \
    --planner.ppo.weights-path weights/ppo_self_play.pt \
    --traffic.type smart --traffic.smart.weights-path weights/smart_1M_epoch_029.pt

# Enable visualization
python pufferlib/ocean/benchmark/eval.py --eval.viz True --map-ids 5
```

---

## RL Training with PufferRL

PufferDrive uses [PufferLib](https://puffer.ai/) for RL training with PPO (Proximal Policy Optimization). Training is configured via INI config files and command-line overrides.

### Basic Training

```bash
# Start training with default config
puffer train puffer_drive

# Train with custom parameters
puffer train puffer_drive --train.learning-rate 0.001 --train.batch-size 262144

# Resume from checkpoint
puffer train puffer_drive --load-model-path experiments/puffer_drive/checkpoints/model_001000.pt

# Evaluate a trained model
puffer eval puffer_drive --load-model-path experiments/puffer_drive/checkpoints/model_001000.pt
```

### Training Workflow

1. **Environment Creation**: Multiple vectorized Drive environments are created across `num_workers` processes, each managing `num_envs` environments
2. **Experience Collection**: The policy collects rollouts of length `bptt_horizon` across all environments simultaneously, filling a batch of size `batch_size`
3. **Policy Update**: The batch is split into `minibatch_size` chunks and PPO gradient updates are applied
4. **Repeat**: Steps 2-3 repeat until `total_timesteps` is reached
5. **Checkpointing**: Model weights are saved every `checkpoint_interval` updates to `experiments/puffer_drive/checkpoints/`

### Configuration Files

The training configuration is defined in two INI files:
- **`pufferlib/config/default.ini`**: Base defaults for all environments
- **`pufferlib/config/ocean/drive.ini`**: Drive-specific overrides

Any parameter can be overridden via command-line: `--section.parameter value`

### Environment Parameters (`[env]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_agents` | 1024 | Total agents managed per environment instance (across all loaded maps) |
| `action_type` | `continuous` | Action space type: `discrete` (7x13=91 actions) or `continuous` |
| `dynamics_model` | `classic` | Vehicle dynamics: `classic` (acceleration + steering) or `jerk` (jerk-based, 4x3=12 actions) |
| `dt` | 0.1 | Simulation timestep in seconds (10 Hz, matching WOMD) |
| `episode_length` | 91 | Steps per episode (91 steps = 9.1 seconds, full WOMD scenario) |
| `num_maps` | 80000 | Number of map binaries to load |
| `split` | `training` | Dataset split: `training`, `validation`, `testing`, or custom path |
| `init_steps` | 0 | Initial trajectory steps to skip (0 = start from beginning) |
| `control_mode` | `control_vehicles` | Which agents to control: `control_vehicles`, `control_agents`, `control_wosac`, `control_sdc_only` |
| `init_mode` | `create_all_valid` | Agent creation: `create_all_valid` (all agents) or `create_only_controlled` |
| `resample_frequency` | 910 | How often to resample new scenarios (in steps) |
| `termination_mode` | 1 | 0 = terminate at `episode_length`, 1 = terminate after all agents are done |

**Reward Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_vehicle_collision` | -0.5 | Penalty for colliding with another vehicle |
| `reward_offroad_collision` | -0.5 | Penalty for going off-road |
| `reward_goal` | 1.0 | Reward for reaching the goal |
| `reward_goal_post_respawn` | 0.25 | Reward for reaching goals after first respawn |

**Goal Behavior:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal_behavior` | 3 | What happens when an agent reaches its goal: 0=respawn at start, 1=generate new goal, 2=stop, 3=remove agent |
| `goal_radius` | 2.0 | Distance threshold (meters) to consider goal reached |
| `goal_speed` | 100.0 | Maximum target speed towards goal (m/s) |
| `goal_target_distance` | 30.0 | Distance for newly generated goals (when `goal_behavior=1`) |

**Collision/Offroad Handling:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `collision_behavior` | 2 | On collision: 0=ignore, 1=stop agent, 2=remove agent |
| `offroad_behavior` | 2 | On offroad: 0=ignore, 1=stop agent, 2=remove agent |

### PPO Training Parameters (`[train]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 2,000,000,000 | Total training steps (2 billion) |
| `learning_rate` | 0.003 | Initial learning rate |
| `anneal_lr` | True | Linearly anneal learning rate to 0 |
| `batch_size` | 524288 | Total samples per training epoch (= `num_agents * num_workers * bptt_horizon`) |
| `minibatch_size` | 32768 | Mini-batch size for gradient updates |
| `bptt_horizon` | 32 | Backpropagation-through-time horizon (sequence length per update) |
| `gamma` | 0.98 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_coef` | 0.2 | PPO clipping coefficient |
| `vf_coef` | 2.0 | Value function loss weight |
| `vf_clip_coef` | 0.2 | Value function clipping coefficient |
| `ent_coef` | 0.005 | Entropy bonus coefficient (encourages exploration) |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `update_epochs` | 1 | Number of PPO epochs per batch |
| `checkpoint_interval` | 1000 | Save model every N updates |

**Optimizer Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | `muon` | Optimizer: `adam` or `muon` |
| `adam_beta1` | 0.9 | Adam beta1 (momentum) |
| `adam_beta2` | 0.999 | Adam beta2 (RMSProp-like) |
| `adam_eps` | 1e-8 | Adam epsilon for numerical stability |

**Advanced Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `prio_alpha` | 0.85 | Priority experience sampling alpha |
| `prio_beta0` | 0.85 | Priority experience sampling beta |
| `vtrace_rho_clip` | 1.0 | V-trace rho clipping (importance sampling) |
| `vtrace_c_clip` | 1.0 | V-trace c clipping |

### Vectorization Parameters (`[vec]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_workers` | 16 | Number of parallel worker processes |
| `num_envs` | 16 | Number of environments per worker |
| `batch_size` | 4 | Environments per batch in vectorized stepping |

### Policy Architecture (`[policy]`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 64 | First hidden layer size |
| `hidden_size` | 256 | Main hidden layer size |

### Observation Space

The observation is a flat vector with three components:

1. **Ego features** (7 for classic, 10 for jerk dynamics):
   - Position relative to goal, speed, heading, steering, acceleration

2. **Partner features** (7 per agent, max 31 agents = 217):
   - Relative position, speed, heading, distance to each nearby agent

3. **Road features** (7 per segment, max 128 segments = 896):
   - Relative position, orientation, type for nearest road segments

Total observation size: 7 + 217 + 896 = **1120** (classic dynamics)

### Hyperparameter Sweeps

```bash
# Run a sweep over learning rate, entropy, and gamma
puffer sweep puffer_drive
```

Sweep ranges are defined in `drive.ini` under `[sweep.*]` sections.

### Evaluation Modes

```bash
# Standard evaluation
puffer eval puffer_drive --load-model-path model.pt

# WOSAC realism evaluation (distributional metrics)
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path model.pt

# Human replay evaluation (SDC only, others follow logs)
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path model.pt
```

---

## Dataset

<details>
<summary>Downloading and using data</summary>

### Downloading Waymo Data

You can download the WOMD data from Hugging Face in two versions:

- **Mini dataset**: [GPUDrive_mini](https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini) contains 1,000 training files and 300 test/validation files
- **Medium dataset**: [10,000 files from the training dataset](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_train)
- **Large dataset**: [GPUDrive](https://huggingface.co/datasets/EMERGE-lab/GPUDrive) contains 100,000 unique scenes

**Note**: Replace 'GPUDrive_mini' with 'GPUDrive' in your download commands if you want to use the full dataset.

### Additional Data Sources

For more training data compatible with PufferDrive, see [ScenarioMax](https://github.com/valeoai/ScenarioMax). The GPUDrive data format is fully compatible with PufferDrive.
</details>


## Visualizer

<details>
<summary>Dependencies and usage</summary>

## Local rendering

To launch an interactive renderer, first build:
```
bash scripts/build_ocean.sh drive local
```

then launch:
```bash
./drive
```
this will run `demo()` with an existing model checkpoint.

## Headless server setup

Run the Raylib visualizer on a headless server and export as .mp4. This will rollout the pre-trained policy in the env.

### Install dependencies

```bash
sudo apt update
sudo apt install ffmpeg xvfb
```

For HPC (There are no root privileges), so install into the conda environment
```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64
conda install -c conda-forge ffmpeg
```

- `ffmpeg`: Video processing and conversion
- `xvfb`: Virtual display for headless environments

### Build and run

1. Build the application:
```bash
bash scripts/build_ocean.sh visualize local
```

2. Run with virtual display:
```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize
```

The `-s` flag sets up a virtual screen at 1280x720 resolution with 24-bit color depth.

---

> To force a rebuild, you can delete the cached compiled executable binary using `rm ./visualize`.

---

</details>


## Benchmarks

### Distributional realism

We provide a PufferDrive implementation of the [Waymo Open Sim Agents Challenge (WOSAC)](https://waymo.com/open/challenges/2025/sim-agents/) for fast, easy evaluation of how well your trained agent matches distributional properties of human behavior. See documentation [here](https://emerge-lab.github.io/PufferDrive/wosac/).

WOSAC evaluation with random policy:
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

WOSAC evaluation with your checkpoint (must be .pt file):
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <your-trained-policy>.pt
```

### Human-compatibility

You may be interested in how compatible your agent is with human partners. For this purpose, we support an eval where your policy only controls the self-driving car (SDC). The rest of the agents in the scene are stepped using the logs. While it is not a perfect eval since the human partners here are static, it will still give you a sense of how closely aligned your agent's behavior is to how people drive. You can run it like this:
```bash
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <your-trained-policy>.pt
```

## Development

<details><summary>Documentation and browser demo</summary>

**Docs**

A browsable documentation site now lives under `docs/` and is configured with mkbooks. To preview locally:
```
brew install mdbook
mdbook serve --open docs
```
Open the served URL to see a local version of the docs.

**Interactive demo**

To edit the browser demo, follow these steps:
- Download [emscripten](https://github.com/emscripten-core/emscripten)
- emscripten install latest
- Activate: `source emsdk/emsdk_env.sh`
- Run `bash scripts/build_ocean.sh drive web`
- This generates a number of `game*` files, move them to `assets/` to include them on the webpage

</details>


## Citation

If you use PufferDrive in your research, please cite:
```bibtex
@software{pufferdrive2025github,
  author = {Daphne Cornelisse* and Spencer Cheng* and Pragnay Mandavilli and Julian Hunt and Kevin Joseph and Waël Doulazmi and Aditya Gupta and Eugene Vinitsky},
  title = {{PufferDrive}: A Fast and Friendly Driving Simulator for Training and Evaluating {RL} Agents},
  url = {https://github.com/Emerge-Lab/PufferDrive},
  version = {2.0.0},
  year = {2025},
}
```
