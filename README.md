# Modular PyTorch Training Framework (Config-Driven)

A minimal yet extensible training framework for PyTorch with fully modular components and YAML configuration.
You can mix & match **models**, **data modules**, **losses**, **optimizers**, **schedulers**, **metrics**, and **callbacks** by editing a single config file — no code changes needed.

## Features
- 🔌 **Pluggable modules** via `target: "package.module.ClassName"` + `params: { ... }`
- ⚙️ **Single YAML config** controls the entire training run (incl. optimizer, scheduler, callbacks)
- 🧪 **DataModule** pattern for train/val/test loaders
- 🧮 **Metrics** computed on-the-fly
- 🧯 **Callbacks**: EarlyStopping, ModelCheckpoint, LearningRate logging
- 🧰 **Utilities**: deterministic seed, checkpointing, git commit capture
- 🖥️ **Mixed precision** (AMP) and gradient accumulation supported
- 🔁 **Resume-from-checkpoint** support
- 🗂️ **Outputs**: checkpoints + `metrics.csv` under `training.ckpt_dir`

## Quickstart
```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train with default config
python train.py --config configs/default.yaml

# Optional: override any config key via dotlist (type-safe best effort)
python train.py --config configs/default.yaml training.epochs=20 optimizer.params.lr=5e-4 callbacks.0.params.patience=10
```

## Project Layout
```
.
├── callbacks/
├── configs/
├── data/
├── losses/
├── metrics/
├── models/
├── trainer/
├── utils/
├── train.py
└── requirements.txt
```

## Add your own components
1. Create a new class anywhere (e.g., `models/resnet.py` with `class ResNet(nn.Module): ...`).
2. Point your config to it:
```yaml
model:
  target: "models.resnet.ResNet"
  params: { num_classes: 1000, ... }
```
3. That’s it! No registries necessary.

## Git Version Control
```bash
git init
git add .
git commit -m "init: training framework"
# work normally with branches/tags as you like
```
The framework stores the **current git commit** in each checkpoint (if run inside a git repo).

---

This is a minimal baseline. Extend freely to match your needs (logging to TensorBoard, W&B, more callbacks, etc.).
