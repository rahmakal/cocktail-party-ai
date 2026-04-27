# Cocktail Party AI

Conv-TasNet project for separating mixed speech into individual speaker sources.

## Structure

```text
.
├── main.py              # CLI entry point
├── configs/             # training and data configs
├── src/                 # project code
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── generate_mixtures.py
│   ├── train.py
│   ├── evaluate.py
│   ├── separate.py
│   └── mixture.py       # legacy Arabic mixture generator
└── requirements.txt
```

Generated data, checkpoints, logs, and outputs are ignored by Git.

## Usage

```bash
python main.py generate --src_dir data/raw_sources --n_mix 8000 --n_src 3
python main.py train
python main.py evaluate --ckpt checkpoints_3src_2/best.ckpt
python main.py separate --mix path/to/mixture.wav --ckpt checkpoints_3src_2/best.ckpt
```

The active configuration files are:

- `configs/data.yaml`
- `configs/train.yaml`
