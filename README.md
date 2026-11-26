# Chorus: Shared N-gram Embeddings with Layer-wise Adaptation

**Stephen Chung and Wenyu Du · DualverseAI**

[Read the paper on arXiv](https://arxiv.org/abs/2611.23386).
Trained weights (Chorus and the matching model without tables, three seeds each): https://huggingface.co/stephenchungmh/chorus

Chorus looks up token and n-gram embeddings once, then reuses them at the input
and intermediate layers with layer-specific adaptation. Sharing the tables and
training them with Ember reduces memory use while improving predictive performance.

This repository provides Chorus, Engram, FactorEngram and a plain model, with the
ClimbMix and FineWeb training settings used in our paper.

**Autonomous Research Program:** This work extends a human-verified finding made
by an [rStation](https://rstation.ai/) agent as part of our
[Autonomous Research Program](https://dualverse.ai/auto_research).

## Getting started

Use Python 3.10 or later and a CUDA-capable PyTorch installation. We used RTX 5090s
for the standard-size experiments; these GPUs need a PyTorch build with CUDA 12.8
or later.

```bash
git clone https://github.com/dualverse-ai/chorus.git
cd chorus
pip install -r requirements.txt
```

Prepare the training and validation data:

```bash
python prepare_data.py --dataset climbmix --data-dir data/climbmix
python prepare_data.py --dataset fineweb --data-dir data/fineweb --steps 10000
```

ClimbMix preparation downloads the public corpus shards and packs them using the
included 8,192-token tokenizer. FineWeb uses the public Parameter Golf tokenized
data. The FineWeb download above supports both model sizes.

## Training

Train Chorus with the nine-layer ClimbMix setting:

```bash
python train.py --config configs/climbmix9_chorus.json \
  --data-dir data/climbmix --out runs/climbmix9_chorus
```

Replace `chorus` in the configuration filename with `engram`, `factorengram`, or
`plain` to train a baseline. Use `--seed 2027` or `--seed 3141` for the other two
paper seeds; the default is `1337`.

| Configuration prefix | Dataset | Model | Table size |
|---|---|---|---|
| `climbmix9` | ClimbMix | 9 layers, width 512 | 76M |
| `fineweb9` | FineWeb | 9 layers, width 512 | 26M |
| `fineweb18` | FineWeb | 18 layers, width 1024 | 60M |

Table size counts added parameters. The plain model has no added n-gram tables.
Use `--table-size 1.2B` to change the budget. Use `--microbatch 8` to reduce GPU
memory use while retaining the configured total batch size.

Training writes validation metrics, a final `model.pt`, and a resumable `last.pt`.
To continue an interrupted run:

```bash
python train.py --config configs/climbmix9_chorus.json \
  --data-dir data/climbmix --out runs/climbmix9_chorus \
  --resume runs/climbmix9_chorus/last.pt
```

[`docs/experiments.md`](docs/experiments.md) lists the training settings and
commands for the comparisons and table-size sweep.

## Evaluation

Compute validation bits per byte from a trained checkpoint:

```bash
python evaluate.py --checkpoint runs/climbmix9_chorus/model.pt \
  --data-dir data/climbmix --output runs/climbmix9_chorus/evaluation.json
```

For the 22-task downstream evaluation of the FineWeb models:

```bash
python prepare_data.py --dataset downstream --data-dir data/downstream
python evaluate.py --checkpoint runs/fineweb18_chorus/model.pt \
  --data-dir data/fineweb --downstream data/downstream \
  --output runs/fineweb18_chorus/downstream.json
```

The evaluator follows the paper’s DCLM protocol and reports each task’s accuracy
and the overall mean score.

## License and acknowledgements

The code is licensed under [Apache 2.0](LICENSE). The backbones build on
[nanochat](https://github.com/karpathy/nanochat) and
[Parameter Golf](https://github.com/openai/parameter-golf).
See [NOTICE.md](NOTICE.md) for attribution and baseline references.
