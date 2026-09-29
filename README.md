# Causal heterogeneous-graph bug triage

Code for the experiments in the thesis: assigning Eclipse Bugzilla reports to
developers with a Heterogeneous Graph Transformer (HGT) under a causal,
test-blind protocol, compared against a Heterogeneous Attention Network (HAN)
with identical scaffolding, a feature-only MLP, and classical baselines.

One script drives everything. Each stage reads the previous stage's output from
`artifacts/` and writes its own there, so the whole chain runs unattended from
the raw export to the results document and figures.

```
python pipeline.py all
```

## Requirements

- Python 3.11 or later, an NVIDIA GPU with 16 GB for the full graphs (training is
  full-batch; the 97-developer graph peaks near 14.5 GB).
- `pip install -r requirements.txt`. `torch-geometric` 2.7 is required: the
  attention analysis recomputes HGTConv's attention from its message arguments.
- The pretrained encoders in the local Hugging Face cache: `bert-base-uncased`
  and `sentence-transformers/all-mpnet-base-v2`. Nothing is downloaded at run time.
- The raw Eclipse Bugzilla export, laid out as
  `<raw-root>/Eclipse/{buglist/bugs*.csv, Bugs/bugs*/<id>.txt, BugHistory/bugs*/<id>.csv}`.
  Pass its location with `--raw-root`.

## Stages

| Command | Reads | Writes |
|---|---|---|
| `build-graph --graph dev97,dev209,mpnet` | raw export | `artifacts/graphs/<g>.pt`, `_labels.json`, `_summary.json`, `_splits.csv` |
| `baselines --graph dev97,dev209` | graph, raw text | `artifacts/runs/<g>/classical_baselines_seed42.json` |
| `train --suite main\|ablations\|seeds\|dev209\|mpnet\|all` | graph | `artifacts/runs/<g>/<run>_seed<s>_{metrics,history,run}.json`, `_model.pt`, `.log` |
| `summarize` | run directories | `experiment_summary.{json,csv}` per run directory |
| `analyse --which eligibility,tenure,leaks,drift,attention,deletion` | raw export, graph, checkpoints | `artifacts/analysis/*.json` |
| `report` | everything above | `artifacts/results.md` |
| `figures` | run directories, summary, baselines | `figures/*.pdf`, `*.png` |

Every stage skips outputs that already exist; pass `--force` to recompute.
Training is restartable at seed granularity, so an interrupted suite resumes
where it stopped. `--device cpu` runs everything on the CPU (slowly).

## What gets built and trained

The matrix lives in [`triage/experiments.py`](triage/experiments.py); nothing
else decides what runs.

**Graphs.** All three use the same eligibility rule (at least 10 label-observed
bugs in the history period and 2 in the training period, then the most active
developers up to a cap), the same 50/20/15/15 chronological split, and the same
similarity edges (five most similar older bugs, bucketed by quantiles fitted on
history edges only).

| Graph | Text encoder | Developer cap | Research relation families |
|---|---|---|---|
| `dev97` | `bert-base-uncased`, mean pooling | 100 (binds at 97) | all five |
| `dev209` | as `dev97` | 500 (never binds; 209) | all five |
| `mpnet` | `all-mpnet-base-v2`, mean pooling, L2 | 100 | all but recency |

**Trainer recipe**, shared by every run: 250 epochs, patience 50 on validation
macro recall, AdamW at 5e-4 with weight decay 1e-3, label smoothing 0.02,
inverse-frequency class weights, hidden 96, 3 layers, 8 heads, dropout 0.3,
full batch, 1,000-sample bootstrap on the test split. The test split is scored
once, after the best validation checkpoint is restored.

**Runs on `dev97`.** `hgt_research_full` and `han_research_full` on 16 seeds;
on seeds 42/73/101: `hgt_causal_base` and `han_causal_base` (all 68 research
relations dropped), `hgt_hybrid_full`, `feature_only` (2 layers) and
`feature_only_wide` (hidden 512), `hgt_notext_full` and `han_notext_full` (bug
features zeroed), one `hgt_ablation_no_<family>` per family and
`han_ablation_no_recency`.
**On `dev209`:** both full arms, 3 seeds. **On `mpnet`:** `hgt_mpnet_no_recency`, 3 seeds.

Cost on one 16 GB GPU: an HGT run on the full graph takes about 25 minutes
(35 on `dev209`), a HAN run about 2, the MLP under a minute. The complete matrix
is 80 runs and roughly 17 GPU-hours, plus the graph builds (BERT over 64k to 88k
reports each).

## Layout

```
pipeline.py            the driver
triage/
  raw.py               parse the Bugzilla export into report-time records
  splits.py            chronological periods with label censoring
  text.py              frozen transformer embeddings
  graph.py             graph assembly and summary
  models.py            HGT, HAN, hybrid and MLP models
  metrics.py           ranking metrics, bootstrap, exact sign-flip test
  train.py             one training run
  baselines.py         classical baselines
  summarize.py         seed aggregation and paired ablation deltas
  experiments.py       the experiment matrix
  analysis/            eligibility, tenure, leaks, drift, attention
  report.py            results.md
  figures.py           figures
```

## Notes on reproduction

- GPU training is not bit-reproducible across hardware or library versions;
  seeds fix initialisation and data order, and every run manifest records the
  device, CUDA, torch and torch-geometric versions it ran with.
- Label maps, split files and the eligibility cache contain developer
  identities from the export and are written only under `artifacts/`, which is
  not committed. The analyses that would expose them write rank tokens instead.
- `test_balanced_top1` and `test_macro_recall` in the metrics files are the same
  quantity (recall averaged over classes at k = 1).
