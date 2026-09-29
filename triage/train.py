"""Train one model configuration on a saved graph and write its metrics."""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData

from .metrics import bootstrap_ranking_intervals, ranking_metrics
from .models import build_model, checkpoint_payload
from .util import describe_device, seed_everything, write_json

SELECTION_METRICS = ("val_macro_recall", "val_macro_f1", "val_mrr", "val_top1", "val_top3")


@dataclass(frozen=True)
class TrainConfig:
    model_type: str = "hgt"                 # hgt | han | hgt_hybrid | feature_only
    epochs: int = 250
    patience: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.02
    compatibility_aux_weight: float = 0.20  # hgt_hybrid only
    hidden_dim: int = 96
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.30
    class_weights: bool = True
    drop_relations: tuple[str, ...] = ()    # relation names removed from the graph before training
    zero_bug_features: bool = False         # structure-only control: bug features replaced by zeros
    seed: int = 42
    bootstrap_samples: int = 1000
    selection_metric: str = "val_macro_recall"
    log_every: int = 5


def class_weights(y: torch.Tensor, train_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y[train_mask], minlength=num_classes).float()
    nonzero = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if torch.any(nonzero):
        weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * float(nonzero.sum()))
    return weights


def drop_relations(data: HeteroData, relations: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Delete every edge type whose relation name is listed; returns what was removed."""
    wanted = set(relations)
    present = {edge_type[1] for edge_type in data.edge_types}
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(f"Relations to drop are not in the graph: {', '.join(missing)}")
    removed = [edge_type for edge_type in list(data.edge_types) if edge_type[1] in wanted]
    for edge_type in removed:
        del data[edge_type]
    if removed:
        print(f"Dropped {len(removed)} edge types; {len(data.edge_types)} remain.", flush=True)
    return removed


def validate_graph(data: HeteroData) -> None:
    for name in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data["bug"], name):
            raise RuntimeError(f"Graph is missing data['bug'].{name}")
        if int(getattr(data["bug"], name).sum().item()) == 0:
            raise RuntimeError(f"Graph has an empty {name.removesuffix('_mask')} split")
    overlap = data["bug"].train_mask.to(torch.int8) + data["bug"].val_mask.to(torch.int8) + data["bug"].test_mask.to(torch.int8)
    if bool((overlap > 1).any()):
        raise RuntimeError("Train, validation, and test masks overlap")


def model_logits_and_aux(model: torch.nn.Module, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor | None]:
    forward_with_aux = getattr(model, "forward_with_aux", None)
    if callable(forward_with_aux):
        return forward_with_aux(data)
    return model(data), None


@torch.no_grad()
def evaluate_splits(
    model: torch.nn.Module,
    data: HeteroData,
    split_masks: dict[str, torch.Tensor],
    bootstrap_split: str | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, float]:
    model.eval()
    logits = model(data)
    y = data["bug"].y
    results: dict[str, float] = {}
    for split_name, mask in split_masks.items():
        split_logits, split_y = logits[mask], y[mask]
        metrics = ranking_metrics(split_logits, split_y)
        metrics["loss"] = float(F.cross_entropy(split_logits, split_y).item()) if split_y.numel() else 0.0
        if split_name == bootstrap_split:
            metrics.update(bootstrap_ranking_intervals(split_logits, split_y, bootstrap_samples, bootstrap_seed))
        results.update({f"{split_name}_{key}": value for key, value in metrics.items()})
    return results


def train_epoch(model: torch.nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, aux_weight: float) -> float:
    model.train()
    optimizer.zero_grad()
    logits, auxiliary_logits = model_logits_and_aux(model, data)
    mask = data["bug"].train_mask
    targets = data["bug"].y[mask]
    loss = criterion(logits[mask], targets)
    if auxiliary_logits is not None and aux_weight > 0.0:
        loss = loss + aux_weight * criterion(auxiliary_logits[mask], targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_run(graph_path: Path, config: TrainConfig, output_stem: Path, device: torch.device) -> dict[str, Any]:
    """Train and evaluate one run; writes ``<stem>_{model.pt,metrics,history,run}.json``."""
    if config.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    started_at = datetime.now(timezone.utc)
    seed_everything(config.seed)
    print(f"Training {output_stem.name}: {config} on {describe_device(device)}", flush=True)

    data: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)
    drop_relations(data, config.drop_relations)
    if config.zero_bug_features:
        data["bug"].x = torch.zeros_like(data["bug"].x)
    validate_graph(data)
    num_classes = int(data["bug"].y.max().item()) + 1

    model = build_model(config.model_type, data, num_classes, config.hidden_dim, config.num_layers, config.heads, config.dropout).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {param_count:,}", flush=True)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    run_json = output_stem.parent / f"{output_stem.name}_run.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "training_mode": "full_batch",
        "graph_path": str(Path(graph_path).resolve()),
        "graph_split_counts": getattr(data, "split_counts", {}),
        "graph_split_boundaries": getattr(data, "split_boundaries", {}),
        "text_encoder": getattr(data, "text_encoder", None),
        "trainable_parameters": param_count,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config),
    }
    write_json(run_json, manifest)

    weights = class_weights(data["bug"].y, data["bug"].train_mask, num_classes).to(device) if config.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    data = data.to(device)

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_loss = 0.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, config.compatibility_aux_weight)
        current = {"epoch": float(epoch), "loss": loss, **evaluate_splits(model, data, {"train": data["bug"].train_mask, "val": data["bug"].val_mask})}
        history.append(current)
        if current[config.selection_metric] > best_val:
            best_val = current[config.selection_metric]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_loss, stale_epochs = epoch, loss, 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % config.log_every == 0:
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | train top1 {current['train_top1']:.4f} | "
                f"val macro recall {current['val_macro_recall']:.4f} | val MRR {current['val_mrr']:.4f}",
                flush=True,
            )
        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience}).", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    # The test split is read only here, after selection on validation is complete.
    model.load_state_dict(best_state)
    metrics = evaluate_splits(
        model, data,
        {"train": data["bug"].train_mask, "val": data["bug"].val_mask, "test": data["bug"].test_mask},
        bootstrap_split="test", bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.seed + 100_003,
    )
    metrics.update(
        {
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "training_loss_at_best": best_loss,
            "selection_metric": config.selection_metric,
            "best_validation_score": best_val,
            "test_evaluated_after_selection": True,
            "seed": config.seed,
            "trainable_parameters": param_count,
        }
    )
    if hasattr(model, "compatibility_gate_logit"):
        metrics["compatibility_gate"] = float(torch.sigmoid(model.compatibility_gate_logit.detach()).cpu().item())

    model_path = output_stem.parent / f"{output_stem.name}_model.pt"
    torch.save(checkpoint_payload(model, data, config, metrics), model_path)
    write_json(output_stem.parent / f"{output_stem.name}_metrics.json", metrics)
    write_json(output_stem.parent / f"{output_stem.name}_history.json", history)
    completed_at = datetime.now(timezone.utc)
    manifest.update(
        {
            "status": "completed",
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "best_epoch": best_epoch,
            "test_evaluated_after_selection": True,
        }
    )
    write_json(run_json, manifest)
    print(f"Finished {output_stem.name}: test top1 {metrics['test_top1']:.4f}, macro recall {metrics['test_macro_recall']:.4f}", flush=True)
    return metrics
