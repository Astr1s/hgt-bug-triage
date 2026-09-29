from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Map ``cuda`` / ``cpu`` / ``auto`` to a device, refusing to fall back silently."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU; pass --device cpu to run on CPU.")
        return torch.device("cuda:0")
    if requested == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    return f"{device} ({props.name}, {props.total_memory / 1024 ** 3:.1f} GB)"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def optional_json(path: Path) -> Any | None:
    path = Path(path)
    return read_json(path) if path.exists() else None
