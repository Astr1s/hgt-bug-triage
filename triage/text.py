"""Transformer embeddings of the report-time bug text."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str = "bert-base-uncased"
    pooling: str = "mean"            # "mean" (attention-mask aware) or "cls"
    normalization: str = "none"      # "none" or "l2"
    max_length: int = 384
    batch_size: int = 16
    local_files_only: bool = True

    @property
    def name(self) -> str:
        name = f"{self.model_name}:{self.pooling}"
        return name if self.normalization == "none" else f"{name}:{self.normalization}"


def pool_transformer_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
    normalization: str,
) -> torch.Tensor:
    if pooling == "cls":
        pooled = last_hidden_state[:, 0, :]
    elif pooling == "mean":
        expanded_mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported transformer pooling mode: {pooling}")
    if normalization == "l2":
        return torch.nn.functional.normalize(pooled, p=2, dim=1)
    if normalization != "none":
        raise ValueError(f"Unsupported embedding normalization: {normalization}")
    return pooled


def encode_texts(texts: list[str], config: TextEncoderConfig, device: torch.device) -> torch.Tensor:
    """Embed every text with a frozen pretrained transformer; returns an [n, hidden] CPU tensor."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=config.local_files_only)
    model = AutoModel.from_pretrained(config.model_name, local_files_only=config.local_files_only)
    model.eval().to(device)

    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=config.max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = pool_transformer_embeddings(
                outputs.last_hidden_state, encoded["attention_mask"], config.pooling, config.normalization
            )
            batches.append(pooled.cpu())
            print(f"Encoded {min(start + config.batch_size, len(texts))}/{len(texts)} bug texts", flush=True)
    return torch.cat(batches, dim=0)
