"""Model definitions."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, HGTConv, Linear

Metadata = tuple[list[str], list[tuple[str, str, str]]]


class BugFeatureMLP(nn.Module):
    """Feature-only baseline that predicts assignees from bug attributes."""

    def __init__(self, bug_in_dim: int, num_classes: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("Feature-only MLP needs at least one hidden layer")
        self.dropout = dropout
        self.input = Linear(bug_in_dim, hidden_dim)
        self.hidden = nn.ModuleList(Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.classifier = Linear(hidden_dim, num_classes)

    @classmethod
    def from_data(cls, data: HeteroData, num_classes: int, hidden_dim: int, num_layers: int, dropout: float, **_: Any) -> "BugFeatureMLP":
        return cls(data["bug"].x.size(-1), num_classes, hidden_dim, num_layers, dropout)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = F.gelu(self.input(data["bug"].x.float()))
        x = self.norms[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer, norm in zip(self.hidden, self.norms[1:]):
            message = F.dropout(F.gelu(layer(x)), p=self.dropout, training=self.training)
            x = norm(x + message)
        return self.classifier(x)


class HGTEncoder(nn.Module):
    """Shared encoder for bug and metadata/developer nodes."""

    def __init__(
        self,
        metadata: Metadata,
        num_nodes: dict[str, int],
        bug_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_types = list(metadata[0])
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.bug_lin = Linear(bug_in_dim, hidden_dim)
        self.embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            if node_type != "bug":
                self.embeddings[node_type] = nn.Embedding(num_nodes[node_type], hidden_dim)

        self.convs = nn.ModuleList(self._build_conv(metadata, hidden_dim, heads) for _ in range(num_layers))
        self.layer_norms = nn.ModuleList(
            nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})
            for _ in range(num_layers)
        )

    @staticmethod
    def _build_conv(metadata: Metadata, hidden_dim: int, heads: int) -> nn.Module:
        return HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=heads)

    def _features(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {"bug": F.gelu(self.bug_lin(data["bug"].x.float()))}
        for node_type, embedding in self.embeddings.items():
            if hasattr(data[node_type], "node_id"):
                node_id = data[node_type].node_id
            else:
                node_id = torch.arange(data[node_type].num_nodes, device=data["bug"].x.device)
            x_dict[node_type] = embedding(node_id)
        return x_dict

    def encode(self, data: HeteroData) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x_dict = self._features(data)
        bug_input = x_dict["bug"]
        for layer_idx, conv in enumerate(self.convs):
            out_dict = conv(x_dict, data.edge_index_dict)
            next_x_dict: dict[str, torch.Tensor] = {}
            for node_type, x in x_dict.items():
                message = out_dict.get(node_type)
                if message is None:
                    updated = x
                else:
                    updated = x + F.dropout(F.gelu(message), p=self.dropout, training=self.training)
                next_x_dict[node_type] = self.layer_norms[layer_idx][node_type](updated)
            x_dict = next_x_dict
        return x_dict, bug_input


class HGTBugAssigner(HGTEncoder):
    """HGT classifier with an anonymous class head."""

    def __init__(
        self,
        metadata: Metadata,
        num_nodes: dict[str, int],
        bug_in_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__(metadata, num_nodes, bug_in_dim, hidden_dim, num_layers, heads, dropout)
        self.classifier_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = Linear(hidden_dim * 2, num_classes)

    @classmethod
    def from_data(cls, data: HeteroData, num_classes: int, hidden_dim: int, num_layers: int, heads: int, dropout: float) -> "HGTBugAssigner":
        return cls(
            metadata=data.metadata(),
            num_nodes={node_type: data[node_type].num_nodes for node_type in data.node_types},
            bug_in_dim=data["bug"].x.size(-1),
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict, bug_input = self.encode(data)
        bug_features = self.classifier_norm(torch.cat([x_dict["bug"], bug_input], dim=-1))
        bug_features = F.dropout(bug_features, p=self.dropout, training=self.training)
        return self.classifier(bug_features)


class HANBugAssigner(HGTBugAssigner):
    """The comparison arm: identical scaffolding, HANConv (own attention dropout off) in place of HGTConv."""

    @staticmethod
    def _build_conv(metadata: Metadata, hidden_dim: int, heads: int) -> nn.Module:
        if heads > 0 and hidden_dim % heads != 0:
            raise ValueError(f"HANConv requires hidden_dim divisible by heads; got {hidden_dim} and {heads}")
        return HANConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=heads, dropout=0.0)


class HGTDeveloperScorer(HGTEncoder):
    """Ranker that scores bugs against the developer node embeddings directly."""

    def __init__(
        self,
        metadata: Metadata,
        num_nodes: dict[str, int],
        bug_in_dim: int,
        num_developers: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        if "developer" not in metadata[0]:
            raise ValueError("Developer scorer requires a 'developer' node type")
        super().__init__(metadata, num_nodes, bug_in_dim, hidden_dim, num_layers, heads, dropout)
        self.num_developers = num_developers
        self.bug_norm = nn.LayerNorm(hidden_dim * 2)
        self.developer_norm = nn.LayerNorm(hidden_dim)
        self.bug_proj = Linear(hidden_dim * 2, hidden_dim)
        self.developer_proj = Linear(hidden_dim, hidden_dim)
        self.developer_bias = nn.Parameter(torch.zeros(num_developers))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    @classmethod
    def from_data(cls, data: HeteroData, num_classes: int, hidden_dim: int, num_layers: int, heads: int, dropout: float) -> "HGTDeveloperScorer":
        if data["developer"].num_nodes != num_classes:
            raise ValueError("Developer scorer needs one developer node per class")
        return cls(
            metadata=data.metadata(),
            num_nodes={node_type: data[node_type].num_nodes for node_type in data.node_types},
            bug_in_dim=data["bug"].x.size(-1),
            num_developers=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    def compatibility_logits(self, x_dict: dict[str, torch.Tensor], bug_input: torch.Tensor) -> torch.Tensor:
        developer_features = x_dict["developer"]
        if developer_features.size(0) != self.num_developers:
            raise RuntimeError("Developer scorer requires full-batch data with all developer nodes present")
        bug_features = self.bug_norm(torch.cat([x_dict["bug"], bug_input], dim=-1))
        bug_features = F.dropout(F.gelu(self.bug_proj(bug_features)), p=self.dropout, training=self.training)
        developer_features = self.developer_norm(developer_features)
        developer_features = F.dropout(F.gelu(self.developer_proj(developer_features)), p=self.dropout, training=self.training)
        bug_features = F.normalize(bug_features, dim=-1)
        developer_features = F.normalize(developer_features, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (bug_features @ developer_features.t()) + self.developer_bias

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict, bug_input = self.encode(data)
        return self.compatibility_logits(x_dict, bug_input)


class HGTHybridDeveloperScorer(HGTDeveloperScorer):
    """HGT classifier gated with the developer-compatibility ranker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.classifier_norm = nn.LayerNorm(self.hidden_dim * 2)
        self.classifier = Linear(self.hidden_dim * 2, self.num_developers)
        self.compatibility_gate_logit = nn.Parameter(torch.tensor(-2.1972246))

    def forward_with_aux(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        x_dict, bug_input = self.encode(data)
        classifier_features = self.classifier_norm(torch.cat([x_dict["bug"], bug_input], dim=-1))
        classifier_features = F.dropout(classifier_features, p=self.dropout, training=self.training)
        classifier_logits = self.classifier(classifier_features)
        compatibility_logits = self.compatibility_logits(x_dict, bug_input)
        gate = torch.sigmoid(self.compatibility_gate_logit)
        return (1.0 - gate) * classifier_logits + gate * compatibility_logits, compatibility_logits

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.forward_with_aux(data)[0]


MODEL_TYPES = {
    "hgt": HGTBugAssigner,
    "han": HANBugAssigner,
    "hgt_hybrid": HGTHybridDeveloperScorer,
    "feature_only": BugFeatureMLP,
}


def build_model(model_type: str, data: HeteroData, num_classes: int, hidden_dim: int, num_layers: int, heads: int, dropout: float) -> nn.Module:
    try:
        cls = MODEL_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {sorted(MODEL_TYPES)}") from exc
    return cls.from_data(data, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout)


def checkpoint_payload(model: nn.Module, data: HeteroData, config: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "metadata": data.metadata(),
        "num_nodes": {node_type: data[node_type].num_nodes for node_type in data.node_types},
        "bug_in_dim": data["bug"].x.size(-1),
        "num_classes": int(data["bug"].y.max().item()) + 1,
        "label_to_id": getattr(data, "label_to_id", {}),
        "id_to_label": getattr(data, "id_to_label", {}),
        "training_config": asdict(config),
        "graph_provenance": {
            "text_encoder": getattr(data, "text_encoder", None),
            "split_boundaries": getattr(data, "split_boundaries", {}),
            "split_counts": getattr(data, "split_counts", {}),
        },
        "metrics": metrics,
    }
