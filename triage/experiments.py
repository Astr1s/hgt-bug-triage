"""The experiment matrix: every graph, run and seed the thesis reports."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .graph import GraphConfig
from .text import TextEncoderConfig
from .train import TrainConfig

CORE_SEEDS = (42, 73, 101)
SIX_SEEDS = (42, 73, 101, 137, 211, 307)
EXTRA_SEEDS = (401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301)
PAIRED_SEEDS = SIX_SEEDS + EXTRA_SEEDS


CANDIDATE_RELATIONS = ("candidate_by_component", "candidate_by_product", "candidate_by_reporter_history", "candidate_by_similar_bug")
COLLABORATION_RELATIONS = ("cc_with", "rev_cc_with", "co_touched_with", "rev_co_touched_with", "reassigned_to", "rev_reassigned_to")
TYPED_HISTORY_RELATIONS = ("blocks_past_bug", "depends_on_past_bug", "duplicate_of_past_bug", "related_to_past_bug")
SIMILARITY_RELATIONS = tuple(
    f"{context}_similar_{bucket}_past_to_bug"
    for context in ("same_component", "cross_component")
    for bucket in ("low", "medium", "high")
)
RECENCY_RELATIONS = tuple(
    name
    for role in ("cc", "fixed", "touched")
    for scope in ("component", "product")
    for bucket in ("recent_30d", "recent_180d", "recent_365d", "older")
    for name in (f"{role}_{scope}_{bucket}", f"rev_{role}_{scope}_{bucket}")
)
FAMILIES: dict[str, tuple[str, ...]] = {
    "candidates": CANDIDATE_RELATIONS,
    "collaboration": COLLABORATION_RELATIONS,
    "recency": RECENCY_RELATIONS,
    "typed_history": TYPED_HISTORY_RELATIONS,
    "relation_similarity": SIMILARITY_RELATIONS,
}
ALL_RESEARCH_RELATIONS = tuple(relation for family in FAMILIES.values() for relation in family)


BERT = TextEncoderConfig(model_name="bert-base-uncased", pooling="mean", normalization="none", max_length=384, batch_size=16)
MPNET = TextEncoderConfig(model_name="sentence-transformers/all-mpnet-base-v2", pooling="mean", normalization="l2", max_length=384, batch_size=16)


@dataclass(frozen=True)
class GraphSpec:
    name: str
    encoder: TextEncoderConfig
    config: GraphConfig
    description: str


GRAPHS: dict[str, GraphSpec] = {
    "dev97": GraphSpec(
        "dev97", BERT, GraphConfig(max_developers=100),
        "primary condition: BERT text, all five research families, developer cap 100 (binds at 97)",
    ),
    "dev209": GraphSpec(
        "dev209", BERT, GraphConfig(max_developers=500),
        "larger label space: as dev97 with developer cap 500 (never binds, 209 developers)",
    ),
    "mpnet": GraphSpec(
        "mpnet", MPNET, GraphConfig(max_developers=100, recency_expertise=False),
        "encoder comparison: all-mpnet-base-v2 with L2-normalised embeddings, recency family omitted",
    ),
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    graph: str
    model_type: str
    drop_family: str | None = None      # a FAMILIES key, "all_research", or None
    hidden_dim: int = 96
    num_layers: int = 3
    zero_bug_features: bool = False

    @property
    def drop_relations(self) -> tuple[str, ...]:
        if self.drop_family is None:
            return ()
        if self.drop_family == "all_research":
            return ALL_RESEARCH_RELATIONS
        return FAMILIES[self.drop_family]

    def train_config(self, seed: int) -> TrainConfig:
        return replace(
            TrainConfig(),
            model_type=self.model_type,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            drop_relations=self.drop_relations,
            zero_bug_features=self.zero_bug_features,
            seed=seed,
        )


_RUN_LIST = (
    # Primary condition.
    RunSpec("hgt_research_full", "dev97", "hgt"),
    RunSpec("han_research_full", "dev97", "han"),
    RunSpec("hgt_causal_base", "dev97", "hgt", drop_family="all_research"),
    RunSpec("han_causal_base", "dev97", "han", drop_family="all_research"),
    RunSpec("hgt_hybrid_full", "dev97", "hgt_hybrid"),
    RunSpec("feature_only", "dev97", "feature_only", num_layers=2),
    RunSpec("feature_only_wide", "dev97", "feature_only", hidden_dim=512),
    RunSpec("hgt_notext_full", "dev97", "hgt", zero_bug_features=True),
    RunSpec("han_notext_full", "dev97", "han", zero_bug_features=True),
    # Relation-family ablations.
    *(RunSpec(f"hgt_ablation_no_{family}", "dev97", "hgt", drop_family=family) for family in FAMILIES),
    RunSpec("han_ablation_no_recency", "dev97", "han", drop_family="recency"),
    # Larger label space.
    RunSpec("hgt_research_full", "dev209", "hgt"),
    RunSpec("han_research_full", "dev209", "han"),
    # Encoder comparison; recency is already absent from that graph.
    RunSpec("hgt_mpnet_no_recency", "mpnet", "hgt"),
)
RUNS: dict[tuple[str, str], RunSpec] = {(spec.graph, spec.name): spec for spec in _RUN_LIST}


Job = tuple[str, str, tuple[int, ...]]
_EXTENSION_SEEDS = tuple(seed for seed in PAIRED_SEEDS if seed not in CORE_SEEDS)

SUITES: dict[str, tuple[Job, ...]] = {
    "main": (
        ("dev97", "feature_only", CORE_SEEDS),
        ("dev97", "hgt_causal_base", CORE_SEEDS),
        ("dev97", "hgt_research_full", CORE_SEEDS),
        ("dev97", "hgt_hybrid_full", CORE_SEEDS),
        ("dev97", "han_research_full", CORE_SEEDS),
        ("dev97", "han_causal_base", CORE_SEEDS),
        ("dev97", "hgt_notext_full", CORE_SEEDS),
        ("dev97", "han_notext_full", CORE_SEEDS),
        ("dev97", "feature_only_wide", CORE_SEEDS),
    ),
    "ablations": (
        *(("dev97", f"hgt_ablation_no_{family}", CORE_SEEDS) for family in FAMILIES),
        ("dev97", "han_ablation_no_recency", CORE_SEEDS),
    ),
    "seeds": (
        ("dev97", "hgt_research_full", _EXTENSION_SEEDS),
        ("dev97", "han_research_full", _EXTENSION_SEEDS),
    ),
    "dev209": (
        ("dev209", "hgt_research_full", CORE_SEEDS),
        ("dev209", "han_research_full", CORE_SEEDS),
    ),
    "mpnet": (("mpnet", "hgt_mpnet_no_recency", CORE_SEEDS),),
}
SUITES["all"] = tuple(job for name in ("main", "ablations", "seeds", "dev209", "mpnet") for job in SUITES[name])

# Classical baselines are reported for these graphs at this seed.
BASELINE_GRAPHS = ("dev97", "dev209")
BASELINE_SEED = 42
