"""Centralized configuration for the TADS pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 2048
    repetition_penalty: float = 1.2


@dataclass
class VLLMModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    batch_size: int = 32
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class EmbeddingModelConfig:
    name: str = "BAAI/bge-m3"
    batch_size: int = 2048
    max_length: int = 64


@dataclass
class ClusteringConfig:
    num_clusters: int = 100
    kmeans_iterations: int = 100
    top_k: int = 3


@dataclass
class SelectionConfig:
    target_count: int = 10000
    score_thresholds: list = field(default_factory=lambda: list(range(1, 11)))
    fields: list = field(default_factory=lambda: ["Topic", "Style", "Audience", "Task"])


@dataclass
class TADSConfig:
    base_dir: str = "."
    data_dir: str = "./data"
    random_seed: int = 42
    split_ratio: float = 0.2
    eval_datasets: list = field(
        default_factory=lambda: ["bbh", "gsm", "mmlu", "truthfulqa", "tydiqa"]
    )

    # Model configs
    annotation: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    keyword_extraction: VLLMModelConfig = field(default_factory=lambda: VLLMModelConfig(
        tensor_parallel_size=4, gpu_memory_utilization=0.85,
        max_model_len=16384, batch_size=32,
        sampling=SamplingConfig(temperature=0.4, top_p=0.9, max_tokens=800, repetition_penalty=1.1),
    ))
    scoring: VLLMModelConfig = field(default_factory=lambda: VLLMModelConfig(
        tensor_parallel_size=4, gpu_memory_utilization=0.95,
        max_model_len=32768, batch_size=64,
        sampling=SamplingConfig(temperature=0.3, top_p=0.9, max_tokens=300),
    ))
    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    # --- Derived paths ---

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.data_dir, "eval")

    @property
    def split_target_dir(self) -> str:
        return os.path.join(self.eval_dir, "split_target")

    @property
    def split_eval_dir(self) -> str:
        return os.path.join(self.eval_dir, "split_eval")

    # Step 1 outputs
    @property
    def combined_target_parquet(self) -> str:
        return os.path.join(self.data_dir, "step1_combined_target.parquet")

    @property
    def target_annotation_pt(self) -> str:
        return os.path.join(self.data_dir, "step1_target_annotation.pt")

    @property
    def target_annotation_json(self) -> str:
        return os.path.join(self.data_dir, "step1_target_annotation.json")

    # Step 2 outputs
    @property
    def cluster_ids_json(self) -> str:
        return os.path.join(self.data_dir, "cluster_ids.json")

    @property
    def tag_id_mapping_json(self) -> str:
        return os.path.join(self.data_dir, "tag_id_mapping.json")

    @property
    def train_embeds_dir(self) -> str:
        return os.path.join(self.data_dir, "train_embeds_and_tags")

    @property
    def train_data_parquet(self) -> str:
        return os.path.join(self.data_dir, "train-00000-of-00001.parquet")

    @property
    def train_with_tags_pt(self) -> str:
        return os.path.join(self.data_dir, "train_with_tags.pt")

    # Step 3 outputs
    @property
    def extracted_keywords_json(self) -> str:
        return os.path.join(self.data_dir, "extracted_keywords.json")

    @property
    def scored_data_pt(self) -> str:
        return os.path.join(self.data_dir, "scored_data.pt")

    @property
    def filtered_dir(self) -> str:
        return os.path.join(self.data_dir, "filtered")

    # Step 4 outputs
    @property
    def selected_dir(self) -> str:
        return os.path.join(self.data_dir, "selected")

    def filtered_pt(self, threshold: int) -> str:
        return os.path.join(self.filtered_dir, f"samples_gte_{threshold}.pt")

    def distribution_pt(self, field_name: str, threshold: int) -> str:
        return os.path.join(
            self.selected_dir,
            f"distribution_10k_{field_name}_gte_{threshold}.pt",
        )

    def fused_json(self, threshold: int) -> str:
        return os.path.join(
            self.selected_dir,
            f"fused_10k_gte_{threshold}.json",
        )


def _apply_dict(cfg: TADSConfig, d: dict) -> TADSConfig:
    """Apply a flat or nested dict onto a TADSConfig."""
    simple_fields = {"base_dir", "data_dir", "random_seed", "split_ratio", "eval_datasets"}
    for k, v in d.items():
        if k in simple_fields:
            setattr(cfg, k, v)

    models = d.get("models", {})
    if "annotation" in models:
        cfg.annotation = _build_vllm_config(models["annotation"])
    if "keyword_extraction" in models:
        cfg.keyword_extraction = _build_vllm_config(models["keyword_extraction"])
    if "scoring" in models:
        cfg.scoring = _build_vllm_config(models["scoring"])
    if "embedding" in models:
        emb = models["embedding"]
        cfg.embedding = EmbeddingModelConfig(
            name=emb.get("name", cfg.embedding.name),
            batch_size=emb.get("batch_size", cfg.embedding.batch_size),
            max_length=emb.get("max_length", cfg.embedding.max_length),
        )

    if "clustering" in d:
        c = d["clustering"]
        cfg.clustering = ClusteringConfig(
            num_clusters=c.get("num_clusters", 100),
            kmeans_iterations=c.get("kmeans_iterations", 100),
            top_k=c.get("top_k", 3),
        )

    if "selection" in d:
        s = d["selection"]
        cfg.selection = SelectionConfig(
            target_count=s.get("target_count", 10000),
            score_thresholds=s.get("score_thresholds", list(range(1, 11))),
            fields=s.get("fields", ["Topic", "Style", "Audience", "Task"]),
        )

    return cfg


def _build_vllm_config(d: dict) -> VLLMModelConfig:
    sampling_d = d.get("sampling", {})
    sampling = SamplingConfig(
        temperature=sampling_d.get("temperature", 1.0),
        top_p=sampling_d.get("top_p", 0.95),
        max_tokens=sampling_d.get("max_tokens", 2048),
        repetition_penalty=sampling_d.get("repetition_penalty", 1.2),
    )
    return VLLMModelConfig(
        name=d.get("name", "Qwen/Qwen2.5-7B-Instruct"),
        tensor_parallel_size=d.get("tensor_parallel_size", 2),
        gpu_memory_utilization=d.get("gpu_memory_utilization", 0.90),
        max_model_len=d.get("max_model_len"),
        batch_size=d.get("batch_size", 32),
        sampling=sampling,
    )


def load_config(yaml_path: str = None, **overrides) -> TADSConfig:
    """Load config from YAML file with optional CLI overrides.

    Args:
        yaml_path: Path to YAML config file. If None, uses defaults.
        **overrides: Direct field overrides (e.g., data_dir="/other/path").
    """
    cfg = TADSConfig()

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _apply_dict(cfg, raw)

    # Apply simple overrides
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    # Resolve base_dir-relative data_dir
    if not os.path.isabs(cfg.data_dir):
        cfg.data_dir = os.path.join(cfg.base_dir, cfg.data_dir)
    cfg.data_dir = os.path.abspath(cfg.data_dir)

    return cfg
