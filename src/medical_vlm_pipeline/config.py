"""Configuration objects for the medical VLM pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    image_root: Path = Path("data/images")
    report_path: Path = Path("data/reports.csv")
    modality: str = "MRI"
    image_size: int = 224
    volume_depth: int | None = None


@dataclass
class EncoderConfig:
    image_encoder: str = "swin_tiny_patch4_window7_224"
    text_encoder: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    embedding_dim: int = 768
    projection_dim: int = 256


@dataclass
class RetrievalConfig:
    engine: str = "faiss"
    index_path: Path = Path("artifacts/faiss.index")
    top_k: int = 5
    quantized: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    temperature: float = 0.07


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    encoders: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
