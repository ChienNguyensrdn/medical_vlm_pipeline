# Source Scaffold

This folder contains the implementation scaffold for the Quantized Retrieval-Augmented Medical Vision-Language Diagnosis project.

## Module Map

```text
medical_vlm_pipeline/
  config.py          Project and model configuration dataclasses
  data.py            Dataset records and loader interfaces
  encoders.py        Image and text encoder interfaces
  alignment.py       Cross-modal contrastive alignment losses
  quantization.py    Latent compression interfaces
  retrieval.py       Vector index and retrieval interfaces
  heads.py           Diagnosis and auxiliary prediction heads
  generation.py      Report generation interfaces
  explainability.py  Attention/retrieval explanation hooks
  pipeline.py        End-to-end orchestration skeleton
```

## First Implementation Milestones

1. Build a dataset adapter for paired medical image/report records.
2. Implement Swin/3D Swin image encoder and PubMedBERT text encoder wrappers.
3. Train image-text alignment with InfoNCE.
4. Add FAISS retrieval over the shared latent space.
5. Add quantized retrieval experiments.
6. Evaluate diagnosis classification, retrieval quality, report quality, and uncertainty.
