#!/usr/bin/env python3
"""Training and Fine-Tuning script for the Quantized Retrieval-Augmented Medical VLM Pipeline."""

import os
import sys
import logging
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import package modules
from medical_vlm_pipeline import (
    PipelineConfig,
    MedicalVLMPipeline,
    MedicalCase,
    MedicalCaseDataset,
    load_iu_chest_xray_cases,
)
from medical_vlm_pipeline.alignment import infonce_loss

# Setup rich console logger
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("train_pipeline")


def generate_synthetic_cases(num_cases: int = 24) -> list[MedicalCase]:
    """Generates synthetic paired Chest X-ray cases if real dataset is not connected."""
    logger.info(f"Generating {num_cases} synthetic X-ray cases for dry-run training...")
    cases = []
    reports = [
        "Normal chest X-ray. The lungs are clear. Cardiomediastinal silhouette is normal.",
        "Mild cardiomegaly is present. Prominence of the pulmonary vasculature suggests mild congestion.",
        "Increased opacity in the right lower lobe, highly suggestive of lobar pneumonia.",
        "Bilateral pleural effusions with bibasilar atelectasis, worse on the left side.",
        "Hyperinflation of the lungs with flattening of the diaphragms, consistent with COPD.",
        "No focal consolidation, pneumothorax, or pleural effusion. Heart size is normal.",
    ]
    projections = ["Frontal", "Lateral"]

    for i in range(num_cases):
        case_id = f"SYN-CXR-{1000 + i}"
        report = reports[i % len(reports)]
        projection = projections[i % len(projections)]

        # Using a dummy path (will trigger dataset's synthetic image fallback)
        case = MedicalCase(
            case_id=case_id,
            image_path=Path(f"data/images/{case_id}.png"),
            report_text=report,
            label=projection,
            modality="Chest X-ray",
        )
        cases.append(case)
    return cases


def main():
    console.print("\n[bold cyan]============================================================[/bold cyan]")
    console.print("[bold white]   TRAINING & FINE-TUNING RETRIEVAL-AUGMENTED MEDICAL VLM   [/bold white]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    # 1. Load Configurations
    config = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training accelerator device: {device}")

    # Define actual dataset paths
    dataset_dir = "/kaggle/input/iu-chest-x-rays-cleaned"
    csv_path = os.path.join(dataset_dir, "cleaned_dataset.csv")
    images_dir = os.path.join(dataset_dir, "resized_images/256")

    # 2. Load Dataset
    if os.path.exists(csv_path):
        logger.info(f"Connected to IU Chest X-rays dataset on Kaggle at: {csv_path}")
        train_cases = load_iu_chest_xray_cases(csv_path, images_dir)
    else:
        logger.warning(
            f"Dataset CSV not found at: {csv_path}. Falling back to high-quality synthetic dry-run."
        )
        train_cases = generate_synthetic_cases(num_cases=32)

    if not train_cases:
        logger.error("No training cases loaded. Exiting.")
        sys.exit(1)

    # 3. Instantiate Tokenizer & Wrapper
    # Use fallback Bidirectional LSTM or HuggingFace Tokenizer
    from transformers import AutoTokenizer

    try:
        logger.info(f"Loading medical text tokenizer: {config.encoders.text_encoder}")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        logger.warning(f"Could not load HuggingFace Tokenizer: {e}. Running text tokenizer-free.")
        tokenizer = None

    # Wrap in PyTorch Dataset & DataLoader
    train_dataset = MedicalCaseDataset(
        cases=train_cases,
        tokenizer=tokenizer,
        max_length=128,
        image_size=config.data.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=len(train_dataset) > 8,
    )

    # 4. Instantiate VLM Pipeline
    class_names = ["Frontal", "Lateral"]
    num_classes = len(class_names)

    logger.info("Instantiating End-to-End Multimodal VLM Pipeline...")
    pipeline = MedicalVLMPipeline(config, num_classes=num_classes, class_names=class_names)
    pipeline = pipeline.to(device)

    # 5. Define Optimizers & Schedulers
    # Separate parameters into encoder/heads for customized learning rates if needed
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 6. Training Loop
    epochs = 10
    best_loss = float("inf")
    logger.info(f"Starting Joint Contrastive-Classification Training for {epochs} Epochs...")

    for epoch in range(1, epochs + 1):
        pipeline.train()
        epoch_loss = 0.0
        epoch_contrastive = 0.0
        epoch_class = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            images = batch["image"].to(device)
            reports = batch["report_text"]
            labels = batch["label"]

            # Map text label string to index tensor
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}
            label_idxs = torch.tensor(
                [label_to_idx.get(lbl, 0) for lbl in labels], dtype=torch.long, device=device
            )

            # A. Classification Loss via forward
            classification_logits = pipeline(images)
            classification_loss = nn.CrossEntropyLoss()(classification_logits, label_idxs)

            # B. Cross-modal Contrastive Loss via embeddings
            # For InfoNCE, we require tokenized input_ids
            if "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                attention_mask = (
                    batch["attention_mask"].to(device) if "attention_mask" in batch else None
                )

                image_embeddings = pipeline.encode_image(images)
                text_embeddings = pipeline.encode_text(input_ids, attention_mask)

                contrastive_loss = infonce_loss(image_embeddings, text_embeddings)
            else:
                # Dummy placeholder if tokenizer is offline/failed
                contrastive_loss = torch.tensor(0.0, device=device)

            # Joint optimization objective
            total_loss = classification_loss + 0.5 * contrastive_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_contrastive += contrastive_loss.item()
            epoch_class += classification_loss.item()

            progress_bar.set_postfix(
                {"Loss": f"{total_loss.item():.4f}", "Class": f"{classification_loss.item():.4f}"}
            )

        scheduler.step()

        # Average metrics
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_contrastive = epoch_contrastive / num_batches
        avg_class = epoch_class / num_batches

        console.print(
            f"[bold green]✓ Epoch {epoch:02d}/{epochs:02d}[/bold green] | "
            f"Total Loss: [bold cyan]{avg_loss:.4f}[/bold cyan] | "
            f"Contrastive (InfoNCE): {avg_contrastive:.4f} | "
            f"Classification: {avg_class:.4f}"
        )

        # Save Best Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path("best_model.pt")
            torch.save(pipeline.state_dict(), checkpoint_path)
            logger.info(f"New best model saved successfully to: {checkpoint_path}")

    # 7. Post-Training Index Generation
    console.print("\n[bold cyan]============================================================[/bold cyan]")
    console.print("[bold white]   BUILDING VECTOR INDEX FROM FINETUNED VLM EMBEDDINGS      [/bold white]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    # Load best checkpoint weights
    if os.path.exists("best_model.pt"):
        pipeline.load_state_dict(torch.load("best_model.pt"))
        logger.info("Loaded best finetuned model weights successfully.")

    # Rebuild optimized Quantized index
    pipeline.build_vector_index(train_loader)

    # Validate end-to-end RAG Inference
    logger.info("Testing post-training RAG clinical diagnosis on query case...")
    query_image = train_dataset[0]["image"].unsqueeze(0).to(device)
    diag_out = pipeline.diagnose(query_image)

    table = Table(title="Finetuned Diagnosis Inference Report", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=25)
    table.add_column("Value")

    table.add_row("Predicted Label", diag_out.diagnosis)
    table.add_row("Confidence Score", f"{diag_out.confidence * 100:.2f}%")
    table.add_row("Uncertainty (Entropy)", f"{diag_out.uncertainty:.4f}" if diag_out.uncertainty is not None else "N/A")
    table.add_row("Retrieved Matches", str(len(diag_out.retrieved_cases)))
    table.add_row("Generated Report", diag_out.report)

    console.print(table)
    console.print(
        "\n[bold green]============================================================[/bold green]"
    )
    console.print("[bold green]✓ PIPELINE TRAINING & RETRIEVAL TUNING COMPLETED SUCCESSFULLY![/bold green]")
    console.print("[bold green]============================================================[/bold green]\n")


if __name__ == "__main__":
    main()
