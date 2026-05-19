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


def calculate_lcs(x: list[str], y: list[str]) -> int:
    """Find length of Longest Common Subsequence between two token sequences."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L F1-Score for sentence strings."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs_len = calculate_lcs(ref_tokens, cand_tokens)
    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(cand_tokens)
    if recall + precision == 0:
        return 0.0
    f1 = (2 * recall * precision) / (recall + precision)
    return f1


def calculate_bleu(reference: str, candidate: str) -> tuple[float, float]:
    """Calculate BLEU-1 and BLEU-4 score using NLTK."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = [reference.lower().split()]
        cand_tokens = candidate.lower().split()
        chencherry = SmoothingFunction()
        bleu1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
        bleu4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        return bleu1, bleu4
    except Exception:
        ref_tokens = set(reference.lower().split())
        cand_tokens = candidate.lower().split()
        if not cand_tokens:
            return 0.0, 0.0
        overlap = sum(1 for w in cand_tokens if w in ref_tokens)
        precision = overlap / len(cand_tokens)
        return precision, precision


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


def plot_training_metrics(csv_path: Path, output_path: Path):
    """Generates loss and classification metrics curve plots from training CSV logs."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Loss curves
        axes[0].plot(df["epoch"], df["total_loss"], label="Total Joint Loss", color="#d62728", linewidth=2.5)
        axes[0].plot(df["epoch"], df["contrastive_loss"], label="Contrastive InfoNCE Loss", color="#1f77b4", linestyle="--")
        axes[0].plot(df["epoch"], df["classification_loss"], label="Classification CE Loss", color="#2ca02c", linestyle=":")
        axes[0].set_title("Training Loss Curves", fontsize=14, fontweight="bold", pad=12)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss Value", fontsize=12)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(fontsize=10)
        
        # Plot 2: Performance metrics
        axes[1].plot(df["epoch"], df["accuracy"] * 100, label="Accuracy (%)", color="#9467bd", linewidth=2.5)
        
        f1_vals = df["f1_score"]
        if f1_vals.max() <= 1.0:
            f1_vals = f1_vals * 100
        axes[1].plot(df["epoch"], f1_vals, label="F1-Score (%)", color="#ff7f0e", linewidth=2.5)
        
        axes[1].set_title("Classification Performance Metrics", fontsize=14, fontweight="bold", pad=12)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Percentage (%)", fontsize=12)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(fontsize=10)
        
        plt.suptitle("Medical VLM End-to-End Joint Training Report", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Successfully generated training curves plot at: {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate training curves plots: {e}")


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
    epochs = 50
    best_loss = float("inf")
    
    # Initialize metrics folder and CSV logging
    metrics_dir = Path("report_kaggle")
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "training_metrics.csv"
    if metrics_file.exists():
        metrics_file.unlink()
        
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

        # Collect predictions and targets for classification metrics evaluation
        pipeline.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"]
                label_idxs = torch.tensor(
                    [label_to_idx.get(lbl, 0) for lbl in labels], dtype=torch.long, device=device
                )
                logits = pipeline(images)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(label_idxs.cpu().numpy())

        # Calculate Accuracy, Precision, Recall, F1-Score
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = float(accuracy_score(all_targets, all_preds))
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
            precision, recall, f1 = float(precision), float(recall), float(f1)
        except ImportError:
            correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
            accuracy = correct / len(all_targets) if len(all_targets) > 0 else 0.0
            precision, recall, f1 = accuracy, accuracy, accuracy

        # Append metrics to CSV for experiment logging
        import csv
        file_exists = metrics_file.exists()
        with open(metrics_file, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "total_loss", "contrastive_loss", "classification_loss", "accuracy", "precision", "recall", "f1_score", "learning_rate"])
            writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_contrastive:.6f}",
                f"{avg_class:.6f}",
                f"{accuracy:.6f}",
                f"{precision:.6f}",
                f"{recall:.6f}",
                f"{f1:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}"
            ])

        console.print(
            f"[bold green]✓ Epoch {epoch:02d}/{epochs:02d}[/bold green] | "
            f"Loss: [bold cyan]{avg_loss:.4f}[/bold cyan] | "
            f"Acc: [bold magenta]{accuracy * 100:.2f}%[/bold magenta] | "
            f"F1: [bold yellow]{f1:.4f}[/bold yellow]"
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

    # 8. Evaluate Text Generation NLP Metrics (BLEU and ROUGE-L)
    logger.info("Evaluating text generation performance across reference database...")
    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    
    # We sample up to 10 cases to prevent long CPU generation times
    eval_cases = train_cases[:10]
    for idx, case in enumerate(eval_cases):
        case_image = train_dataset[idx]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = pipeline.diagnose(case_image)
            cand_report = out.report if out.report is not None else ""
            ref_report = case.report_text
            
            b1, b4 = calculate_bleu(ref_report, cand_report)
            r_l = calculate_rouge_l(ref_report, cand_report)
            
            bleu1_scores.append(b1)
            bleu4_scores.append(b4)
            rouge_l_scores.append(r_l)
            
    mean_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0
    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    # Save final text generation evaluation metrics to a separate JSON file
    import json
    text_metrics = {
        "mean_bleu_1": mean_bleu1,
        "mean_bleu_4": mean_bleu4,
        "mean_rouge_l": mean_rouge_l,
        "num_evaluated_cases": len(eval_cases)
    }
    with open(metrics_dir / "text_generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(text_metrics, f, indent=4)
    logger.info(f"Saved text generation evaluation metrics to {metrics_dir / 'text_generation_metrics.json'}")

    # Generate training curves plot
    plot_training_metrics(metrics_file, metrics_dir / "training_curves.png")

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
    table.add_row("Mean BLEU-1", f"{mean_bleu1 * 100:.2f}%")
    table.add_row("Mean BLEU-4", f"{mean_bleu4 * 100:.2f}%")
    table.add_row("Mean ROUGE-L", f"{mean_rouge_l * 100:.2f}%")
    table.add_row("Generated Report", diag_out.report)

    console.print(table)
    console.print(
        "\n[bold green]============================================================[/bold green]"
    )
    console.print("[bold green]✓ PIPELINE TRAINING & RETRIEVAL TUNING COMPLETED SUCCESSFULLY![/bold green]")
    console.print("[bold green]============================================================[/bold green]\n")


if __name__ == "__main__":
    main()
