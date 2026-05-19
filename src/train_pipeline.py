#!/usr/bin/env python3
"""Training and Fine-Tuning script for the Quantized Retrieval-Augmented Medical VLM Pipeline."""

import os
import sys
import logging
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the quantized retrieval-augmented medical VLM pipeline."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/kaggle/input/iu-chest-x-rays-cleaned",
        help="Dataset directory containing cleaned_dataset.csv and resized_images/256.",
    )
    parser.add_argument(
        "--synthetic-cases",
        type=int,
        default=32,
        help="Number of synthetic fallback cases when the real dataset is unavailable.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip Matplotlib training curve generation for faster local smoke tests.",
    )
    return parser.parse_args()


def is_device_usable(device: torch.device) -> bool:
    """Run a tiny kernel smoke test before trusting an accelerator."""
    if device.type == "cpu":
        return True

    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)

        probe = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        ).to(device)
        x = torch.randn(2, 3, 16, 16, device=device)
        y = probe(x).mean()
        y.backward()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        logger.warning(
            "Accelerator %s failed the kernel smoke test and will be skipped: %s",
            device,
            exc,
        )
        return False


def select_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device if is_device_usable(device) else torch.device("cpu")
    if requested_device == "mps":
        is_mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device = torch.device("mps" if is_mps_available else "cpu")
        return device if is_device_usable(device) else torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if is_device_usable(device):
            return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if is_device_usable(device):
            return device
    return torch.device("cpu")


def to_jsonable(value: Any) -> Any:
    """Convert nested training objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return to_jsonable(vars(value))
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        snapshot["cuda_device_count"] = torch.cuda.device_count()
        snapshot["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return snapshot


def retrieval_results_to_dict(results: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item.case_id,
            "score": item.score,
            "label": item.label,
            "report_text": item.report_text,
        }
        for item in results
    ]


def write_case_index(path: Path, cases: list[MedicalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "image_path", "label", "modality", "report_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case.case_id,
                "image_path": str(case.image_path),
                "label": case.label,
                "modality": case.modality,
                "report_text": case.report_text,
            })


def save_training_checkpoint(
    path: Path,
    pipeline: MedicalVLMPipeline,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    config: PipelineConfig,
    class_names: list[str],
    label_to_idx: dict[str, int],
    epoch_records: list[dict[str, Any]],
    dataset_source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": pipeline.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": to_jsonable(asdict(config)),
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "epoch_records": to_jsonable(epoch_records),
        "dataset_source": dataset_source,
    }
    torch.save(checkpoint, path)


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
        matplotlib_cache_dir = Path("artifacts/matplotlib").resolve()
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

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
    args = parse_args()

    console.print("\n[bold cyan]============================================================[/bold cyan]")
    console.print("[bold white]   TRAINING & FINE-TUNING RETRIEVAL-AUGMENTED MEDICAL VLM   [/bold white]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    # 1. Load Configurations
    config = PipelineConfig()
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    device = select_device(args.device)
    logger.info(f"Using training accelerator device: {device}")
    if args.device == "mps" and device.type != "mps":
        logger.warning("Apple MPS was requested but is not available in this Python/PyTorch environment.")

    # Initialize metrics folder and durable run metadata early.
    metrics_dir = Path("report_kaggle")
    metrics_dir.mkdir(exist_ok=True)
    model_artifacts_dir = metrics_dir / "model_artifacts"
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    run_timer_start = time.perf_counter()

    # Define actual dataset paths
    dataset_dir = args.dataset_dir
    csv_path = os.path.join(dataset_dir, "cleaned_dataset.csv")
    images_dir = os.path.join(dataset_dir, "resized_images/256")

    # 2. Load Dataset
    if os.path.exists(csv_path):
        logger.info(f"Connected to IU Chest X-rays dataset on Kaggle at: {csv_path}")
        train_cases = load_iu_chest_xray_cases(csv_path, images_dir)
        dataset_source = "real"
    else:
        logger.warning(
            f"Dataset CSV not found at: {csv_path}. Falling back to high-quality synthetic dry-run."
        )
        train_cases = generate_synthetic_cases(num_cases=args.synthetic_cases)
        dataset_source = "synthetic"

    if not train_cases:
        logger.error("No training cases loaded. Exiting.")
        sys.exit(1)

    write_json(metrics_dir / "environment.json", environment_snapshot(device))
    write_json(
        metrics_dir / "run_config.json",
        {
            "run_started_at": run_started_at,
            "argv": sys.argv,
            "args": vars(args),
            "config": asdict(config),
            "dataset": {
                "source": dataset_source,
                "dataset_dir": dataset_dir,
                "csv_path": csv_path,
                "images_dir": images_dir,
                "num_cases": len(train_cases),
                "label_counts": count_values([case.label or "" for case in train_cases]),
                "modality_counts": count_values([case.modality or "" for case in train_cases]),
                "sample_case_ids": [case.case_id for case in train_cases[:10]],
            },
            "environment": environment_snapshot(device),
        },
    )
    write_case_index(metrics_dir / "case_index.csv", train_cases)

    # 3. Instantiate Tokenizer & Wrapper
    # Use fallback Bidirectional LSTM or HuggingFace Tokenizer
    try:
        from transformers import AutoTokenizer

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
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > config.training.batch_size,
    )

    # 4. Instantiate VLM Pipeline
    class_names = ["Frontal", "Lateral"]
    num_classes = len(class_names)

    logger.info("Instantiating End-to-End Multimodal VLM Pipeline...")
    pipeline = MedicalVLMPipeline(config, num_classes=num_classes, class_names=class_names)
    pipeline = pipeline.to(device)

    # 5. Define Optimizers & Schedulers
    # Separate parameters into encoder/heads for customized learning rates if needed
    optimizer = torch.optim.AdamW(
        pipeline.parameters(),
        lr=config.training.learning_rate,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs, 1),
    )

    # 6. Training Loop
    epochs = config.training.epochs
    best_loss = float("inf")
    
    # Initialize metrics CSV logging
    metrics_file = metrics_dir / "training_metrics.csv"
    if metrics_file.exists():
        metrics_file.unlink()
        
    logger.info(f"Starting Joint Contrastive-Classification Training for {epochs} Epochs...")

    epoch_records: list[dict[str, Any]] = []
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    write_json(
        model_artifacts_dir / "label_map.json",
        {
            "class_names": class_names,
            "label_to_idx": label_to_idx,
            "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
        },
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate_start = optimizer.param_groups[0]["lr"]
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
        per_class_metrics: dict[str, dict[str, float]] = {}
        confusion_matrix_data: list[list[int]] = []
        try:
            from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
            accuracy = float(accuracy_score(all_targets, all_preds))
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
            precision, recall, f1 = float(precision), float(recall), float(f1)
            class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
                all_targets,
                all_preds,
                labels=list(range(num_classes)),
                average=None,
                zero_division=0,
            )
            confusion_matrix_data = confusion_matrix(
                all_targets,
                all_preds,
                labels=list(range(num_classes)),
            ).tolist()
            for idx, class_name in enumerate(class_names):
                per_class_metrics[class_name] = {
                    "precision": float(class_precision[idx]),
                    "recall": float(class_recall[idx]),
                    "f1_score": float(class_f1[idx]),
                    "support": int(class_support[idx]),
                }
        except ImportError:
            correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
            accuracy = correct / len(all_targets) if len(all_targets) > 0 else 0.0
            precision, recall, f1 = accuracy, accuracy, accuracy

        scheduler.step()
        learning_rate_end = optimizer.param_groups[0]["lr"]
        epoch_duration_sec = time.perf_counter() - epoch_start

        is_best = avg_loss < best_loss
        epoch_record = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "contrastive_loss": avg_contrastive,
            "classification_loss": avg_class,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "learning_rate_start": learning_rate_start,
            "learning_rate": learning_rate_end,
            "epoch_duration_sec": epoch_duration_sec,
            "num_batches": num_batches,
            "num_train_samples": len(train_dataset),
            "batch_size": config.training.batch_size,
            "device": str(device),
            "is_best": is_best,
            "best_loss_before_epoch": best_loss,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix_data,
        }
        epoch_records.append(epoch_record)

        # Append metrics to CSV for experiment logging
        file_exists = metrics_file.exists()
        with open(metrics_file, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "epoch",
                    "total_loss",
                    "contrastive_loss",
                    "classification_loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "learning_rate_start",
                    "learning_rate",
                    "epoch_duration_sec",
                    "num_batches",
                    "num_train_samples",
                    "batch_size",
                    "device",
                    "is_best",
                    "best_loss_before_epoch",
                    *[
                        f"{class_name}_{metric_name}"
                        for class_name in class_names
                        for metric_name in ("precision", "recall", "f1_score", "support")
                    ],
                    "confusion_matrix_json",
                ])
            writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_contrastive:.6f}",
                f"{avg_class:.6f}",
                f"{accuracy:.6f}",
                f"{precision:.6f}",
                f"{recall:.6f}",
                f"{f1:.6f}",
                f"{learning_rate_start:.8f}",
                f"{learning_rate_end:.8f}",
                f"{epoch_duration_sec:.4f}",
                num_batches,
                len(train_dataset),
                config.training.batch_size,
                str(device),
                int(is_best),
                f"{best_loss:.6f}",
                *[
                    per_class_metrics.get(class_name, {}).get(metric_name, 0.0)
                    for class_name in class_names
                    for metric_name in ("precision", "recall", "f1_score", "support")
                ],
                json.dumps(confusion_matrix_data),
            ])

        write_json(
            metrics_dir / "epoch_metrics.json",
            {
                "class_names": class_names,
                "records": epoch_records,
            },
        )

        console.print(
            f"[bold green]✓ Epoch {epoch:02d}/{epochs:02d}[/bold green] | "
            f"Loss: [bold cyan]{avg_loss:.4f}[/bold cyan] | "
            f"Acc: [bold magenta]{accuracy * 100:.2f}%[/bold magenta] | "
            f"F1: [bold yellow]{f1:.4f}[/bold yellow]"
        )

        # Save Best Checkpoint
        if is_best:
            best_loss = avg_loss
            checkpoint_path = Path("best_model.pt")
            torch.save(pipeline.state_dict(), checkpoint_path)
            torch.save(pipeline.state_dict(), model_artifacts_dir / "best_model_state_dict.pt")
            save_training_checkpoint(
                model_artifacts_dir / "best_training_checkpoint.pt",
                pipeline=pipeline,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_loss,
                config=config,
                class_names=class_names,
                label_to_idx=label_to_idx,
                epoch_records=epoch_records,
                dataset_source=dataset_source,
            )
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
    retriever_index_base = model_artifacts_dir / "retriever_index" / "faiss_retriever"
    pipeline.retriever.save(retriever_index_base)
    torch.save(pipeline.state_dict(), model_artifacts_dir / "final_model_state_dict.pt")
    save_training_checkpoint(
        model_artifacts_dir / "final_training_checkpoint.pt",
        pipeline=pipeline,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs,
        best_loss=best_loss,
        config=config,
        class_names=class_names,
        label_to_idx=label_to_idx,
        epoch_records=epoch_records,
        dataset_source=dataset_source,
    )
    write_json(
        model_artifacts_dir / "model_metadata.json",
        {
            "model_name": "Quantized Retrieval-Augmented Medical VLM",
            "config": asdict(config),
            "class_names": class_names,
            "label_to_idx": label_to_idx,
            "dataset_source": dataset_source,
            "num_train_cases": len(train_cases),
            "best_loss": best_loss,
            "artifacts": {
                "root_best_model_state_dict": str(Path("best_model.pt")),
                "best_model_state_dict": str(model_artifacts_dir / "best_model_state_dict.pt"),
                "final_model_state_dict": str(model_artifacts_dir / "final_model_state_dict.pt"),
                "best_training_checkpoint": str(model_artifacts_dir / "best_training_checkpoint.pt"),
                "final_training_checkpoint": str(model_artifacts_dir / "final_training_checkpoint.pt"),
                "retriever_index_base": str(retriever_index_base),
                "label_map": str(model_artifacts_dir / "label_map.json"),
                "case_index": str(metrics_dir / "case_index.csv"),
            },
            "reload_notes": [
                "Instantiate PipelineConfig and MedicalVLMPipeline with the saved class_names.",
                "Load model weights from best_model_state_dict.pt or final_model_state_dict.pt.",
                "Load retriever using pipeline.retriever.load(Path('retriever_index/faiss_retriever')).",
                "Use final_training_checkpoint.pt to resume optimizer/scheduler state.",
            ],
        },
    )

    # 8. Evaluate Text Generation NLP Metrics (BLEU and ROUGE-L)
    logger.info("Evaluating text generation performance across reference database...")
    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    generation_samples: list[dict[str, Any]] = []
    
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
            generation_samples.append({
                "sample_index": idx,
                "case_id": case.case_id,
                "label": case.label,
                "modality": case.modality,
                "reference_report": ref_report,
                "generated_report": cand_report,
                "bleu_1": b1,
                "bleu_4": b4,
                "rouge_l": r_l,
                "predicted_diagnosis": out.diagnosis,
                "confidence": out.confidence,
                "uncertainty": out.uncertainty,
                "retrieved_cases": retrieval_results_to_dict(out.retrieved_cases),
                "step_metrics": out.step_metrics or {},
            })
            
    mean_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0
    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    # Save final text generation evaluation metrics to a separate JSON file
    text_metrics = {
        "mean_bleu_1": mean_bleu1,
        "mean_bleu_4": mean_bleu4,
        "mean_rouge_l": mean_rouge_l,
        "num_evaluated_cases": len(eval_cases),
        "per_case_bleu_1": bleu1_scores,
        "per_case_bleu_4": bleu4_scores,
        "per_case_rouge_l": rouge_l_scores,
    }
    with open(metrics_dir / "text_generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(text_metrics, f, indent=4)
    logger.info(f"Saved text generation evaluation metrics to {metrics_dir / 'text_generation_metrics.json'}")

    write_json(
        metrics_dir / "text_generation_samples.json",
        {
            "samples": generation_samples,
        },
    )
    text_samples_csv = metrics_dir / "text_generation_samples.csv"
    with open(text_samples_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index",
            "case_id",
            "label",
            "predicted_diagnosis",
            "confidence",
            "uncertainty",
            "bleu_1",
            "bleu_4",
            "rouge_l",
            "num_retrieved",
            "mean_retrieval_score",
            "reference_report",
            "generated_report",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in generation_samples:
            retrieved_scores = [item["score"] for item in sample["retrieved_cases"]]
            writer.writerow({
                "sample_index": sample["sample_index"],
                "case_id": sample["case_id"],
                "label": sample["label"],
                "predicted_diagnosis": sample["predicted_diagnosis"],
                "confidence": sample["confidence"],
                "uncertainty": sample["uncertainty"],
                "bleu_1": sample["bleu_1"],
                "bleu_4": sample["bleu_4"],
                "rouge_l": sample["rouge_l"],
                "num_retrieved": len(retrieved_scores),
                "mean_retrieval_score": (
                    sum(retrieved_scores) / len(retrieved_scores) if retrieved_scores else 0.0
                ),
                "reference_report": sample["reference_report"],
                "generated_report": sample["generated_report"],
            })

    # Generate training curves plot
    if args.skip_plot:
        logger.info("Skipping training curves plot generation as requested.")
    else:
        plot_training_metrics(metrics_file, metrics_dir / "training_curves.png")

    # Validate end-to-end RAG Inference
    logger.info("Testing post-training RAG clinical diagnosis on query case...")
    query_image = train_dataset[0]["image"].unsqueeze(0).to(device)
    diag_out = pipeline.diagnose(query_image)
    run_duration_sec = time.perf_counter() - run_timer_start

    inference_report = {
        "case_id": train_cases[0].case_id,
        "label": train_cases[0].label,
        "diagnosis": diag_out.diagnosis,
        "confidence": diag_out.confidence,
        "uncertainty": diag_out.uncertainty,
        "generated_report": diag_out.report,
        "retrieved_cases": retrieval_results_to_dict(diag_out.retrieved_cases),
        "step_metrics": diag_out.step_metrics or {},
    }
    write_json(metrics_dir / "inference_report.json", inference_report)

    artifacts_manifest = {
        "run_started_at": run_started_at,
        "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_duration_sec": run_duration_sec,
        "best_loss": best_loss,
        "dataset_source": dataset_source,
        "num_train_cases": len(train_cases),
        "device": str(device),
        "artifacts": {
            "checkpoint": str(Path("best_model.pt")),
            "best_model_state_dict": str(model_artifacts_dir / "best_model_state_dict.pt"),
            "final_model_state_dict": str(model_artifacts_dir / "final_model_state_dict.pt"),
            "best_training_checkpoint": str(model_artifacts_dir / "best_training_checkpoint.pt"),
            "final_training_checkpoint": str(model_artifacts_dir / "final_training_checkpoint.pt"),
            "retriever_index_base": str(retriever_index_base),
            "model_metadata": str(model_artifacts_dir / "model_metadata.json"),
            "label_map": str(model_artifacts_dir / "label_map.json"),
            "case_index": str(metrics_dir / "case_index.csv"),
            "run_config": str(metrics_dir / "run_config.json"),
            "environment": str(metrics_dir / "environment.json"),
            "training_metrics_csv": str(metrics_file),
            "epoch_metrics_json": str(metrics_dir / "epoch_metrics.json"),
            "text_generation_metrics": str(metrics_dir / "text_generation_metrics.json"),
            "text_generation_samples_json": str(metrics_dir / "text_generation_samples.json"),
            "text_generation_samples_csv": str(metrics_dir / "text_generation_samples.csv"),
            "inference_report": str(metrics_dir / "inference_report.json"),
            "training_curves": str(metrics_dir / "training_curves.png"),
        },
    }
    write_json(metrics_dir / "artifacts_manifest.json", artifacts_manifest)
    write_json(
        metrics_dir / "run_summary.json",
        {
            **artifacts_manifest,
            "final_inference": inference_report,
            "text_generation_metrics": text_metrics,
            "last_epoch": epoch_records[-1] if epoch_records else None,
        },
    )

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
