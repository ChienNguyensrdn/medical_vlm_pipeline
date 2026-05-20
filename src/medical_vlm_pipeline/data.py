"""Dataset interfaces for paired medical images and reports."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MedicalCase:
    case_id: str
    image_path: Path
    report_text: str
    label: str | None = None
    modality: str | None = None


class MedicalCaseDataset(Dataset):
    """PyTorch dataset for paired medical images and clinical reports."""

    def __init__(
        self,
        cases: Iterable[MedicalCase],
        transform: Callable[[Any], Tensor] | None = None,
        tokenizer: Any | None = None,
        max_length: int = 128,
        image_size: int = 224,
        volume_depth: int | None = None,
    ) -> None:
        """
        Args:
            cases: Iterable of MedicalCase objects.
            transform: Optional image transform function/object.
            tokenizer: Optional text tokenizer (e.g., PubMedBERT tokenizer).
            max_length: Maximum token length for report text.
            image_size: Target 2D image height/width.
            volume_depth: Depth dimension if using 3D volumes (MRI/CT).
        """
        self.cases = list(cases)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.volume_depth = volume_depth

    def __len__(self) -> int:
        return len(self.cases)

    def _load_image(self, path: Path) -> Tensor:
        """Loads an image with support for standard files, fallback, or mock generation."""
        if not path.exists():
            # Robust fallback: Generate synthetic medical image tensor
            logger.warning(f"Image path {path} does not exist. Generating synthetic data.")
            if self.volume_depth is not None:
                # 3D Volume (channels=1, depth, H, W)
                return torch.randn(1, self.volume_depth, self.image_size, self.image_size)
            else:
                # 2D Image (channels=3, H, W)
                return torch.randn(3, self.image_size, self.image_size)

        try:
            # Check file extension
            ext = path.suffix.lower()
            if ext in [".dcm", ".dicom"]:
                import pydicom
                from monai.transforms import LoadImage
                # Pydicom DICOM file
                loader = LoadImage(image_only=True)
                img_data = loader(str(path))
                # Convert to torch tensor
                img_tensor = torch.as_tensor(img_data, dtype=torch.float32)
                # Ensure correct dimensions (C, H, W) or (C, D, H, W)
                if len(img_tensor.shape) == 2:
                    img_tensor = img_tensor.unsqueeze(0)  # C=1
                elif len(img_tensor.shape) == 3 and self.volume_depth is not None:
                    img_tensor = img_tensor.unsqueeze(0)  # C=1, D, H, W
                return img_tensor

            elif ext in [".nii", ".nii.gz"]:
                import nibabel as nib
                # NIfTI MRI volume
                nifti = nib.load(str(path))
                img_data = nifti.get_fdata()
                img_tensor = torch.as_tensor(img_data, dtype=torch.float32)
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)  # C=1, D, H, W
                return img_tensor

            else:
                # Standard 2D image (PNG, JPEG)
                img = Image.open(path).convert("RGB")
                if self.transform:
                    return self.transform(img)
                else:
                    # Default normalization/resize transform
                    import torchvision.transforms as T
                    default_transform = T.Compose([
                        T.Resize((self.image_size, self.image_size)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    return default_transform(img)

        except Exception as e:
            logger.error(f"Error loading image {path}: {e}. Returning synthetic tensor.")
            if self.volume_depth is not None:
                return torch.randn(1, self.volume_depth, self.image_size, self.image_size)
            else:
                return torch.randn(3, self.image_size, self.image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]
        image_tensor = self._load_image(case.image_path)

        item = {
            "case_id": case.case_id,
            "image": image_tensor,
            "report_text": case.report_text,
            "label": case.label if case.label is not None else "",
            "modality": case.modality if case.modality is not None else "",
        }

        if self.tokenizer is not None:
            # Tokenize clinical report text
            tokens = self.tokenizer(
                case.report_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Remove the batch dimension added by return_tensors="pt"
            item["input_ids"] = tokens["input_ids"].squeeze(0)
            if "attention_mask" in tokens:
                item["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return item


def infer_report_label(report_text: str) -> str:
    """Infer a weak pathology label from IU Chest X-ray report text."""
    text = report_text.lower()
    rules = [
        ("Pneumothorax", ("pneumothorax",)),
        ("Pleural Effusion", ("pleural effusion", "effusions")),
        ("Pneumonia", ("pneumonia", "consolidation", "infiltrate", "opacity")),
        ("Cardiomegaly", ("cardiomegaly", "enlarged heart", "heart size is enlarged")),
        ("Atelectasis", ("atelectasis",)),
        ("Edema/Congestion", ("edema", "congestion", "vascular prominence", "pulmonary vascular")),
        ("COPD/Hyperinflation", ("copd", "hyperinflation", "emphysema")),
        ("Nodule/Mass", ("nodule", "mass", "lesion")),
        ("Normal", ("normal chest", "no acute", "clear lungs", "lungs are clear")),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "Other"


def choose_label_column(columns: Iterable[str], requested: str) -> str | None:
    """Choose a usable label column from a cleaned IU CSV."""
    columns_set = set(columns)
    if requested != "auto":
        return requested if requested in columns_set else None

    preferred = (
        "diagnosis",
        "diagnoses",
        "finding",
        "findings",
        "impression",
        "disease",
        "pathology",
        "label",
        "class",
        "projection",
    )
    for candidate in preferred:
        if candidate in columns_set:
            return candidate
    return None


def load_iu_chest_xray_cases(
    csv_path: str | Path,
    images_dir: str | Path,
    label_column: str = "projection",
    derive_labels_from_report: bool = False,
    min_class_count: int = 2,
) -> list[MedicalCase]:
    """Helper to parse IU Chest X-ray cleaned_dataset.csv and load paired cases.

    Args:
        csv_path: Path to cleaned_dataset.csv.
        images_dir: Path to the specific resized folder (e.g. 'resized_images/256').
        label_column: CSV column used as the classification target, or "auto".
        derive_labels_from_report: Build weak pathology labels from report text.
        min_class_count: Rare labels below this count are grouped into "Other".
    """
    import pandas as pd
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    if not csv_path.exists():
        logger.error(f"Dataset CSV not found at: {csv_path}")
        return []

    logger.info(f"Loading IU Chest X-ray cases from: {csv_path}")
    df = pd.read_csv(csv_path)

    selected_label_column = choose_label_column(df.columns, label_column)
    if derive_labels_from_report:
        logger.info("Using weak pathology labels derived from org_caption text.")
    elif selected_label_column is not None:
        logger.info(f"Using CSV column '{selected_label_column}' as classification label.")
    else:
        logger.warning(
            "Requested label column '%s' was not found. Falling back to projection labels.",
            label_column,
        )

    cases = []
    # Verify required columns are present
    required_cols = {"image_id", "org_caption"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            f"CSV columns {df.columns} do not contain expected {required_cols}. "
            "Loading empty case list."
        )
        return []

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        image_path = images_dir / image_id

        # Clean report text
        caption = str(row["org_caption"]) if not pd.isna(row["org_caption"]) else ""
        projection = (
            str(row["projection"]) if "projection" in df.columns and not pd.isna(row["projection"])
            else "Frontal"
        )
        if derive_labels_from_report:
            label = infer_report_label(caption)
        elif selected_label_column is not None and not pd.isna(row[selected_label_column]):
            label = str(row[selected_label_column]).strip() or projection
        else:
            label = projection

        case = MedicalCase(
            case_id=image_id.split(".")[0],
            image_path=image_path,
            report_text=caption,
            label=label,
            modality="Chest X-ray",
        )
        cases.append(case)

    if min_class_count > 1:
        label_counts: dict[str, int] = {}
        for case in cases:
            label_counts[case.label or "Other"] = label_counts.get(case.label or "Other", 0) + 1
        if any(count < min_class_count for count in label_counts.values()):
            cases = [
                MedicalCase(
                    case_id=case.case_id,
                    image_path=case.image_path,
                    report_text=case.report_text,
                    label=case.label if label_counts.get(case.label or "Other", 0) >= min_class_count else "Other",
                    modality=case.modality,
                )
                for case in cases
            ]

    logger.info(f"Successfully loaded {len(cases)} cases from IU Chest X-ray CSV.")
    return cases
