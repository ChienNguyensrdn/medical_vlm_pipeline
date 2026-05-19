"""End-to-end validation and demonstration script for the Quantized Medical VLM Pipeline."""

import os
import sys
import logging
from pathlib import Path

# Ensure package directory is on path
sys.path.append(str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from medical_vlm_pipeline import PipelineConfig, MedicalVLMPipeline
from medical_vlm_pipeline.data import MedicalCase, MedicalCaseDataset
from medical_vlm_pipeline.alignment import infonce_loss
from medical_vlm_pipeline.explainability import retrieval_explanation, MedicalGradCAM

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(temp_dir: Path) -> tuple[list[MedicalCase], MedicalCase]:
    """Generates synthetic medical cases with dummy images and clinical report templates."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Reference database cases (mixture of Meningioma, Glioma, Healthy scans)
    clinical_reports = [
        "Contrast-enhanced MRI scan showing a well-defined extra-axial dural-based mass in the left frontal region, highly suggestive of meningioma. Mild surrounding vasogenic edema is noted.",
        "T2-weighted MRI scan reveals an infiltrative high-signal intensity lesion within the right temporal lobe, indicating high-grade glioma. Significant mass effect on adjacent sulci.",
        "Brain MRI scan exhibits normal ventricles, sulci, and cisterns. No focal signal abnormalities or abnormal contrast enhancement is observed. Normal study.",
        "Dural-based extra-axial tumor in the cerebellopontine angle showing uniform post-contrast enhancement. Suggests meningioma.",
        "Diffuse expansion of the brainstem with hyperintense signal on FLAIR sequence, representative of pontine glioma.",
        "No evidence of acute intracranial hemorrhage, mass effect, or midline shift. Ventricular system is unremarkable.",
    ]

    labels = ["Meningioma", "Glioma", "Healthy", "Meningioma", "Glioma", "Healthy"]

    reference_cases = []
    for i in range(12):
        case_id = f"REF-{1000 + i}"
        img_path = temp_dir / f"{case_id}.png"

        # Note: data.py will automatically generate synthetic tensors if the files don't exist
        # But we create a dummy file to test standard image load pathway
        try:
            from PIL import Image
            import numpy as np
            img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img_np).save(img_path)
        except Exception:
            pass  # Fall back to dataloader-level tensor generation

        report_text = clinical_reports[i % len(clinical_reports)]
        label = labels[i % len(labels)]

        reference_cases.append(
            MedicalCase(
                case_id=case_id,
                image_path=img_path,
                report_text=report_text,
                label=label,
                modality="MRI"
            )
        )

    # Query Case (to diagnose)
    query_id = "QUERY-999"
    query_img_path = temp_dir / f"{query_id}.png"
    try:
        from PIL import Image
        import numpy as np
        img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img_np).save(query_img_path)
    except Exception:
        pass

    query_case = MedicalCase(
        case_id=query_id,
        image_path=query_img_path,
        report_text="Enhancing mass along the right parietal convex convexity, typical for meningioma lesion.",
        label="Meningioma",
        modality="MRI"
    )

    return reference_cases, query_case


def main() -> None:
    logger.info("=" * 60)
    logger.info("STARTING QUANTIZED RETRIEVAL-AUGMENTED VLM PIPELINE DEMO")
    logger.info("=" * 60)

    # Setup directories
    workspace_dir = Path(__file__).parent.parent
    temp_data_dir = workspace_dir / "data" / "images"

    # 1. Generate Dummy Data
    reference_cases, query_case = generate_synthetic_data(temp_data_dir)
    logger.info(f"Generated {len(reference_cases)} reference cases and 1 query case.")

    # 2. Configure Pipeline
    config = PipelineConfig()
    config.encoders.projection_dim = 128  # Faster for demo
    config.retrieval.top_k = 2
    config.retrieval.quantized = True  # Enable Product Quantization

    # 3. Instantiate Pipeline Orchestrator
    logger.info("Instantiating End-to-End Multimodal Pipeline...")
    pipeline = MedicalVLMPipeline(config, num_classes=3, class_names=["Healthy", "Meningioma", "Glioma"])

    # 4. Prepare Reference Dataloader
    dataset = MedicalCaseDataset(reference_cases, image_size=224)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 5. Build Index Database (Quantization + FAISS Vector DB)
    logger.info("Building Aligned and Quantized Reference Vector Index...")
    pipeline.build_vector_index(loader)

    # 6. Execute Diagnosis Inference on Query Scan
    logger.info("Loading Query Scan & Running RAG Medical Inference...")
    query_dataset = MedicalCaseDataset([query_case], image_size=224)
    query_item = query_dataset[0]
    query_image = query_item["image"].unsqueeze(0)  # Shape (1, C, H, W)

    # Execute orchestrator pipeline
    diagnosis_output = pipeline.diagnose(query_image)

    # 7. Print Multimodal Diagnostic Outputs
    logger.info("=" * 60)
    logger.info("               DIAGNOSIS PIPELINE OUTPUTS               ")
    logger.info("=" * 60)
    logger.info(f"Query Case ID:      {query_case.case_id}")
    logger.info(f"Predicted Diagnosis: {diagnosis_output.diagnosis}")
    logger.info(f"Confidence Score:    {diagnosis_output.confidence * 100:.2f}%")
    logger.info("-" * 60)

    logger.info(f"Retrieved Similar Precedent Cases:")
    for idx, case in enumerate(diagnosis_output.retrieved_cases):
        logger.info(f"  [{idx + 1}] Case ID: {case.case_id} | Similarity: {case.score * 100:.2f}%")
        logger.info(f"      Report:  {case.report_text[:120]}...")
        logger.info(f"      Label:   {case.label}")

    logger.info("-" * 60)
    logger.info(f"Generated Clinical Report:")
    logger.info(diagnosis_output.report)
    logger.info("=" * 60)

    # 8. Contrastive InfoNCE Loss Validation (Stage 4 Alignment)
    logger.info("Validating Cross-modal Contrastive Loss calculation...")
    # Mock aligned visual and textual batch features
    batch_size = 4
    dim = config.encoders.projection_dim
    image_embeds = torch.randn(batch_size, dim)
    text_embeds = torch.randn(batch_size, dim)

    loss = infonce_loss(image_embeds, text_embeds, temperature=config.training.temperature)
    logger.info(f"InfoNCE Loss on Batch of size {batch_size}: {loss.item():.4f}")

    # 9. Explainability Hook Validation (Stage 9 GradCAM)
    logger.info("Running Visual Explainability Hook (GradCAM) on Image Encoder...")
    # Target final convolution layer in fallback baseline (or Swin if available)
    if hasattr(pipeline.image_encoder, "fallback_cnn"):
        target_layer = pipeline.image_encoder.fallback_cnn[7]  # Conv2d block
        gradcam = MedicalGradCAM(pipeline, target_layer)

        heatmap = gradcam.generate_heatmap(query_image, class_idx=1)
        logger.info(f"GradCAM Heatmap Generated Successfully. Output Tensor Shape: {heatmap.shape}")
        gradcam.remove_hooks()
    else:
        logger.info("Custom target layer not found for GradCAM in image encoder backbone.")

    logger.info("=" * 60)
    logger.info("PIPELINE DEMONSTRATION RUN COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
