# Kaggle Run Checklist

Use this checklist when moving the pipeline from local synthetic smoke test to Kaggle GPU training.

## 1. Kaggle Runtime

- Accelerator: enable **GPU T4** or stronger.
- Internet: enable if you want Kaggle to download pretrained weights from Hugging Face/timm.
- Python: use the Kaggle default Python 3.10/3.11 runtime.

## 2. Dataset

Expected default dataset path:

```text
/kaggle/input/iu-chest-x-rays-cleaned
  cleaned_dataset.csv
  resized_images/256/
```

The loader expects these CSV columns:

```text
image_id
org_caption
projection
```

If the dataset path differs, run with:

```bash
python src/train_pipeline.py \
  --epochs 10 \
  --batch-size 8 \
  --device auto \
  --dataset-dir /kaggle/input/YOUR_DATASET_DIR
```

## 3. Dependencies

Install dependencies from the project root:

```bash
pip install -r src/requirements.txt
```

If Kaggle has dependency conflicts, install the minimum training set first:

```bash
pip install torch torchvision timm transformers monai pydicom nibabel SimpleITK \
  scikit-learn faiss-cpu pandas matplotlib rich tqdm nltk rouge-score
```

## 4. Model Download Notes

The default config now points to concrete model IDs:

```text
image_encoder = swin_tiny_patch4_window7_224
text_encoder = microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

If Kaggle internet is disabled, the code will fall back to:

- custom CNN image encoder
- cached `bert-base-uncased` or fallback text model
- clinical template report generator if FLAN-T5 cannot load

For stronger real experiments, keep Kaggle internet enabled or attach a Hugging Face model cache as a Kaggle dataset.

## 5. Output Artifacts To Check

After training, verify:

```text
best_model.pt
report_kaggle/training_metrics.csv
report_kaggle/text_generation_metrics.json
report_kaggle/training_curves.png
```

The local smoke test may use `--skip-plot`; on Kaggle, omit it to generate `training_curves.png`.

## 6. Known Local-vs-Kaggle Difference

On this Mac local run, PyTorch reported:

```text
cuda = False
mps = False
```

So the local test ran on CPU with synthetic fallback data. Kaggle GPU should report `device: cuda` and use the real IU Chest X-ray dataset when the dataset path is mounted correctly.
