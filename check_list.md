# TrustMed-RAG / Medical VLM Pipeline - Implementation Checklist

Checklist này phản ánh trạng thái hiện tại của repo: phần nào đã có code và smoke test, phần nào mới là POC, và phần nào cần làm tiếp trước khi coi là bản training/evaluation ổn định.

## 1. Documentation And Research Framing

- [x] Cập nhật `medical_vlm_pipeline.md` sang hướng TrustMed-RAG 10-stage.
- [x] Bổ sung mục tiêu nghiên cứu: retrieval, grounding, knowledge graph, uncertainty, reasoning, controlled generation.
- [x] Đề xuất cấu trúc paper và tên đề tài.
- [x] Tách rõ trạng thái implemented/partial/planned trong tài liệu.
- [ ] Bổ sung bảng mapping paper references trong `paper-refs/` vào related work.

## 2. Dataset And Labeling

- [x] `MedicalCaseDataset` hỗ trợ paired image-report data.
- [x] Hỗ trợ PNG/JPG, DICOM, NIfTI và synthetic fallback.
- [x] Sửa weak-label extraction để hiểu negation như `no pneumothorax`, `negative for pleural effusion`.
- [x] Tạo audit thống kê lớp từ Kaggle output.
- [x] Phát hiện vấn đề lệch lớp: Normal khoảng 69%.
- [x] Phát hiện split leak cũ theo image/report.
- [x] Thêm grouped split theo `report` để tránh train/validation leakage.
- [ ] Chuyển bài toán chính sang binary Normal vs Abnormal hoặc multi-label nếu cần paper mạnh hơn.

## 3. Core Baseline Pipeline

- [x] `MedicalVLMPipeline` chạy end-to-end: image encoder, text encoder, projection, quantizer, retriever, diagnosis head, report generator.
- [x] InfoNCE contrastive loss đã có trong training loop.
- [x] Product Quantization và FAISS retriever đã có fallback an toàn.
- [x] Diagnosis head có MC Dropout uncertainty.
- [x] Text generation có FLAN-T5/fallback template.
- [x] GradCAM/retrieval explanation hooks đã có.
- [x] Smoke test baseline và TrustMed modules pass local.

## 4. TrustMed-RAG 10-Stage Modules

- [x] Stage 1: Data preprocessing/data loader.
- [x] Stage 2: Vision-language encoding.
- [x] Stage 3: Lesion/anatomy grounding module POC.
- [x] Stage 4: Retrieval-augmented evidence search.
- [x] Stage 5: Dynamic medical knowledge graph builder and encoder POC.
- [x] Stage 6: Adaptive multimodal fusion module POC.
- [x] Stage 7: Uncertainty estimator POC.
- [x] Stage 8: Clinical reasoning agent POC.
- [x] Stage 9: Controlled report generation through generator plus uncertainty/recommendation text.
- [x] Stage 10: Offline metrics and smoke tests.
- [ ] Train loop vẫn đang train `MedicalVLMPipeline`; chưa fine-tune full `TrustMedRAGPipeline` end-to-end.
- [ ] Grounding/KG/reasoning hiện là POC heuristic/neural hybrid, chưa được supervised bằng ground-truth region labels.

## 5. Training Improvements

- [x] Thêm class weighting mặc định `balanced`.
- [x] Thêm macro precision/recall/F1 để tránh accuracy đẹp giả trên dataset lệch lớp.
- [x] Thêm per-class metrics và confusion matrix vào CSV/JSON.
- [x] Lưu `label_map.json`, class weights, checkpoints, metadata, run config.
- [x] Thêm `--train-eval-every 0` để tránh full train re-eval mỗi epoch.
- [x] Thêm `--eval-contrastive` optional vì contrastive eval tốn thời gian.
- [x] Thêm `--skip-post-train` để smoke test epoch 1 không mất thời gian build index/sinh báo cáo.
- [ ] Cần chạy lại Kaggle 1 epoch với `--skip-post-train` để đo đúng tốc độ train.
- [ ] Cần chạy lại full 150 epoch sau khi xác nhận GPU/multi-GPU thật sự hoạt động.

## 6. Kaggle And GPU Workflow

- [x] `run_on_kaggle.ipynb` clone/pull source mới từ GitHub.
- [x] Notebook kiểm tra CUDA và cài PyTorch wheel tương thích T4 khi cần.
- [x] Fail fast nếu yêu cầu CUDA nhưng phải fallback CPU.
- [x] Thêm grouped split, class weighting, macro metrics vào notebook.
- [x] Thêm T4x2 config: `MULTI_GPU`, `GPU_IDS`, global batch size.
- [x] Thêm `nvidia-smi` monitor log để kiểm tra GPU utilization.
- [x] Tắt `CUDA_LAUNCH_BLOCKING` mặc định để tránh train chậm khi không debug CUDA.
- [ ] Cần xác nhận từ Kaggle log rằng `multi_gpu.enabled=true`.
- [ ] Cần kiểm tra `nvidia_smi_during_train.csv` để xem cả GPU 0 và GPU 1 có utilization/memory hay không.

## 7. Artifacts And Reproducibility

- [x] Xuất `training_metrics.csv`, `epoch_metrics.json`, `run_config.json`, `environment.json`.
- [x] Xuất best/final model state dict và training checkpoint.
- [x] Xuất retriever index khi không dùng `--skip-post-train`.
- [x] Package artifacts trong notebook.
- [x] Thêm dataset audit local trong `report_kaggle`.
- [ ] Không commit artifact Kaggle lớn vào repo trừ khi cần release kết quả.
- [ ] Nên chuyển PDF/paper refs lớn sang Git LFS hoặc chỉ lưu citation metadata.

## 8. Validation Status

- [x] `python -m compileall src/medical_vlm_pipeline src/train_pipeline.py` pass.
- [x] `pytest -q src/tests` pass: 7 tests.
- [x] Local smoke train CPU pass.
- [ ] Kaggle T4x2 smoke run còn cần xác nhận bằng log mới.
- [ ] Full training report sau grouped split/weak label fix chưa có kết quả cuối.

## 9. Immediate Next Steps

1. Push code/docs hiện tại lên GitHub.
2. Trên Kaggle, pull commit mới nhất.
3. Chạy `EPOCHS=1`, `SKIP_POST_TRAIN=True`, `TRAIN_EVAL_EVERY=0`, `CUDA_LAUNCH_BLOCKING=False`.
4. Kiểm tra `environment.json` và `nvidia_smi_during_train.csv`.
5. Nếu GPU ổn, chạy lại full train 150 epoch.
6. Sau khi có kết quả mới, phân tích macro F1, per-class recall, confusion matrix.
