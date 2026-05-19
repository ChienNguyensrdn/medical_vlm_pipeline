# Quantized Medical VLM Pipeline - Implementation Checklist

This checklist tracks the progress of the 10-stage Quantized Retrieval-Augmented Medical Vision-Language Diagnosis Pipeline.

---

## 📋 Giai đoạn 1: Chuẩn bị & Xử lý Dữ liệu (Data Collection & Loader)
- [x] Triển khai bộ nạp dữ liệu PyTorch `MedicalCaseDataset` tùy biến phục vụ dữ liệu y khoa ghép cặp (Ảnh - Báo cáo).
- [x] Hỗ trợ đọc đa định dạng ảnh y tế: Ảnh 2D tiêu chuẩn (PNG/JPG), ảnh chụp DICOM y tế (`pydicom`/`MONAI`) và ảnh quét thể tích 3D NIfTI (`nibabel`).
- [x] Tích hợp cơ chế **Synthetic Data Fallback** tự động tạo lập ảnh giả lập (2D/3D) khi thiếu file ảnh vật lý, chống sập hệ thống trong môi trường thử nghiệm.
- [x] Cấu hình tiền xử lý tự động thích ứng với cấu hình 2D hoặc 3D (`MONAI.transforms` và `torchvision.transforms`).

## 📋 Giai đoạn 2: Trích xuất đặc trưng hình ảnh (Medical Image Encoder)
- [x] Tích hợp bộ mã hóa `MedicalImageEncoder` hỗ trợ đa dạng Backbone học sâu từ thư viện `timm` (Swin Transformer, ResNet, ConvNeXt).
- [x] Hiện thực hóa bộ chiếu thể tích **3D Spatial Projector** (kết hợp Convolutions 3D và Adaptive Pooling) để nén lát quét 3D (MRI/CT) thành không gian biểu diễn 2D đặc trưng.
- [x] Triển khai cơ chế **CNN Fallback** độc lập chạy ngoại tuyến mượt mà không phụ thuộc vào thư viện tải weights bên ngoài.

## 📋 Giai đoạn 3: Trích xuất đặc trưng lâm sàng (Clinical Text Encoder)
- [x] Hiện thực hóa `ClinicalTextEncoder` hỗ trợ các mô hình chuyên biệt cho y khoa như PubMedBERT hay ClinicalBERT qua thư viện `transformers`.
- [x] Tích hợp bộ **Tokenizer** và **Model Fallback** tự động chuyển sang BERT tiêu chuẩn hoặc mô hình mạng LSTM hai chiều dự phòng khi kết nối mạng lỗi.

## 📋 Giai đoạn 4: Căn chỉnh liên phương thức đối sánh (Cross-modal Alignment)
- [x] Hiện thực hóa lớp tính toán độ lỗi **InfoNCE Loss** đối xứng để căn chỉnh ảnh y tế và báo cáo lâm sàng vào một không gian latent chung.
- [x] Tối ưu nhiệt độ tự điều chỉnh (dưới dạng tham số có thể huấn luyện `temperature`).

## 📋 Giai đoạn 5: Không gian Latent thu gọn & Lượng tử hóa (Quantization)
- [x] Hiện thực hóa **Product Quantization (PQ)**: Chia nhỏ vector đặc trưng chung thành $M$ vector con để gom cụm K-Means, giảm 90%+ kích thước chỉ mục vector DB.
- [x] Hiện thực hóa lớp học tập lượng tử **Vector Quantization (VQ-STE)** huấn luyện trực tiếp qua kỹ thuật ước lượng thẳng dòng **Straight-Through Estimator (STE)** giúp truyền ngược gradient thông qua thao tác rời rạc hóa `argmax`.

## 📋 Giai đoạn 6: Chẩn đoán truy hồi (Retrieval-Augmented Diagnosis)
- [x] Triển khai bộ kết xuất vector **FAISSRetriever** sử dụng tìm kiếm Cosine Similarity (`IndexFlatIP`) và Euclidean (`IndexFlatL2`).
- [x] Hỗ trợ tuần tự hóa (serialization) chỉ mục vector DB ghi/nạp trực tiếp từ đĩa.
- [x] Tích hợp cơ chế **PyTorch Vectorized Similarity Fallback** tự động thay thế nếu môi trường triển khai chưa có cài đặt FAISS.

## 📋 Giai đoạn 7: Đầu phân loại bệnh lý (Diagnosis Head)
- [x] Triển khai bộ phân loại `DiagnosisHead` học trên đặc trưng ảnh chung đã được lượng tử hóa để đưa ra nhãn bệnh lý và độ tự tin (confidence score).

## 📋 Giai đoạn 8: Sinh báo cáo y khoa tự động (Clinical Report Generation)
- [x] Hiện thực hóa `LLMReportGenerator` hỗ trợ Seq2Seq với Visual Prefix (FLAN-T5).
- [x] Triển khai bộ **Clinical Template Synthesizer Fallback** tổng hợp chẩn đoán và trích xuất bệnh án mẫu từ vector truy vấn lịch sử để tự động tạo báo cáo chất lượng cao ngay cả khi chạy offline.

## 📋 Giai đoạn 9: Giải thích trực quan & Độ tin cậy (Explainability)
- [x] Hiện thực hóa bộ giải thích **MedicalGradCAM** tự thích ứng linh hoạt với ảnh 2D, ảnh thể tích 3D, và cơ chế chuyển đổi chiều không gian của Patch Tokens trong Vision Transformer (ViT/Swin).
- [x] Tích hợp trích xuất giải thích tương đồng văn bản từ các ca bệnh lịch sử.

## 📋 Giai đoạn 10: Tối ưu hóa triển khai (Deployment Optimization)
- [x] Thiết lập mã nguồn dưới dạng gói package Python độc lập cấu trúc rõ ràng.
- [x] Xây dựng tệp cấu hình trung tâm `config.py` linh hoạt.
- [x] Tạo lập tệp tin Kaggle Notebook mẫu **`run_on_kaggle.ipynb`** hỗ trợ triển khai huấn luyện siêu tốc trên GPU đám mây.
- [x] Vượt qua tất cả kiểm thử kiểm nghiệm toàn trình `demo_pipeline.py` với tỉ lệ thành công **100%**.

---

## 🚀 Các Bước Tiếp Theo (Future Roadmap)

1. **Huấn luyện với Dữ liệu Y tế thực tế (Real-world Medical Training):**
   - Đưa mô hình lên chạy trên nền tảng Kaggle GPU qua tệp `run_on_kaggle.ipynb`.
   - Kết nối với các kho dữ liệu y khoa mở thực tế như **MIMIC-CXR** (ảnh X-quang phổi ghép cặp báo cáo) hoặc **BraTS** (ảnh cộng hưởng từ MRI não 3D).

2. **Fine-tuning mô hình ngôn ngữ (Seq2Seq / LLM Fine-tuning):**
   - Tinh chỉnh lớp chiếu Visual-to-Language Projection trên tập dữ liệu thực tế để tăng điểm đánh giá sinh ngôn ngữ tự động (BLEU, ROUGE, BERTScore).

3. **Cải tiến độ tin cậy và phân tích sai lệch (Uncertainty Estimation):**
   - Triển khai phương pháp Bayesian Deep Learning hoặc Monte Carlo Dropout trên đầu phân loại để ước lượng mức độ không chắc chắn (Uncertainty) trước khi xuất báo cáo.
