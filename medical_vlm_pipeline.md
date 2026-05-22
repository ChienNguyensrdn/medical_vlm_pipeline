# TrustMed-RAG: Uncertainty-aware Retrieval-Augmented Knowledge-Grounded Medical Vision-Language Agent

## 1. Mục tiêu nghiên cứu

Mục tiêu là xây dựng một pipeline AI y tế có khả năng hiểu ảnh y khoa, truy xuất các ca bệnh tương tự, kết hợp tri thức y khoa, ước lượng độ không chắc chắn, reasoning trước khi sinh báo cáo, và đánh giá mức độ đáng tin cậy của báo cáo sinh ra.

Pipeline này phù hợp cho các bài toán:

- Radiology Report Generation
- Medical Visual Question Answering
- Medical Image-Text Retrieval
- Clinical Decision Support
- Longitudinal Disease Progression Analysis
- Trustworthy Medical Vision-Language AI

---

## 2. Tổng quan pipeline

```text
Medical Image / Clinical Context
        ↓
Data Preprocessing
        ↓
Vision-Language Encoding
        ↓
Lesion / Anatomy Grounding
        ↓
Retrieval-Augmented Evidence Search
        ↓
Dynamic Medical Knowledge Graph Construction
        ↓
Adaptive Multimodal Fusion
        ↓
Uncertainty Estimation
        ↓
Clinical Reasoning Agent
        ↓
Controlled Report Generation
        ↓
Image-aware and Clinical Evaluation
        ↓
Human Feedback / Model Update
```

---

## 3. Input của hệ thống

### 3.1. Dữ liệu chính

```text
I  : Medical image, ví dụ CXR, CT, MRI
C  : Clinical context, ví dụ triệu chứng, tiền sử, chỉ định chụp
R0 : Prior report nếu có
L  : Pathology labels nếu có
E  : EHR, lab test, medication history nếu có
```

### 3.2. Ví dụ input

```text
Image: Chest X-ray
Clinical note: Fever, cough, shortness of breath
Prior report: No acute cardiopulmonary abnormality
Pathology label: Pneumonia uncertain
```

### 3.3. Output mong muốn

```text
Findings:
- Có/không có bất thường nào.
- Bất thường nằm ở vùng giải phẫu nào.
- Mức độ chắc chắn.

Impression:
- Chẩn đoán chính hoặc khả năng cao nhất.
- Các chẩn đoán phân biệt nếu cần.

Evidence:
- Vùng ảnh hỗ trợ kết luận.
- Ca bệnh tương tự đã retrieve.
- Tri thức y khoa liên quan.

Recommendation:
- Theo dõi, chụp thêm, xét nghiệm, hoặc cần bác sĩ xác nhận.
```

---

# 4. Stage 1 — Data Preprocessing

## 4.1. Mục tiêu

Chuẩn hóa dữ liệu đầu vào và giảm nhiễu để tránh mô hình học các shortcut không liên quan đến bệnh lý.

## 4.2. Các bước xử lý ảnh

```text
Input image
    ↓
DICOM / PNG / JPG loading
    ↓
Intensity normalization
    ↓
Resize / crop
    ↓
Artifact checking
    ↓
Organ / lung segmentation
    ↓
Anatomy region extraction
```

## 4.3. Các kỹ thuật có thể dùng

### Với chest X-ray

```text
- Histogram normalization
- CLAHE
- Lung field segmentation
- Bone suppression nếu cần
- Remove text marker / device artifact nếu có
```

### Với CT / MRI

```text
- HU normalization đối với CT
- N4 bias field correction đối với MRI
- Slice selection
- 3D volume resampling
- Organ segmentation
```

## 4.4. Output

```text
I_clean      : ảnh đã chuẩn hóa
M_anatomy    : mask vùng giải phẫu
M_artifact   : mask artifact nếu phát hiện được
I_region     : các vùng ảnh quan trọng
```

## 4.5. Ý nghĩa nghiên cứu

Medical VLM có thể học nhầm shortcut như tube, drain, marker, hoặc device thay vì tổn thương thật. Do đó, preprocessing không chỉ để làm sạch ảnh mà còn để giảm bias và tăng độ tin cậy.

---

# 5. Stage 2 — Vision-Language Encoding

## 5.1. Mục tiêu

Biến ảnh y khoa và văn bản lâm sàng thành biểu diễn embedding trong không gian chung.

```text
Image → Vision Encoder → Visual Tokens
Text  → Text Encoder   → Text Tokens
```

## 5.2. Vision encoder

Có thể dùng:

```text
- ResNet
- DenseNet
- ViT
- Swin Transformer
- ConvNeXt
- MedViT
- 3D ViT cho CT/MRI
- Slice-based Transformer cho 3D medical imaging
```

## 5.3. Text encoder

Có thể dùng:

```text
- BioBERT
- ClinicalBERT
- PubMedBERT
- SapBERT
- BlueBERT
- Medical LLM encoder
```

## 5.4. Vision-language backbone

Có thể dùng:

```text
- CLIP / MedCLIP
- BioViL
- BLIP / BLIP-2 fine-tuned medical
- LLaVA-Med
- Med-Flamingo
- MedGemma
```

## 5.5. Công thức biểu diễn

```text
Z_v = f_v(I_clean)
Z_t = f_t(C, R0)
Z_mm = Align(Z_v, Z_t)
```

Trong đó:

```text
Z_v  : visual embedding
Z_t  : text embedding
Z_mm : multimodal aligned embedding
```

## 5.6. Hạn chế của stage này

Encoder thông thường chỉ học global alignment giữa ảnh và báo cáo. Điều này chưa đủ vì ảnh y khoa cần fine-grained alignment giữa từng vùng giải phẫu và từng mô tả bệnh lý.

---

# 6. Stage 3 — Lesion / Anatomy Grounding

## 6.1. Mục tiêu

Liên kết từng cụm từ trong báo cáo với vùng ảnh tương ứng.

```text
Text phrase ↔ Image region
```

## 6.2. Ví dụ

```text
"right lower lobe opacity" ↔ vùng thùy dưới phổi phải
"pleural effusion" ↔ vùng màng phổi
"cardiomegaly" ↔ cardiac silhouette
"lung nodule" ↔ vùng nodule nghi ngờ
```

## 6.3. Pipeline grounding

```text
Visual tokens
    ↓
Anatomy-aware region proposal
    ↓
Text phrase extraction
    ↓
Phrase-region matching
    ↓
Grounding score estimation
```

## 6.4. Kỹ thuật có thể dùng

```text
- Cross-attention map
- Grad-CAM
- Score-CAM
- Weakly supervised localization
- Grounding DINO fine-tuned medical
- SAM / MedSAM for segmentation
- Anatomy-aware attention
```

## 6.5. Output

```text
A = {a1, a2, ..., an}       : anatomy regions
P = {p1, p2, ..., pm}       : pathology phrases
G_ground = Match(P, A)      : phrase-region grounding map
S_ground                  : grounding confidence score
```

## 6.6. Ý nghĩa

Grounding giúp trả lời câu hỏi:

```text
Report này có thật sự dựa vào ảnh không?
Hay mô hình chỉ sinh câu nghe có vẻ hợp lý?
```

Đây là phần rất quan trọng để giảm hallucination.

---

# 7. Stage 4 — Retrieval-Augmented Evidence Search

## 7.1. Mục tiêu

Truy xuất các ca bệnh tương tự để hỗ trợ reasoning và tăng explainability.

```text
Current case → Retrieve similar cases → Evidence set
```

## 7.2. Cơ sở dữ liệu retrieval

Vector database nên lưu:

```text
- image embedding
- report embedding
- pathology labels
- anatomy regions
- diagnosis
- follow-up outcome
- treatment response nếu có
- uncertainty score nếu có
```

## 7.3. Vector database có thể dùng

```text
- FAISS
- Milvus
- ChromaDB
- Weaviate
- Qdrant
```

## 7.4. Retrieval score

Không nên chỉ dùng cosine similarity. Nên kết hợp nhiều thành phần:

```text
Score(q, c_i) =
    α · Sim_visual(q, c_i)
  + β · Sim_text(q, c_i)
  + γ · Sim_pathology(q, c_i)
  + δ · Sim_anatomy(q, c_i)
  + η · Sim_clinical(q, c_i)
  - λ · Uncertainty(c_i)
```

Trong đó:

```text
q   : current query case
c_i : candidate retrieved case
α, β, γ, δ, η, λ : trọng số
```

## 7.5. Các kiểu retrieval

### 7.5.1. Visual retrieval

Tìm ảnh giống ảnh hiện tại.

```text
CXR hiện tại → CXR tương tự
```

### 7.5.2. Textual retrieval

Tìm báo cáo hoặc mô tả bệnh án tương tự.

```text
clinical note → prior report tương tự
```

### 7.5.3. Pathology-aware retrieval

Tìm các case có bệnh lý tương tự.

```text
opacity + fever → pneumonia cases
```

### 7.5.4. Anatomy-aware retrieval

Tìm các case cùng vùng giải phẫu.

```text
right lower lobe lesion → right lower lobe cases
```

### 7.5.5. Temporal retrieval

Tìm các chuỗi bệnh tiến triển tương tự.

```text
CXR_t1 → CXR_t2 → outcome
```

### 7.5.6. Counterfactual retrieval

Tìm các ca gần giống nhưng kết quả khác.

```text
case giống hiện tại nhưng không phải pneumonia
```

## 7.6. Output

```text
R = {r1, r2, ..., rk}
```

Mỗi retrieved case gồm:

```text
r_i = {
    image,
    report,
    label,
    diagnosis,
    anatomy,
    outcome,
    similarity_score
}
```

## 7.7. Ý nghĩa nghiên cứu

Retrieval giúp mô hình không chỉ dựa vào tham số đã học, mà còn có bằng chứng từ các ca bệnh thực tế. Đây là hướng gần với cách bác sĩ so sánh với các ca từng gặp.

---

# 8. Stage 5 — Dynamic Medical Knowledge Graph Construction

## 8.1. Mục tiêu

Xây dựng graph tri thức riêng cho từng case, kết hợp tri thức y khoa nền và bằng chứng retrieve được.

```text
Base Medical KG
+ current findings
+ retrieved evidence
+ clinical context
→ case-specific dynamic KG
```

## 8.2. Nguồn tri thức

Có thể dùng:

```text
- RadGraph
- UMLS
- SNOMED CT
- MeSH
- ICD
- CheXpert labels
- Radiology ontology
```

## 8.3. Node trong graph

```text
- Anatomy node
- Pathology node
- Symptom node
- Imaging finding node
- Diagnosis node
- Treatment node
- Outcome node
- Uncertainty node
```

## 8.4. Edge trong graph

```text
- located_in
- associated_with
- indicates
- causes
- excludes
- suggests
- progresses_to
- treated_by
- similar_to
```

## 8.5. Ví dụ graph

```text
Opacity
    → located_in → Right Lower Lobe
    → associated_with → Pneumonia
    → supported_by → Retrieved Case 03
    → confidence → 0.78

Pleural Effusion
    → located_in → Pleural Space
    → indicates → Fluid Accumulation
    → confidence → 0.64
```

## 8.6. Công thức graph

```text
G_case = (V_case, E_case)
```

Trong đó:

```text
V_case = V_base ∪ V_detected ∪ V_retrieved ∪ V_context
E_case = E_base ∪ E_inferred ∪ E_retrieved
```

## 8.7. Graph encoder

Có thể dùng:

```text
- GCN
- GAT
- R-GCN
- Graph Transformer
- Heterogeneous Graph Transformer
- Temporal Graph Neural Network
```

## 8.8. Output

```text
Z_g = GraphEncoder(G_case)
```

## 8.9. Ý nghĩa nghiên cứu

Graph giúp mô hình có cấu trúc reasoning rõ ràng hơn thay vì chỉ dựa vào embedding black-box.

---

# 9. Stage 6 — Adaptive Multimodal Fusion

## 9.1. Mục tiêu

Hợp nhất các nguồn thông tin:

```text
visual embedding
text embedding
retrieved evidence embedding
knowledge graph embedding
```

## 9.2. Input

```text
Z_v : visual features
Z_t : text features
Z_r : retrieval evidence features
Z_g : graph features
```

## 9.3. Fusion đơn giản

```text
Z_f = concat(Z_v, Z_t, Z_r, Z_g)
```

Nhưng cách này yếu vì không học được quan hệ chéo giữa vùng ảnh, text, graph và retrieved cases.

## 9.4. Fusion đề xuất

```text
Z_vg = CrossAttention(Q = Z_v, K = Z_g, V = Z_g)
Z_vr = CrossAttention(Q = Z_v, K = Z_r, V = Z_r)
Z_gt = CrossAttention(Q = Z_g, K = Z_t, V = Z_t)
```

Sau đó:

```text
w_v, w_t, w_r, w_g = AdaptiveGate(Z_v, Z_t, Z_r, Z_g)

Z_f = w_v Z_v + w_t Z_t + w_r Z_r + w_g Z_g
```

## 9.5. Adaptive gate

```text
w = softmax(MLP([Z_v, Z_t, Z_r, Z_g]))
```

## 9.6. Output

```text
Z_f : fused multimodal representation
```

## 9.7. Ý nghĩa

Adaptive fusion cho phép model tự quyết định nguồn nào đáng tin hơn trong từng trường hợp.

Ví dụ:

```text
Ảnh rõ → tăng trọng số visual
Ảnh mờ → tăng trọng số retrieval và knowledge graph
Clinical note mạnh → tăng trọng số text
Uncertainty cao → giảm trọng số nguồn không đáng tin
```

---

# 10. Stage 7 — Uncertainty Estimation

## 10.1. Mục tiêu

Ước lượng mức độ không chắc chắn của mô hình trước khi sinh báo cáo.

```text
Z_f → uncertainty score
```

## 10.2. Các loại uncertainty

### Aleatoric uncertainty

Do dữ liệu nhiễu hoặc không rõ ràng.

Ví dụ:

```text
Ảnh mờ, low contrast, lesion nhỏ, overlapping structure
```

### Epistemic uncertainty

Do mô hình thiếu kiến thức hoặc gặp domain lạ.

Ví dụ:

```text
Bệnh hiếm, scanner khác, population khác, pattern chưa thấy trong training
```

## 10.3. Kỹ thuật ước lượng

```text
- Monte Carlo Dropout
- Deep Ensemble
- Evidential Deep Learning
- Bayesian Neural Network
- Dempster-Shafer Theory
- Conformal Prediction
- Temperature Scaling
```

## 10.4. Output

```text
U_global    : uncertainty toàn report
U_region    : uncertainty theo vùng ảnh
U_finding   : uncertainty theo từng finding
U_retrieval : uncertainty của retrieved evidence
U_graph     : uncertainty của graph reasoning
```

## 10.5. Ví dụ output

```text
Finding: right lower lobe opacity
Confidence: 0.76
Uncertainty: medium
Reason: image quality moderate, retrieved evidence partially consistent
```

## 10.6. Decision rule

```text
if U_global < τ1:
    generate confident report
elif τ1 <= U_global < τ2:
    generate cautious report with uncertainty statement
else:
    defer to human radiologist
```

## 10.7. Ý nghĩa

Medical AI không nên luôn luôn trả lời chắc chắn. Một hệ thống đáng tin cậy phải biết khi nào nó không chắc.

---

# 11. Stage 8 — Clinical Reasoning Agent

## 11.1. Mục tiêu

Thay vì sinh report trực tiếp, hệ thống cần reasoning trước.

```text
Evidence → Hypothesis → Verification → Conclusion
```

## 11.2. Input

```text
Z_f        : fused representation
R          : retrieved cases
G_case     : dynamic knowledge graph
U          : uncertainty scores
Grounding  : phrase-region map
```

## 11.3. Reasoning pipeline

```text
Step 1: Detect candidate findings
Step 2: Locate findings in anatomy regions
Step 3: Retrieve similar cases
Step 4: Query knowledge graph
Step 5: Check contradiction
Step 6: Estimate uncertainty
Step 7: Form diagnostic hypothesis
Step 8: Prepare report evidence
```

## 11.4. Candidate finding extraction

```text
F = {f1, f2, ..., fn}
```

Ví dụ:

```text
f1 = opacity
f2 = pleural effusion
f3 = cardiomegaly
```

## 11.5. Hypothesis generation

```text
H = Reason(F, G_case, R)
```

Ví dụ:

```text
Opacity + fever + lower lobe + similar pneumonia cases
→ hypothesis: pneumonia
```

## 11.6. Contradiction checking

Kiểm tra xem report có mâu thuẫn hay không.

Ví dụ mâu thuẫn:

```text
Findings: No pleural effusion.
Impression: Moderate pleural effusion.
```

Hoặc:

```text
Report nói left lung nhưng grounding nằm ở right lung.
```

## 11.7. Reasoning output

```text
ReasoningTrace = {
    findings,
    anatomy_locations,
    supporting_evidence,
    retrieved_cases,
    graph_relations,
    contradiction_status,
    uncertainty_scores,
    diagnostic_hypotheses
}
```

## 11.8. Ví dụ reasoning trace

```text
Finding:
- Right lower lobe opacity

Grounding:
- Region: right lower lung zone
- Grounding score: 0.82

Retrieved evidence:
- 7 similar cases retrieved
- 5 diagnosed as pneumonia
- 2 diagnosed as atelectasis

Knowledge graph:
- opacity located_in lower lobe
- opacity associated_with pneumonia
- fever supports infection

Uncertainty:
- medium

Conclusion:
- Possible right lower lobe pneumonia
- Recommend clinical correlation
```

---

# 12. Stage 9 — Controlled Report Generation

## 12.1. Mục tiêu

Sinh báo cáo y khoa có kiểm soát, grounded, và có uncertainty statement.

## 12.2. Input cho generator

```text
ReasoningTrace
+ visual evidence
+ retrieved evidence
+ graph evidence
+ uncertainty scores
```

## 12.3. Report format đề xuất

```text
Findings:
- ...

Impression:
- ...

Evidence:
- ...

Uncertainty:
- ...

Recommendation:
- ...
```

## 12.4. Ví dụ output

```text
Findings:
There is a patchy opacity in the right lower lung zone. No pleural effusion or pneumothorax is identified.

Impression:
Findings are suspicious for right lower lobe pneumonia.

Evidence:
The abnormality is localized to the right lower lung region and is supported by similar retrieved pneumonia cases.

Uncertainty:
Moderate uncertainty due to partial overlap with atelectatic changes.

Recommendation:
Clinical correlation and follow-up imaging are recommended if symptoms persist.
```

## 12.5. Kỹ thuật generation

```text
- Template-guided generation
- Constrained decoding
- Medical entity control
- Retrieval-conditioned generation
- Knowledge-graph-conditioned generation
- Uncertainty-aware generation
```

## 12.6. Cách giảm hallucination

```text
- Chỉ sinh finding nếu có visual grounding
- Chỉ sinh diagnosis nếu có graph support hoặc retrieved support
- Thêm uncertainty nếu evidence yếu
- Dùng contradiction checker sau khi sinh
```

---

# 13. Stage 10 — Image-aware and Clinical Evaluation

## 13.1. Vấn đề của metric truyền thống

Các metric như BLEU, ROUGE, METEOR chỉ đo độ giống văn bản, không đảm bảo report đúng về mặt lâm sàng.

Một câu report có thể fluent nhưng sai bệnh lý.

## 13.2. Nhóm metric cần dùng

### 13.2.1. Language metrics

```text
BLEU
ROUGE-L
METEOR
CIDEr
BERTScore
```

### 13.2.2. Clinical correctness metrics

```text
CheXbert F1
CheXpert label F1
RadGraph F1
Entity F1
Relation F1
Negation accuracy
```

### 13.2.3. Grounding metrics

```text
Visual grounding score
Phrase-region alignment score
Region consistency
Attention correctness
Localization IoU nếu có annotation
```

### 13.2.4. Hallucination metrics

```text
False positive finding rate
False negative omission rate
Unsupported finding rate
Contradiction rate
```

### 13.2.5. Uncertainty metrics

```text
Expected Calibration Error
Brier Score
Negative Log-Likelihood
Coverage Risk
Selective Prediction Accuracy
```

### 13.2.6. Retrieval metrics

```text
Recall@K
Precision@K
MRR
nDCG
Case similarity validity
Clinical relevance score
```

## 13.3. Evaluation pipeline

```text
Generated report
    ↓
Clinical entity extraction
    ↓
Compare with reference report
    ↓
Check image grounding
    ↓
Check hallucination / omission
    ↓
Check uncertainty calibration
    ↓
Human expert review nếu cần
```

---

# 14. Training Strategy

## 14.1. Stage-wise training

### Phase 1 — Pretrain vision-language encoder

```text
Image-report contrastive learning
```

Loss:

```text
L_align = ContrastiveLoss(Z_v, Z_t)
```

### Phase 2 — Train grounding module

```text
Phrase-region alignment
```

Loss:

```text
L_ground = CrossEntropy(region_label, predicted_region)
```

Hoặc weak supervision nếu không có annotation vùng.

### Phase 3 — Train retrieval module

```text
Positive case: same pathology / same anatomy
Negative case: different pathology / irrelevant anatomy
```

Loss:

```text
L_retrieval = TripletLoss(q, positive, negative)
```

### Phase 4 — Train graph encoder

```text
Node prediction
Relation prediction
Graph contrastive learning
```

Loss:

```text
L_graph = L_node + L_relation + L_graph_contrastive
```

### Phase 5 — Train fusion and uncertainty module

```text
Multimodal classification + calibration
```

Loss:

```text
L_uncertainty = NLL + CalibrationLoss
```

### Phase 6 — Train report generator

```text
Grounded report generation
```

Loss:

```text
L_gen = CrossEntropy(Y_ref, Y_pred)
```

### Phase 7 — Reinforcement learning / preference optimization nếu có

Reward có thể gồm:

```text
Reward =
    clinical correctness
  + grounding score
  + factual consistency
  - hallucination penalty
  - contradiction penalty
```

## 14.2. Tổng loss

```text
L_total =
    λ1 L_align
  + λ2 L_ground
  + λ3 L_retrieval
  + λ4 L_graph
  + λ5 L_uncertainty
  + λ6 L_gen
  + λ7 L_consistency
```

---

# 15. Dataset có thể dùng

## 15.1. Chest X-ray

```text
- MIMIC-CXR
- IU X-Ray
- CheXpert
- CheXpert Plus
- PadChest
- ChestX-ray14
```

## 15.2. CT / 3D Imaging

```text
- CT-RATE
- LIDC-IDRI
- MSD Medical Segmentation Decathlon
```

## 15.3. Medical VQA

```text
- VQA-RAD
- SLAKE
- PathVQA
- MIMIC-Diff-VQA
```

## 15.4. Knowledge resources

```text
- RadGraph
- UMLS
- SNOMED CT
- MeSH
- ICD
```

---

# 16. Baseline models để so sánh

## 16.1. Report generation

```text
- R2Gen
- R2GenCMN
- M2Trans
- CXR-RePaiR
- RGRG
- BLIP fine-tuned
- LLaVA-Med
- MedGemma
```

## 16.2. Vision-language pretraining

```text
- MedCLIP
- BioViL
- GLoRIA
- MGCA
- MedKLIP
```

## 16.3. Retrieval-based methods

```text
- kNN retrieval baseline
- FAISS visual retrieval
- report-based retrieval
- pathology-aware retrieval
```

## 16.4. Knowledge-enhanced methods

```text
- Knowledge graph guided RRG
- RadGraph-enhanced generation
- Grounded knowledge-enhanced VLP
```

---

# 17. Ablation Study đề xuất

## 17.1. Bỏ retrieval

```text
Full model vs without retrieval
```

Mục tiêu:

```text
Đánh giá retrieval có giúp factual correctness không.
```

## 17.2. Bỏ knowledge graph

```text
Full model vs without KG
```

Mục tiêu:

```text
Đánh giá graph có giúp reasoning và giảm hallucination không.
```

## 17.3. Bỏ uncertainty module

```text
Full model vs without uncertainty
```

Mục tiêu:

```text
Đánh giá khả năng calibration và selective prediction.
```

## 17.4. Bỏ grounding

```text
Full model vs without grounding
```

Mục tiêu:

```text
Đánh giá report có còn grounded vào image không.
```

## 17.5. Thay adaptive fusion bằng concatenation

```text
Adaptive fusion vs simple concat
```

Mục tiêu:

```text
Đánh giá vai trò của attention/gating.
```

---

# 18. Novelty có thể viết trong paper

## Contribution 1

Đề xuất framework retrieval-augmented medical vision-language agent cho report generation, trong đó hệ thống truy xuất các ca bệnh tương tự trước khi reasoning và sinh report.

## Contribution 2

Xây dựng dynamic case-specific medical knowledge graph để kết hợp anatomy, pathology, clinical context, và retrieved evidence.

## Contribution 3

Đề xuất uncertainty-aware reasoning mechanism để mô hình biết khi nào nên sinh báo cáo chắc chắn, khi nào cần báo cáo thận trọng, và khi nào cần chuyển cho bác sĩ.

## Contribution 4

Đề xuất grounding-aware report generation nhằm giảm hallucination bằng cách yêu cầu mỗi finding phải có bằng chứng từ vùng ảnh hoặc knowledge graph.

## Contribution 5

Thiết kế evaluation protocol kết hợp clinical correctness, visual grounding, hallucination rate, retrieval relevance, và uncertainty calibration.

---

# 19. Research gaps mà pipeline này giải quyết

## Gap 1 — Medical VLM thiếu clinical reasoning

Nhiều mô hình hiện tại chỉ align ảnh và text, chưa có reasoning rõ ràng.

Pipeline này thêm:

```text
retrieval + graph + reasoning agent
```

## Gap 2 — Report generation dễ hallucination

Pipeline này thêm:

```text
grounding + contradiction checking + uncertainty estimation
```

## Gap 3 — Knowledge graph thường static

Pipeline này dùng:

```text
dynamic case-specific knowledge graph
```

## Gap 4 — Retrieval hiện còn shallow

Pipeline này dùng:

```text
visual + pathology + anatomy + clinical + uncertainty-aware retrieval
```

## Gap 5 — Model không biết khi nào không chắc

Pipeline này thêm:

```text
uncertainty-aware decision rule
```

---

# 20. Hướng mở rộng

## 20.1. Longitudinal reasoning

```text
Image_t1 → Image_t2 → Image_t3 → disease progression
```

Ứng dụng:

```text
- treatment response prediction
- disease progression forecasting
- follow-up report generation
```

## 20.2. Personalized medical AI

```text
patient-specific memory
hospital-specific adaptation
doctor-specific reporting style
```

## 20.3. Federated medical VLM

```text
privacy-preserving training
cross-hospital collaboration
non-IID adaptation
```

## 20.4. Multi-agent clinical system

```text
Retrieval Agent
Grounding Agent
Knowledge Graph Agent
Uncertainty Agent
Reasoning Agent
Report Agent
Verifier Agent
```

---

# 21. Pseudocode tổng quát

```python
class TrustMedRAG:
    def __init__(self,
                 vision_encoder,
                 text_encoder,
                 retriever,
                 graph_builder,
                 fusion_module,
                 uncertainty_estimator,
                 reasoning_agent,
                 report_generator):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.retriever = retriever
        self.graph_builder = graph_builder
        self.fusion_module = fusion_module
        self.uncertainty_estimator = uncertainty_estimator
        self.reasoning_agent = reasoning_agent
        self.report_generator = report_generator

    def forward(self, image, clinical_context, prior_report=None):
        image_clean = preprocess_image(image)

        visual_features = self.vision_encoder(image_clean)
        text_features = self.text_encoder(clinical_context, prior_report)

        grounding_map = anatomy_lesion_grounding(
            visual_features,
            text_features
        )

        retrieved_cases = self.retriever.search(
            visual_features=visual_features,
            text_features=text_features,
            grounding_map=grounding_map
        )

        case_graph = self.graph_builder.build(
            clinical_context=clinical_context,
            grounding_map=grounding_map,
            retrieved_cases=retrieved_cases
        )

        fused_features = self.fusion_module(
            visual_features=visual_features,
            text_features=text_features,
            retrieved_cases=retrieved_cases,
            case_graph=case_graph
        )

        uncertainty = self.uncertainty_estimator(fused_features)

        reasoning_trace = self.reasoning_agent.reason(
            fused_features=fused_features,
            retrieved_cases=retrieved_cases,
            case_graph=case_graph,
            grounding_map=grounding_map,
            uncertainty=uncertainty
        )

        report = self.report_generator.generate(
            reasoning_trace=reasoning_trace,
            uncertainty=uncertainty
        )

        return report, reasoning_trace, uncertainty
```

---

# 22. Tên đề tài gợi ý

## Option 1

```text
TrustMed-RAG: Uncertainty-aware Retrieval-Augmented Knowledge-Grounded Vision-Language Model for Trustworthy Radiology Report Generation
```

## Option 2

```text
A Dynamic Knowledge-Grounded Retrieval-Augmented Medical Vision-Language Agent for Explainable Radiology Report Generation
```

## Option 3

```text
Uncertainty-aware Clinical Reasoning with Retrieval-Augmented Medical Vision-Language Models
```

## Option 4

```text
Toward Trustworthy Medical Vision-Language Agents: Retrieval, Dynamic Knowledge Graphs, and Uncertainty-aware Reasoning
```

---

# 23. Cấu trúc paper đề xuất

## Abstract

Trình bày vấn đề hallucination, lack of grounding, thiếu reasoning, và đề xuất TrustMed-RAG.

## 1. Introduction

- Medical VLM phát triển mạnh.
- Nhưng còn hallucination, shortcut bias, thiếu visual grounding.
- Retrieval và knowledge graph có tiềm năng nhưng chưa kết hợp uncertainty-aware reasoning.
- Đề xuất framework mới.

## 2. Related Work

```text
- Radiology Report Generation
- Medical Vision-Language Models
- Retrieval-Augmented Generation in Medicine
- Knowledge Graph for Medical AI
- Uncertainty Estimation and Trustworthy AI
```

## 3. Method

```text
- Problem formulation
- Vision-language encoder
- Grounding module
- Retrieval module
- Dynamic KG construction
- Adaptive fusion
- Uncertainty estimator
- Reasoning and generation
```

## 4. Experiments

```text
- Dataset
- Baselines
- Metrics
- Implementation details
- Main results
- Ablation studies
```

## 5. Discussion

```text
- Clinical usefulness
- Trustworthiness
- Failure cases
- Limitations
```

## 6. Conclusion

Tóm tắt đóng góp và hướng mở rộng.

---

# 24. Kết luận

Pipeline này có thể tạo ra một hướng nghiên cứu mạnh vì nó kết hợp nhiều vấn đề đang rất quan trọng trong Medical AI:

```text
Medical Vision-Language Model
+ Retrieval-Augmented Generation
+ Dynamic Knowledge Graph
+ Visual Grounding
+ Uncertainty Estimation
+ Clinical Reasoning Agent
+ Trustworthy Report Generation
```

Điểm mới chính không nằm ở một module đơn lẻ, mà nằm ở cách kết hợp các module để tạo thành một hệ thống medical AI có khả năng reasoning, kiểm chứng bằng chứng, và biết khi nào không chắc chắn.

---

# 25. Trạng thái triển khai trong repo

Phần trên mô tả kiến trúc nghiên cứu mục tiêu. Repo hiện tại đã có một bản POC khá đầy đủ, nhưng cần phân biệt rõ giữa module đã chạy được, module đã được kiểm thử smoke, và phần cần thực nghiệm thêm trên Kaggle.

## 25.1. Đã triển khai và kiểm thử local

```text
src/medical_vlm_pipeline/data.py
src/medical_vlm_pipeline/encoders.py
src/medical_vlm_pipeline/alignment.py
src/medical_vlm_pipeline/quantization.py
src/medical_vlm_pipeline/retrieval.py
src/medical_vlm_pipeline/heads.py
src/medical_vlm_pipeline/generation.py
src/medical_vlm_pipeline/explainability.py
src/medical_vlm_pipeline/grounding.py
src/medical_vlm_pipeline/knowledge_graph.py
src/medical_vlm_pipeline/fusion.py
src/medical_vlm_pipeline/uncertainty.py
src/medical_vlm_pipeline/reasoning.py
src/medical_vlm_pipeline/pipeline.py
```

Các kiểm tra local hiện tại:

```text
python -m compileall src/medical_vlm_pipeline src/train_pipeline.py
pytest -q src/tests
```

Trạng thái gần nhất:

```text
compileall: pass
pytest: 7 passed
```

## 25.2. Pipeline đang có hai mức

### Baseline trainable pipeline

`MedicalVLMPipeline` là pipeline chính đang được `train_pipeline.py` train trên Kaggle.

Nó bao gồm:

```text
Image encoder
Text encoder
Projection head
Product quantization
FAISS retrieval
Diagnosis head
Report generation
MC Dropout uncertainty
```

Đây là nhánh phù hợp để chạy thực nghiệm 150 epoch trước vì đã có training loop, metric logging, checkpoint, và artifact packaging.

### TrustMed-RAG full POC pipeline

`TrustMedRAGPipeline` là bản POC 10-stage theo kiến trúc nghiên cứu trong tài liệu này.

Nó bao gồm:

```text
Grounding
Retrieval
Dynamic knowledge graph
Adaptive multimodal fusion
Uncertainty estimation
Clinical reasoning
Controlled generation
```

Pipeline này đã có smoke test, nhưng chưa phải training target chính. Các module grounding, knowledge graph, fusion và reasoning hiện vẫn thiên về POC/heuristic-neural hybrid, chưa được supervised bằng region-level annotation hoặc clinical reasoning label.

## 25.3. Những cải tiến đã thêm cho chất lượng thực nghiệm

### Weak label fix

Nhãn yếu từ report đã được sửa để hiểu negation:

```text
no pneumothorax
negative for pleural effusion
without evidence of consolidation
```

Điều này tránh lỗi nghiêm trọng trước đó: câu phủ định vẫn bị gán nhãn bệnh dương tính.

### Grouped validation split

Split cũ theo image gây leakage vì một report/study có nhiều ảnh. Repo đã thêm:

```text
--split-group report
```

Mục tiêu là tránh cùng report hoặc paired views xuất hiện đồng thời ở train và validation.

### Imbalanced classification

Dataset IU CXR sau weak label fix bị lệch mạnh về Normal. Training đã thêm:

```text
--class-weighting balanced
macro_precision
macro_recall
macro_f1_score
per-class metrics
confusion matrix
```

Do đó, đánh giá không nên chỉ dựa vào accuracy.

### Kaggle T4/T4x2 workflow

Notebook và training script đã thêm:

```text
--device cuda
--multi-gpu auto
--gpu-ids 0,1
CUDA_LAUNCH_BLOCKING=False
nvidia-smi monitor log
```

Mục tiêu là kiểm tra rõ GPU có được dùng thật hay không, thay vì chỉ nhìn `torch.cuda.is_available()`.

### Fast smoke mode

Để kiểm tra epoch 1 không bị chậm vì post-processing, training script đã thêm:

```text
--train-eval-every 0
--eval-contrastive      # optional, mặc định tắt
--skip-post-train       # bỏ build retriever và text generation khi chỉ test tốc độ
```

Điều này giúp đo đúng thời gian train epoch, không bị lẫn thời gian build index hoặc sinh báo cáo.

## 25.4. Những phần chưa nên coi là hoàn tất

```text
1. Full TrustMedRAGPipeline chưa được train end-to-end.
2. Grounding chưa có supervised region labels.
3. Knowledge graph hiện dùng ontology seed nhỏ, chưa nối UMLS/RadGraph thật.
4. Reasoning agent là structured reasoning POC, chưa được đánh giá bởi bác sĩ.
5. T4x2 DataParallel cần xác nhận bằng nvidia_smi_during_train.csv trên Kaggle.
6. Full 150 epoch sau label fix + grouped split chưa có báo cáo kết quả cuối.
7. Bài toán hiện vẫn là weak single-label; hướng đúng hơn có thể là binary Normal/Abnormal hoặc multi-label pathology classification.
```

## 25.5. Tiêu chí coi bản thực nghiệm tiếp theo là hợp lệ

Một run Kaggle mới nên được coi là hợp lệ khi có đủ:

```text
environment.json:
  device = cuda:0
  cuda_available = true
  multi_gpu.enabled = true nếu dùng T4x2

run_config.json:
  derive_labels_from_report = true
  split_group = report
  class_weighting = balanced

training_metrics.csv:
  có macro_f1_score
  có val_macro_f1_score
  có per-class metrics

nvidia_smi_during_train.csv:
  GPU 0 có utilization/memory
  GPU 1 có utilization/memory nếu chạy T4x2

epoch_metrics.json:
  confusion matrix không collapse vào một lớp duy nhất
```

Nếu các điều kiện này đạt, khi đó mới nên dùng kết quả để viết phần experiment trong paper.
