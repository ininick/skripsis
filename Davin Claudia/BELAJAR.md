# BELAJAR BARENG CLAUDE — Catatan Sesi
## Davin Raffilio | Persiapan Sidang 15 Juni 2026

---

## FORMAT BELAJAR
Setiap topik dijelaskan dengan format:
- 📖 Teori singkat
- 🔄 Before vs After (contoh nyata)
- ✅ Yang dipakai + alasan
- ❌ Alternatif yang tidak dipakai + alasan
- 🎤 Cara jawab ke dosen (kalimat siap pakai)

---

## 📚 PROGRESS BELAJAR — DAFTAR ISI SESI

### ✅ Sudah Dipelajari:
1. Import library (torch, nn, F, Dataset, DataLoader, AutoTokenizer, AutoModel)
2. SEED=42 & set_seed (reproducibility)
3. MAX_LENGTH=512 (hard limit BERT)
4. GRU_HIDDEN=256 (dari ablation)
5. Cara kerja GRU (sticky note + update/reset gate)
6. Training loop (forward, loss, backward)
7. Cross Entropy Loss (`-log(p)`)
8. Class Weight (×1.5 amplifikasi)
9. CPU vs GPU & `.to(device)`
10. Focal Loss & gamma=2.0
11. GRU_LAYERS=2 (2 tim analis berlapis)
12. DROPOUT=0.2 (lampu mati random)
13. Data Loading (3 CSV → concat → df_seed)
14. EDA & Preprocessing (stratified split, label remap, class weights)
15. Tokenisasi (`attention_mask`, `padding=False`, DataCollator)
16. Arsitektur IndoBertGRU (komponen + forward step-by-step)
17. Tensor shape `[batch, token, 768]`
18. Padding & PAD token (kenapa ada lalu dibuang)
19. BiGRU & h_n[-2], h_n[-1] (gedung 4 lantai)
20. Linear layer (matriks × bobot + bias)
21. IndoBERT belajar via Masked Language Modeling
22. **KLARIFIKASI**: 768 dimensi = MAKNA bukan SENTIMEN
23. **SESI 19**: Kenapa BERT di CPU untuk XGBoost pipeline
24. **SESI 20**: Debug OOM notebook XGBoost (reuse embeddings + memory cleanup)
25. **SESI 21**: Proses labeling dataset — reverse engineering & metodologi
26. **SESI 22**: Semi-supervised labeling pipeline (TF-IDF + LR dari contoh manual)
27. **SESI 23**: Scoreboard final 7 model + analisis BiLSTM overfitting

### ⏳ Belum Dibahas (Next Session):
- FocalLossTrainer & compute_metrics detail
- TrainingArguments (14 parameter + justifikasi)
- `trainer.train()` — apa yang terjadi di balik layar
- Ablation Study — semua cell masih di-comment, perlu dirun
- K-Fold CV — semua cell masih di-comment, perlu dirun
- XAI (Integrated Gradients DL, SHAP XGBoost/RFC) — perlu dirun
- Statistical Testing (Bootstrap CI + McNemar) — perlu dirun
- Paper update: isi placeholder K-Fold, McNemar, Efficiency setelah dirun

---

## SESI 19: KENAPA BERT DI CPU UNTUK PIPELINE XGBOOST?

Log Kaggle menunjukkan:
```
BERT device: cpu (CPU for compatibility)
GPU available: Tesla T4
XGBoost will use GPU for training
```

**Pertanyaan wajar: kenapa tidak semuanya GPU?**

### Jawabannya: disengaja, bukan bug!

Pipeline XGBoost punya **2 tahap yang beda ekosistem**:

```
Tahap 1: IndoBERT → extract embeddings (inference saja, dilakukan SEKALI)
Tahap 2: XGBoost → training classifier (butuh banyak iterasi)
```

### Kenapa tidak bisa keduanya GPU sekaligus?

PyTorch (untuk BERT) dan XGBoost GPU backend sama-sama pakai CUDA, tapi **memory management-nya sendiri-sendiri**. Kalau dijalankan bersamaan di GPU yang sama bisa konflik alokasi memori CUDA → crash.

### Trade-off yang masuk akal:

| | IndoBERT | XGBoost |
|---|---|---|
| Mode | Inference (tidak ditraining!) | Training (ribuan iterasi) |
| Dilakukan | Sekali saja | Berkali-kali sampai konvergen |
| Device | CPU → aman | GPU → perlu parallelism! |

🎤 **Jawab dosen:**
> *"Pada pipeline IndoBERT + XGBoost, IndoBERT dijalankan di CPU untuk feature extraction karena prosesnya hanya inference satu kali. GPU dialokasikan penuh untuk XGBoost training yang membutuhkan paralelisme tinggi via CUDA backend, menghindari konflik alokasi memori CUDA antara dua framework."*

---

## SESI 20: DEBUG OOM NOTEBOOK XGBOOST

### Root Cause:
Notebook XGBoost punya **14 model ditraining sekaligus** (1 utama + 8 ablasi + 5 K-Fold) tanpa cleanup memori antara model. Ditambah ekstraksi embedding dilakukan berulang.

### Fixes yang dilakukan:
1. **Ablation cells**: ganti `device='cuda'` → `device='cpu'` + `del model; gc.collect()` per iterasi
2. **K-Fold**: reuse `np.vstack([X_train, X_val])` bukan ekstrak ulang
3. **Setelah ekstraksi**: `del bert_model; gc.collect()` — bebaskan 430MB
4. **Sementara**: comment semua reviewer cells → save model dulu → uncomment nanti

### Pelajaran:
XGBoost tidak pakai HuggingFace Trainer → memory management **harus manual** berbeda dengan GRU/CNN yang otomatis diurus Trainer.

---

## SESI 21: PROSES LABELING DATASET

### Flow Asli (Reverse Engineered dari Code di Repo):

```
RAW DATA (scraping BeautifulSoup)
CNBC: 5,343 | Detik: 9,999 | Kompas: 9,980
        ↓
CLEANING (remove duplikat, boilerplate, artikel pendek)
        ↓
SAMPLING SEED (random seed=42)
~800 artikel per portal = ~2,400 total
        ↓
KEYWORD-ASSISTED PRE-LABELING
Script kasih 'label_suggest' berdasarkan:
- POS_KW: apresiasi, tumbuh, kondusif, berhasil...
- NEG_KW: korupsi, tersangka, anjlok, gagal...
        ↓
MANUAL REVIEW oleh tim (isi Excel)
~2,400 artikel dilabeli manual dengan panduan keyword
        ↓
AUTO-LABEL SISA ARTIKEL
TF-IDF + Logistic Regression belajar dari 2,400 contoh
→ prediksi label 22,900 artikel sisanya
        ↓
DATASET FINAL BERLABEL: 25,322 artikel
```

### File bukti ada di repo (`DATASET guide/`):
- `Step 2`: sampling scripts + keyword lexicon
- `Step 3`: Excel hasil labeling manual tim
- `Step 3.5`: `auto_label_from_examples.ipynb` (TF-IDF + LR)
- `Step 6`: dataset final

🎤 **Jawab dosen soal labeling:**
> *"Proses anotasi menggunakan pendekatan semi-supervised. Pertama, kami sampling ~800 artikel per portal (seed=42) dan menggunakan keyword-assisted pre-labeling sebagai panduan — sistem memberikan saran label berdasarkan lexicon sentimen domain-spesifik. Tim kemudian mereview dan memfinalisasi label secara manual (~2,400 artikel). Selanjutnya, TF-IDF + Logistic Regression dilatih dari contoh manual tersebut untuk memprediksi label 22,900 artikel sisanya. Dataset final 25,322 artikel adalah gabungan label manual dan prediksi."*

🎤 **Kalau ditanya 'kenapa tidak label semua manual?':**
> *"Melabeli 25,322 artikel secara fully manual tidak feasible dari sisi waktu dan sumber daya. Pendekatan semi-supervised ini adalah praktik standar dalam NLP research untuk dataset berskala besar."*

---

## SESI 22: SEMI-SUPERVISED LABELING PIPELINE

### Cara Kerja `auto_label_from_examples.ipynb`:

**Step 1 — TF-IDF Vectorization:**
Ubah teks menjadi vektor numerik berdasarkan frekuensi kata.
- `max_features=20,000`: pakai 20K kata paling penting
- `ngram_range=(1,2)`: unigram + bigram ("korupsi", "tidak korupsi")
- Hasilnya: setiap artikel jadi vektor 20K dimensi

**Step 2 — Logistic Regression belajar dari contoh manual:**
```python
clf.fit(X_manual, y_manual)
# X_manual = TF-IDF dari 2,400 artikel berlabel manual
# y_manual = label -1/0/1 yang tim isi
```
Model belajar: "kombinasi kata apa yang cenderung → label apa?"

**Step 3 — Predict sisa artikel:**
```python
df_unlabeled['label'] = clf.predict(X_unlabeled)
```

### Kenapa TF-IDF + Logistic Regression, bukan IndoBERT?
- **Interpretable**: bisa lihat kata paling berpengaruh per kelas
- **Reproducible**: deterministik
- **Fully grounded in manual labels**: murni belajar dari contoh manusia
- **No AI/LLM**: murni statistik

---

## SESI 23: SCOREBOARD FINAL + ANALISIS BILSTM

### Scoreboard 7 Model:

| Model | Macro F1 | Accuracy |
|---|---|---|
| IndoBERT Fine-Tuned | 82.85% 🥇 | 83.81% |
| IndoBERT + CNN | 82.55% 🥈 | 83.55% |
| IndoBERT + GRU | 82.45% 🥉 | 83.51% |
| IndoBERT + BiLSTM | 76.56% | 77.93% |
| IndoBERT + XGBoost | 68.28% | 70.64% |
| IndoBERT + RFC | 63.27% | 70.33% |
| IndoBERT Base (Frozen) | 35.80% | 37.93% |

### Analisis BiLSTM (Epoch Training):
BiLSTM training loss terus turun (7.49 → epoch 15) tapi validation loss terus **naik** dari epoch 3 (0.29 → 0.91) → **classic overfitting**.

**Kenapa BiLSTM overfitting lebih parah dari GRU?**
- LSTM punya **4 gate** (input, forget, cell, output) vs GRU **3 gate** (update, reset, hidden)
- Parameter LSTM lebih banyak → lebih mudah hafal training data
- Pada 25K dataset, GRU cukup dan lebih efisien

🎤 **Jawab dosen:**
> *"BiLSTM menunjukkan tanda overfitting — validation loss terus meningkat dari epoch 3 meskipun training loss turun. Ini konsisten dengan teori bahwa LSTM dengan 4 gate memiliki lebih banyak parameter dibanding GRU dengan 3 gate, sehingga lebih rentan overfitting pada dataset medium-size 25K artikel."*

### Insight Penting — IndoBERT Base vs Fine-Tuned:
- Base Frozen: 35.80% — hampir random (3-class random = 33%)
- Fine-Tuned: 82.85%
- Gap: **57%** — membuktikan fine-tuning SANGAT penting!

🎤 **Jawab dosen:**
> *"Perbandingan antara IndoBERT Base (frozen, 35.80%) dengan IndoBERT Fine-Tuned (82.85%) memberikan bukti empiris tentang pentingnya task-specific fine-tuning. Model frozen hanya mengandalkan representasi general IndoBERT tanpa adaptasi ke domain sentimen politik, menghasilkan performa hampir setara random baseline."*

---

*Dokumen materi pembelajaran. Untuk progress skripsi & status model lihat CLAUDE.md.*
*Last updated: 5 Juni 2026*
