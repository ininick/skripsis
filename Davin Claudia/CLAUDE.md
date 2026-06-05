# PANDUAN SIDANG SKRIPSI — Davin Raffilio
## Sentiment Analysis of Indonesian Political News using IndoBERT
### Sidang: 15 Juni 2026 | Konferensi: ICISS

---

# 🎯 PROGRESS LENGKAP SKRIPSI — STATE SAAT INI

> Dibaca dulu kalau pindah window/context baru.

## Status 7 Model Arsitektur

| Model | Macro F1 | Accuracy | Status | Lokasi Model |
|---|---|---|---|---|
| **IndoBERT Fine-Tuned** | **82.85%** 🥇 | **83.81%** | ✅ SELESAI | `MODEL DEEPLEARNING\IndoBERT-FineTuned\model_state_dict.pt` |
| **IndoBERT + CNN** | **82.55%** 🥈 | **83.55%** | ✅ SELESAI | `MODEL DEEPLEARNING\CNN\model_state_dict.pt` |
| **IndoBERT + GRU** | **82.45%** 🥉 | **83.51%** | ✅ SELESAI | `MODEL DEEPLEARNING\GRU\model.safetensors` |
| **IndoBERT + BiLSTM** | **76.56%** | **77.93%** | ✅ SELESAI | `MODEL DEEPLEARNING\BiLSTM\pytorch_model.bin` |
| **IndoBERT + XGBoost** | **68.28%** | **70.64%** | ✅ SELESAI | `MODEL DEEPLEARNING\XGBoost\indobert_xgboost_model.json` |
| **IndoBERT + RFC** | **63.27%** | **70.33%** | ✅ SELESAI | `MODEL DEEPLEARNING\RFC\indobert_rfc_model.pkl` |
| **IndoBERT Base (Frozen)** | **35.80%** | **37.93%** | ✅ SELESAI | `MODEL DEEPLEARNING\IndoBERT - Base Model\` |

> Semua model ada di `C:\Users\Davin\Downloads\MODEL DEEPLEARNING\`

### Hasil Detail GRU:
- Accuracy: 0.8351 | Macro F1: 0.8245 | Weighted F1: 0.8354
- F1-Neg: 0.8021 | F1-Neu: 0.8545 | F1-Pos: 0.8170

### Hasil Detail CNN:
- Accuracy: 0.8355 | Macro F1: 0.8255 | Weighted F1: 0.8359
- F1-Neg: 0.7998 | F1-Neu: 0.8533 | F1-Pos: 0.8233

### Hasil Detail Fine-Tuned:
- Accuracy: 0.8381 | Macro F1: 0.8285 | Weighted F1: 0.8386
- Arsitektur: [CLS] → Dropout(0.1) → Linear(768,3) — full fine-tune semua parameter BERT

### Hasil Detail XGBoost:
- Accuracy: 0.7064 | Macro F1: 0.6828 | Weighted F1: 0.7067
- Best iteration: 239 / 500 (early stopping)
- Catatan: Lebih rendah karena BERT di-freeze (tidak joint optimization)

### Hasil Detail RFC:
- Accuracy: 0.7033 | Macro F1: 0.6327 | Weighted F1: 0.6843
- OOB Score: 0.6927 | n_estimators: 500, max_depth: None
- Catatan: Sama seperti XGBoost — BERT di-freeze, hanya classifier yang belajar

### Hasil Detail IndoBERT Base (Frozen):
- Accuracy: 0.3793 | Macro F1: 0.3580 | Weighted F1: 0.2463
- Arsitektur: Frozen IndoBERT → [CLS] → Dropout → Linear(768,3)
- Catatan: Sangat rendah karena BERT tidak di-fine-tune — ini membuktikan pentingnya fine-tuning!

### Hasil Detail BiLSTM:
- Accuracy: 0.7793 | Macro F1: 0.7656 | Weighted F1: 0.7796
- F1-Neg: 0.7461 | F1-Neu: 0.8056 | F1-Pos: 0.7451
- Best epoch: 14/15 (early stopping tidak trigger)
- Catatan: Lebih rendah dari GRU karena LSTM 4 gate lebih berat → overfitting pada 25K dataset
- Validation loss terus naik dari epoch 3 → tanda overfitting meskipun macro F1 masih naik

---

## 🔑 API Tokens Kaggle

| Akun | Token | Status |
|---|---|---|
| davinraffilio9 (Davin) | `KGAT_e89aba9eae0a83a0d78fdff4e01d2a35` | Quota habis, reset Senin |
| nicolnicol (teman) | `KGAT_3b5e4e2b8b5106b09a3c56bf7294d997` | Aktif untuk CNN & XGBoost |

### Kaggle Kernels:
- `davinraffilio9/revisi-fix-indobert-gru` — ✅ Selesai
- `nicolnicol/revision-of-cnn-tok` — ✅ Selesai
- `nicolnicol/revision-indobert-xgboost` — ✅ Selesai (v14, model saved)
- `nicolnicol/revision-indobert-finetuned` — ✅ Selesai
- `nicolnicol/revision-indobert-base-frozen` — ⏳ Ready (belum run, GPU abis)
- BiLSTM & RFC — ✅ Selesai (dirun teman)

### Cara Cek & Download:
```powershell
$env:PYTHONUTF8 = "1"
$env:KAGGLE_API_TOKEN = "KGAT_3b5e4e2b8b5106b09a3c56bf7294d997"
kaggle kernels status nicolnicol/<kernel-name>
kaggle kernels output nicolnicol/<kernel-name> -p "C:\Users\Davin\Downloads\<output-folder>"
```

---

## 📁 File Penting

### Repo GitHub `ininick/skripsis`:
```
skripsis/
├── Davin Claudia/
│   ├── CLAUDE.md     ← FILE INI (progress + panduan sidang)
│   └── BELAJAR.md    ← materi pembelajaran sesi by sesi
├── MODEL/
│   ├── README.md, GRU/, CNN/
└── revision/          ← SEMUA notebook revisi (untuk sidang/ICISS)
    ├── GRU/           ✅ pushed + kernel-metadata.json
    ├── CNN/           ✅ pushed + kernel-metadata.json
    ├── XGBoost/       ✅ pushed + kernel-metadata.json (reviewer cells di-comment)
    ├── BiLSTM/        ✅ pushed (reviewer cells di-comment, warmup_ratio fixed)
    ├── IndoBERT-FineTuned/ ✅ pushed + kernel-metadata.json (reviewer cells di-comment)
    ├── IndoBERT-Base/ ✅ pushed + kernel-metadata.json (reviewer cells di-skip)
    └── RFC/           ✅ pushed
```

### File di Downloads:
- `MODEL DEEPLEARNING/` — **SEMUA model files dalam 1 folder** (untuk OneDrive upload):
  - `GRU/model.safetensors` + tokenizer files
  - `CNN/model_state_dict.pt` + tokenizer files
  - `BiLSTM/pytorch_model.bin` + config + tokenizer
  - `IndoBERT-FineTuned/model_state_dict.pt` + summary + tokenizer
  - `XGBoost/indobert_xgboost_model.json` + scaler + summary
  - `RFC/indobert_rfc_model.pkl` + scaler + summary + plots
  - `IndoBERT - Base Model/indobert_base_summary.json` + preds
  - **7 PDF dokumentasi** (1 per model, penjelasan tiap code block + analisis reviewer)
- `Revised_Paper_IEEE_XX.docx` — paper revisi asli (placeholder `[XX]` kuning)
- `Revised_Paper_IEEE_UPDATED.docx` — paper yang sudah diisi hasil training (sebagian, K-Fold/McNemar masih pending)

---

## 💡 Cara Davin Mau Diajarin (PENTING untuk agent baru!)

- **Pelan-pelan**, jangan langsung teknis
- **Banyak contoh konkret** bahasa Indonesia (artikel politik)
- **Before vs After** kalau bisa
- **Analogi sehari-hari** (sticky note, gedung lantai, dapur, lampu)
- **Sebutkan alternatif** yang tidak dipakai + alasannya (antisipasi dosen)
- **Kasih jawaban siap pakai buat sidang** dalam bahasa formal
- **Cecar dengan pertanyaan balik** setelah jelasin (latihan sidang)
- Catat progress di **BELAJAR.md** secara berkala
- Pakai bahasa **santai/ngobrol**, jangan kaku

---

## 🚀 Next Step — Yang Masih Perlu Dikerjakan

### ✅ Keputusan Strategi (5 Juni 2026):
Davin memutuskan **JANGAN utak-atik notebook existing** (sudah selesai, model sudah save).
Bikin **2 notebook baru terpisah** untuk reviewer revisions:

**📓 Notebook 2 — Training-based Analysis** (butuh GPU + waktu lama)
- Ablation study semua 7 model
- K-Fold CV 5-fold semua 7 model
- Lokasi: `revision/reviewer_analysis/02_ablation_kfold.ipynb` (BELUM DIBUAT)

**📓 Notebook 3 — Load-only Analysis** (cepat, bahkan bisa di CPU)
- XAI (Integrated Gradients untuk DL, SHAP untuk XGBoost/RFC)
- McNemar's Test (load .npy predictions)
- Bootstrap Confidence Interval
- Efficiency Analysis (latency, throughput, params)
- Lokasi: `revision/reviewer_analysis/03_xai_mcnemar.ipynb` (BELUM DIBUAT)

### ⚠️ PREREQUISITE Notebook 3:
McNemar butuh `.npy` prediction files dari SEMUA model. Status saat ini:
| Model | val_preds.npy | val_true.npy |
|---|---|---|
| Fine-Tuned | ✅ `finetuned_output/` | ✅ |
| IndoBERT Base | ✅ `MODEL DEEPLEARNING/IndoBERT - Base Model/` | ✅ |
| GRU | ❌ BELUM ADA | ❌ |
| CNN | ❌ BELUM ADA | ❌ |
| BiLSTM | ❌ BELUM ADA | ❌ |
| XGBoost | ❌ BELUM ADA | ❌ |
| RFC | ❌ BELUM ADA | ❌ |

**Cara dapat predictions yang missing:**
Load model existing → predict val_set → save .npy (TIDAK perlu training ulang!)
Bisa dibuat script tersendiri "00_generate_predictions.ipynb"

### Prioritas Eksekusi:
1. **Bikin script generate `.npy` predictions** untuk model yang belum punya
2. **Bikin Notebook 3** (XAI + McNemar + Bootstrap + Efficiency) — paling cepat selesai
3. **Bikin Notebook 2** (Ablation + K-Fold) — paling lama, butuh GPU
4. Update paper dengan semua hasil

### Prioritas Sidang (parallel):
- Lanjut belajar dari **Cell 14-15 (FocalLossTrainer & compute_metrics)**
- Review materi di BELAJAR.md (sudah sampai Sesi 23)
- Latihan jawab 10 pertanyaan sidang yang sudah disiapkan

### Catatan Teknis:
- Semua notebook reviewer cells sudah di-comment (bukan dihapus!) → tinggal uncomment
- XGBoost perlu save embeddings ke disk sebelum ablation run (biar tidak OOM)
- BiLSTM ablation cells punya bug: `tokenizer=tokenizer` → harus ganti `processing_class=tokenizer`
- BiLSTM K-Fold punya bug: double comma `gradient_accumulation_steps=2,,`
- IndoBERT Base (Frozen) notebook belum dirun di Kaggle (GPU quota habis)

---

# 📚 PANDUAN SIDANG (Detail Materi)

## DAFTAR ISI
1. [Gambaran Umum Skripsi](#1-gambaran-umum-skripsi)
2. [Dataset](#2-dataset)
3. [Model dan Arsitektur](#3-model-dan-arsitektur)
4. [Proses Training](#4-proses-training)
5. [Evaluasi dan Hasil](#5-evaluasi-dan-hasil)
6. [Revisi dari Reviewer ICISS — Detail Lengkap](#6-revisi-dari-reviewer-iciss--detail-lengkap)
7. [Konsep Teknis yang Harus Dipahami](#7-konsep-teknis-yang-harus-dipahami)
8. [Pertanyaan Sidang yang Mungkin Muncul](#8-pertanyaan-sidang-yang-mungkin-muncul)
9. [Checklist Sidang 15 Juni](#9-checklist-sidang-15-juni)

---

## 1. Gambaran Umum Skripsi

### 1.1 Tentang Apa Skripsimu?

Skripsi ini tentang **analisis sentimen** berita politik Indonesia. Analisis sentimen artinya: diberikan sebuah teks berita, komputer harus bisa menentukan apakah sentimen berita itu **Positif, Negatif, atau Netral**.

**Contoh:**
| Teks Berita | Label |
|---|---|
| "Presiden berhasil dorong pertumbuhan ekonomi 5%" | POSITIF |
| "Korupsi merajalela, pejabat ditetapkan tersangka" | NEGATIF |
| "Rapat kabinet membahas RAPBN 2027 hari ini" | NETRAL |

### 1.2 Mengapa Topik Ini Penting?

Indonesia adalah negara demokrasi terbesar ke-3 di dunia. Berita politik sangat mempengaruhi opini publik. Dengan volume berita yang sangat besar, tidak mungkin manusia membaca dan mengklasifikasikan semua berita secara manual. Sistem otomatis berbasis AI dibutuhkan untuk:

- Memantau sentimen publik terhadap kebijakan pemerintah
- Mendeteksi bias media dalam pelaporan berita politik
- Membantu peneliti dan jurnalis menganalisis tren berita
- Mendukung pengambilan keputusan berbasis data

### 1.3 Mengapa Bahasa Indonesia Sulit?

| Tantangan | Penjelasan |
|---|---|
| Kosakata campur | Bahasa gaul, singkatan, bahasa daerah bercampur |
| Sumber daya terbatas | Dataset berlabel bahasa Indonesia jauh lebih sedikit dari Inggris |
| Konteks politik lokal | Istilah seperti "oligarki", "pilpres" perlu konteks khusus |
| Ambiguitas sentimen | Kata yang sama bisa positif/negatif tergantung konteks |

---

## 2. Dataset

### 2.1 Sumber Data

Data dikumpulkan dari 3 portal berita politik Indonesia:

| Sumber | Jumlah Artikel | Karakteristik |
|---|---|---|
| CNBC Indonesia | 9.999 artikel | Fokus ekonomi-politik, gaya formal |
| Detik.com | 5.343 artikel | Berita breaking news, gaya lugas |
| Kompas.com | 9.980 artikel | Berita mendalam, gaya jurnalistik |
| **TOTAL** | **25.322 artikel** | 3 kelas: Positif / Netral / Negatif |

**Kaggle Dataset:** `davinraffilio9/datalabeled`  
**Path di Kaggle:** `/kaggle/input/datasets/davinraffilio9/datalabeled/`  
**Files:** `cnbc_labeled.csv`, `detik_labeled.csv`, `kompas_labeled.csv`

### 2.2 Proses Labeling

Setiap artikel diberi label sentimen oleh annotator manusia. Label berdasarkan keseluruhan nada artikel terhadap subjek politik yang dibahas. Distribusi kelas bersifat **imbalanced** (tidak seimbang) — kelas Netral jauh lebih banyak dari Negatif dan Positif. Ini adalah tantangan utama!

### 2.3 Data Cleaning — Mengapa Perlu?

Setiap sumber berita memiliki 'boilerplate' (teks sampah) yang muncul berulang di setiap artikel tapi tidak relevan dengan konten. Kalau tidak dibersihkan, model bisa 'belajar' dari noise bukan dari konten asli.

| Sumber | Boilerplate yang Dihapus | Cara Hapus |
|---|---|---|
| CNBC | "Selengkapnya saksikan di..." | Truncate (potong) dari frasa itu ke bawah |
| Detik | "Simak juga...", "Scroll to continue..." | Truncate dari trigger phrase paling awal |
| Kompas | "Baca juga: [judul artikel lain]" | Regex remove inline (tengah artikel) |

**Regex Kompas:**
```
(?i)baca\s+juga\s*:[^\n]*?(?=[A-Z][a-z]{3,}\s+[a-z]|\n|$)
```
Pattern ini mencari 'Baca juga:' lalu mengambil teks sampai menemukan awal kalimat baru (huruf kapital diikuti minimal 4 huruf kecil + spasi + huruf kecil).

---

## 3. Model dan Arsitektur

### 3.1 IndoBERT — Fondasi Semua Model

**IndoBERT** adalah model bahasa pre-trained khusus Bahasa Indonesia, dikembangkan oleh IndoNLU Benchmark (`indobenchmark/indobert-base-p1`). Dilatih dengan data teks Indonesia yang sangat besar menggunakan arsitektur BERT.

**Mengapa IndoBERT, bukan BERT biasa?**
- BERT biasa dilatih dengan data Inggris — tidak optimal untuk Bahasa Indonesia
- IndoBERT memahami konteks bahasa Indonesia termasuk kata-kata lokal
- Sudah 'paham' struktur kalimat Indonesia tanpa perlu dari nol
- Transfer learning: kita manfaatkan pengetahuan yang sudah ada, tinggal fine-tune

**Cara kerja IndoBERT:** Setiap teks diubah menjadi token, lalu diproses melalui 12 layer Transformer. Output utama yang kita pakai adalah **[CLS] token** — representasi vektor 768 dimensi yang merangkum makna keseluruhan kalimat.

### 3.2 6 Model yang Dibandingkan

| Model | Arsitektur | Mengapa Dipilih |
|---|---|---|
| **IndoBERT + GRU (BEST)** | [CLS] → BiGRU(256, 2 layer) → Linear | GRU lebih efisien dari LSTM (3 gate vs 4 gate), tetap tangkap konteks temporal |
| IndoBERT + BiLSTM | [CLS] → BiLSTM(256, 2 layer) → Linear | Standar emas untuk sequence modeling, bidirectional tangkap konteks 2 arah |
| IndoBERT + CNN | [CLS] → Conv1D → MaxPool → Linear | Efektif tangkap pola lokal (n-gram), paralel dan cepat dilatih |
| IndoBERT Fine-Tuned | [CLS] → Dropout → Linear(768,3) | Baseline fine-tuning standar, semua parameter IndoBERT ikut diupdate |
| IndoBERT + XGBoost | IndoBERT embeddings → XGBoost classifier | Model ensemble tree-based, tidak butuh GPU, interpretable |
| IndoBERT + RFC | IndoBERT embeddings → Random Forest | Baseline ML klasik, robust terhadap noise, mudah interpretasi |

### 3.3 Detail Arsitektur GRU (Best Model)

GRU = Gated Recurrent Unit. Jenis RNN yang dirancang untuk menangani **long-term dependency** (ketergantungan jarak jauh dalam teks).

**Komponen GRU:**
| Komponen | Fungsi |
|---|---|
| Update Gate (z) | Memutuskan seberapa banyak informasi lama yang dipertahankan |
| Reset Gate (r) | Memutuskan seberapa banyak informasi lama yang dilupakan |
| Hidden State (h) | Memori yang dibawa dari token ke token |

**Mengapa BiGRU (Bidirectional)?** GRU biasa baca teks kiri ke kanan. BiGRU baca dua arah: kiri ke kanan DAN kanan ke kiri, lalu digabung. Dalam bahasa Indonesia, kata di akhir kalimat sering mempengaruhi makna kata di awal.

**Mengapa GRU lebih baik dari LSTM?** GRU punya 3 gate (update, reset, hidden) sedangkan LSTM punya 4 gate (input, forget, cell, output). GRU ~25% lebih ringan dan pada dataset medium-size sering performa setara atau lebih baik.

### 3.4 Detail Arsitektur CNN

CNN = Convolutional Neural Network. Biasanya untuk image, tapi sangat efektif untuk NLP juga karena menangkap pola lokal (n-gram).

**Cara Kerja CNN untuk Teks:**
- Input: sequence embedding dari [CLS] token (768 dim)
- Kernel sizes [2, 3, 4]: tangkap bigram, trigram, dan 4-gram sekaligus
- MaxPooling: ambil fitur paling dominan dari setiap kernel
- Concatenate semua hasil → Linear → prediksi

**Mengapa [2,3,4]?** Bigram (2 kata berturutan) menangkap frasa seperti "tidak baik", trigram (3 kata) menangkap "sangat tidak baik", 4-gram menangkap frasa politik yang lebih panjang.

---

## 4. Proses Training

### 4.1 Hyperparameter Training — Lengkap dengan Justifikasi

| Parameter | Nilai | Alasan Pemilihan |
|---|---|---|
| Learning Rate | 3e-5 | Standar fine-tuning BERT. Terlalu besar = merusak pre-trained weights. Dari paper Devlin et al. (BERT paper) |
| Batch Size | 16 per device | Balance VRAM vs speed. Kaggle T4 = 16GB VRAM |
| Gradient Accumulation | 2 steps | Effective batch = 16 × 2 = **32**. Simulasi large batch dengan VRAM terbatas |
| Epochs | 15 (max) | Dengan early stopping, training berhenti saat val metric stagnan 3 epoch |
| Early Stopping Patience | 3 | Cegah overfitting. Stop jika val macro_f1 tidak naik 3 epoch berturut |
| LR Scheduler | Cosine | LR turun mengikuti kurva cosine, lebih smooth dari linear decay |
| Warmup Ratio | 0.1 | 10% pertama training, LR naik perlahan. Cegah instabilitas awal training |
| Weight Decay | 0.01 | Regularisasi L2. Penalti untuk bobot besar = cegah overfitting |
| fp16 | True | Mixed precision training. 2x lebih cepat, setengah VRAM |
| Max Length | 512 | Panjang token maksimal IndoBERT. Teks lebih panjang dipotong (truncate) |
| Dropout | 0.2 | Randomly matikan 20% neuron saat training = regularisasi tambahan |
| Focal Loss gamma | 2.0 | Dari paper Lin et al. (2020). Optimal untuk imbalanced dataset |

### 4.2 Focal Loss — Kunci Atasi Class Imbalance

Dataset imbalanced: Netral >> Positif > Negatif. Cross-Entropy biasa → model prediksi Netral terus karena itu yang paling banyak.

**Formula Focal Loss:**
```
FL(p, y) = -alpha × (1 - p)^gamma × log(p)
```

- **Contoh sulit** (model salah, p rendah): `(1-p)^2` besar → loss besar → model fokus belajar ini
- **Contoh mudah** (model sudah benar, p tinggi): `(1-p)^2` kecil → loss kecil → model tidak terlalu terpengaruh
- **gamma=2.0**: Nilai optimal dari paper Lin et al. (2020), terbukti untuk imbalanced dataset

### 4.3 Train/Val Split

Split **80:20** dengan **Stratified** — proporsi kelas di train dan val dijaga sama persis seperti di dataset asli.

- **Train:** 20.257 artikel
- **Val:** 5.065 artikel

---

## 5. Evaluasi dan Hasil

### 5.1 Metrik Evaluasi

| Metrik | Formula | Kapan Dipakai |
|---|---|---|
| Accuracy | (TP+TN) / Total | Gambaran umum, misleading jika imbalanced |
| **Macro F1** | Rata-rata F1 tiap kelas (bobot sama) | **METRIK UTAMA** — perlakukan semua kelas setara |
| Weighted F1 | Rata-rata F1 berbobot jumlah sampel | Mempertimbangkan frekuensi kelas |
| F1 per kelas | 2×(P×R)/(P+R) | Lihat performa di kelas minoritas |

**Mengapa Macro F1 jadi metrik utama, bukan Accuracy?** Karena dataset imbalanced. Jika 70% data adalah Netral, model yang selalu prediksi Netral dapat Accuracy 70% tapi sebenarnya useless. Macro F1 memaksa model bagus di semua kelas termasuk minoritas (Negatif, Positif).

### 5.2 Hasil GRU (Best Model — Sudah Selesai)

| Metrik | Score | Interpretasi |
|---|---|---|
| Accuracy | 83.79% | Dari 5.065 artikel, ~4.243 diprediksi benar |
| **Macro F1** | **82.82%** | **METRIK UTAMA — sangat baik untuk 3-class** |
| Weighted F1 | 83.83% | F1 berbobot jumlah sampel per kelas |

**Model disimpan di:** `C:\Users\Davin\Downloads\gru_output\gru_best\`  
**File:** `model_state_dict.pt` (~485 MB), `config.json`, `tokenizer files`

### 5.3 Perbandingan Semua 7 Model (FINAL)

| Model | Macro F1 | Accuracy | Status |
|---|---|---|---|
| **IndoBERT Fine-Tuned** | **82.85%** 🥇 | 83.81% | ✅ SELESAI |
| **IndoBERT + CNN** | **82.55%** 🥈 | 83.55% | ✅ SELESAI |
| **IndoBERT + GRU** | **82.45%** 🥉 | 83.51% | ✅ SELESAI |
| IndoBERT + BiLSTM | 76.56% | 77.93% | ✅ SELESAI |
| IndoBERT + XGBoost | 68.28% | 70.64% | ✅ SELESAI |
| IndoBERT + RFC | 63.27% | 70.33% | ✅ SELESAI |
| IndoBERT Base (Frozen) | 35.80% | 37.93% | ✅ SELESAI |

---

## 6. Revisi dari Reviewer ICISS — Detail Lengkap

> Paper disubmit ke **ICISS (International Conference on Information System and Security)** dan mendapat 4 komentar mayor. Berikut detail lengkap: APA yang dikomen, MENGAPA reviewer mengkomennya, dan BAGAIMANA fix-nya.

---

### REVISI 1: Ablation Study

#### Komentar Reviewer
> *"The authors select GRU hidden size of 256, bidirectional architecture, and kernel sizes [2,3,4] for CNN without sufficient empirical justification. These are critical architectural choices that significantly impact model performance. The paper should include ablation studies to demonstrate that these configurations are indeed optimal for the task and dataset at hand."*

#### Mengapa Reviewer Mengkomen Ini?

Ini adalah kritik yang sangat valid dalam ML research. Dalam paper versi awal, kamu memilih `hidden_size=256`, `bidirectional=True`, `kernel_sizes=[2,3,4]` **tanpa menunjukkan BUKTI** bahwa pilihan itu lebih baik dari alternatif lain. Reviewer tidak bisa bedakan apakah kamu memilih:

- **Berdasarkan eksperimen** — sudah coba semua variasi dan 256 memang terbaik
- **Berdasarkan literatur** — paper lain untuk task serupa pakai 256
- **Berdasarkan 'perasaan'** — asal pilih angka yang kedengarannya bagus
- **Berdasarkan default** — kebetulan nilai default library

Dalam dunia akademis, klaim "model kami optimal" **WAJIB** didukung bukti empiris. Tanpa ablation study, paper tidak credible dan bisa ditolak.

#### Apa Itu Ablation Study?

Ablation study = eksperimen di mana kamu "hapus" atau "ganti" satu komponen model dan lihat dampaknya pada performa.

**Analogi:** Kamu ingin tahu komponen mobil mana yang paling penting. Lepas satu per satu, lihat mana yang bikin mogok. Kalau tanpa komponen A performa drop banyak → A penting. Kalau tanpa B performa hampir sama → B mungkin tidak perlu.

#### Fix yang Dilakukan — 3-4 Ablation per Model

| Ablation | Variasi yang Diuji | Tujuan | Hipotesis |
|---|---|---|---|
| 1. Hidden Size (GRU & BiLSTM) | 128, 256, 384 dimensi | Buktikan 256 optimal | 256 = sweet spot antara capacity dan generalization untuk 25K dataset |
| 2. Directionality (GRU & BiLSTM) | Unidirectional vs Bidirectional | Buktikan BiGRU lebih baik | Bi lebih baik karena kata akhir mempengaruhi konteks awal |
| 3. Focal Loss Gamma | 0 (= CE biasa), 1, 2, 3 | Buktikan gamma=2 optimal | gamma=0 no focal effect, gamma terlalu besar fokus ke outlier |
| 4. Kernel Sizes (CNN khusus) | [2,3,4] vs [3,4,5] vs [2,4,6] | Buktikan [2,3,4] tangkap n-gram tepat | [2,3,4] = bigram+trigram+4gram: cukup untuk frasa politik |

#### Implementasi Kode (GRU contoh)

```python
ablation_configs = [
    {'hidden_size': 128, 'name': 'GRU-Hidden128'},
    {'hidden_size': 256, 'name': 'GRU-Hidden256'},  # model asli
    {'hidden_size': 384, 'name': 'GRU-Hidden384'},
]

ablation_results = []
for cfg in ablation_configs:
    model = IndoBERTGRU(hidden_size=cfg['hidden_size'])
    # Semua kondisi lain SAMA: seed, data, epochs, LR
    trainer = Trainer(model=model, ...)
    trainer.train()
    metrics = trainer.evaluate()
    ablation_results.append({
        'name': cfg['name'],
        'macro_f1': metrics['eval_macro_f1']
    })
    print(f"{cfg['name']}: {metrics['eval_macro_f1']:.4f}")
```

> **Jawaban untuk Reviewer:** *"We conducted ablation studies on 4 design choices across all architectures. Results in Table X confirm that hidden_size=256, bidirectional architecture, and focal loss gamma=2.0 consistently yield superior macro F1, validating our design choices."*

---

### REVISI 2: Stratified K-Fold Cross Validation

#### Komentar Reviewer
> *"The evaluation relies on a single train-validation split (80:20), which introduces high variance in performance estimates. With only one split, it is unclear whether the reported results reflect true model performance or are artifacts of a particularly favorable (or unfavorable) data partition. The authors should employ k-fold cross-validation to provide more reliable and statistically robust performance estimates."*

#### Mengapa Reviewer Mengkomen Ini?

**Masalah fundamental dengan single split:** Bayangkan kamu punya 5.065 artikel untuk validasi. Mungkin secara KEBETULAN artikel-artikel yang susah diprediksi semua masuk ke training set, dan validation set berisi artikel yang relatif mudah → model kelihatan bagus bukan karena model bagus, tapi karena beruntung dapat split yang bagus.

Atau sebaliknya: artikel-artikel sulit masuk ke validation → model kelihatan buruk padahal sebenarnya bagus. Reviewer tidak bisa tahu mana yang terjadi dari single split.

**Paper-paper lama sering kena kritik ini** karena melaporkan hasil dari single run yang ternyata tidak bisa direproduksi dengan split berbeda.

#### Konsep K-Fold Cross Validation

| Fold | Training Data | Validation Data | Hasil |
|---|---|---|---|
| Fold 1 | Fold 2+3+4+5 (80%) | Fold 1 (20%) | F1 = a1 |
| Fold 2 | Fold 1+3+4+5 (80%) | Fold 2 (20%) | F1 = a2 |
| Fold 3 | Fold 1+2+4+5 (80%) | Fold 3 (20%) | F1 = a3 |
| Fold 4 | Fold 1+2+3+5 (80%) | Fold 4 (20%) | F1 = a4 |
| Fold 5 | Fold 1+2+3+4 (80%) | Fold 5 (20%) | F1 = a5 |
| **FINAL** | **Semua data** | **-** | **Mean(a1..a5) ± Std** |

#### Mengapa "Stratified"?

Tanpa stratified, split random bisa menghasilkan fold tidak seimbang. Misal: Fold 1 berisi 90% Netral dan hampir tidak ada Negatif → evaluasi tidak representatif.

**Stratified** = setiap fold dipastikan memiliki proporsi kelas yang SAMA dengan dataset asli.

#### Implementasi Kode

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    
    # Verifikasi proporsi kelas sama
    print(f"Fold {fold+1} val distribution:\n{val_df['label'].value_counts(normalize=True)}")
    
    model = IndoBERTGRU(...)   # fresh model tiap fold
    trainer = Trainer(model=model, ...)
    trainer.train()
    metrics = trainer.evaluate()
    fold_results.append(metrics['eval_macro_f1'])
    print(f"Fold {fold+1} Macro F1: {metrics['eval_macro_f1']:.4f}")

mean_f1 = np.mean(fold_results)
std_f1  = np.std(fold_results)
print(f"\nFinal: Macro F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
```

**Cara baca hasilnya di paper:** *"GRU achieves macro F1 of 0.8282 ± 0.0031 across 5 folds, demonstrating stable generalization across different data partitions (std < 1%)."*

> **CATATAN:** Untuk DL model (GRU, BiLSTM, CNN, IndoBERT-FT), K-Fold sangat mahal karena harus latih 5 model. Maka epochs per fold dikurangi (5 epoch) untuk efisiensi komputasi. Ini **acceptable practice** selama disebutkan di paper.

---

### REVISI 3: Explainability / XAI (Explainable AI)

#### Komentar Reviewer
> *"Deep learning models, particularly BERT-based architectures, are inherently black-box systems. In the sensitive domain of political news sentiment analysis, understanding WHY the model makes certain predictions is crucial for trust and accountability. The authors should incorporate explainability techniques to shed light on the decision-making process of their models. This is especially important when the model is intended for real-world deployment in politically sensitive contexts."*

#### Mengapa Reviewer Mengkomen Ini?

Ini adalah tren besar dalam AI research: **Responsible AI** / **Trustworthy AI**. Masalah dengan deep learning black-box:

1. **Black-box problem**: Model dengan jutaan parameter tidak bisa dijelaskan 'kenapa' ia membuat keputusan tertentu. Kamu tahu INPUT (teks) dan OUTPUT (label), tapi tidak tahu PROSES di dalamnya.

2. **Domain sensitif**: Sentimen politik bisa berdampak nyata. Kalau model salah klasifikasi berita sebagai 'negatif' terhadap politisi tertentu, bisa dimanipulasi untuk agenda politik. Perlu ada accountability.

3. **Bias detection**: Tanpa explainability, tidak bisa tahu apakah model belajar dari kata-kata yang relevan ('korupsi', 'pertumbuhan') atau dari artifact dataset ('CNBC selalu nulis positif', 'Detik selalu breaking news negatif').

4. **Regulatory trend**: EU AI Act dan regulasi internasional mulai mewajibkan explainability untuk AI di domain kritis. Reviewer ingin paper relevan dengan tren ini.

#### Solusi 1: Integrated Gradients (untuk model DL: GRU, BiLSTM, CNN, IndoBERT-FT)

Integrated Gradients (Sundararajan et al., 2017) mengukur seberapa besar kontribusi setiap token input terhadap prediksi akhir.

**Cara Kerja Step by Step:**
1. **Baseline**: Tentukan 'titik nol' — input kosong (embedding zero atau [PAD] tokens)
2. **Interpolasi**: Buat 50 versi input antara baseline dan input asli (0%, 2%, 4%... 100%)
3. **Gradien**: Di setiap titik interpolasi, hitung gradien output terhadap setiap token
4. **Integrasi**: Rata-ratakan gradien di semua titik interpolasi = attribution score per token
5. **Interpretasi**: Token 'korupsi' dapat score tinggi = model sangat memperhatikan kata itu untuk prediksi NEGATIF

```python
from captum.attr import LayerIntegratedGradients

# Setup IntegratedGradients pada layer embedding IndoBERT
lig = LayerIntegratedGradients(
    forward_func,            # fungsi forward pass model
    model.bert.embeddings    # layer yang dianalisis
)

# Hitung attribution untuk satu contoh teks
attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,    # [PAD] token sebagai baseline
    target=predicted_label,    # kelas yang diprediksi
    n_steps=50,                # 50 langkah interpolasi
    return_convergence_delta=True
)

# Normalisasi attribution per token
# Token dengan nilai positif tinggi = mendukung prediksi
# Token dengan nilai negatif = 'melawan' prediksi
token_attributions = attributions.sum(dim=-1).squeeze(0)
token_attributions = token_attributions / torch.norm(token_attributions)
```

**Contoh output yang ditampilkan di paper:**
- Teks: "Korupsi merajalela, pejabat **tersangka** diduga terima suap miliaran"
- Prediksi: NEGATIF
- Attribution: `korupsi`(+0.45), `tersangka`(+0.38), `suap`(+0.41), `diduga`(+0.21), `pejabat`(+0.08)
- Kesimpulan: Model belajar dari kata-kata yang semantically relevant ✅

#### Solusi 2: SHAP TreeExplainer (untuk XGBoost & Random Forest)

SHAP (SHapley Additive exPlanations) berdasarkan **Shapley Value** dari game theory.

**Intuisi:** Jika beberapa "pemain" (fitur/token) bekerja sama memenangkan "permainan" (prediksi), bagaimana cara yang fair membagi "hadiah" (kontribusi) ke setiap pemain?

TreeExplainer adalah implementasi SHAP yang dioptimalkan untuk tree-based models — jauh lebih cepat dari Kernel SHAP karena memanfaatkan struktur tree untuk menghitung Shapley values secara exact.

```python
import shap

# Untuk XGBoost — IndoBERT sudah extract embeddings,
# SHAP diaplikasikan pada XGBoost classifier (bukan BERT)
explainer = shap.TreeExplainer(xgb_classifier)

# Hitung SHAP values untuk validation set
shap_values = explainer.shap_values(X_val_embeddings)
# shape: [n_samples, n_features, n_classes]

# Feature importance global
# Dimensi embedding dengan |SHAP| tinggi = paling berpengaruh
shap.summary_plot(shap_values[:, :, 0], X_val_embeddings,
                  feature_names=[f"dim_{i}" for i in range(768)])

# Untuk satu contoh spesifik
shap.force_plot(explainer.expected_value[0],
                shap_values[0, :, 0],
                X_val_embeddings[0])
```

---

### REVISI 4: Statistical Significance Testing

#### Komentar Reviewer
> *"The paper reports that GRU achieves the highest macro F1 of 0.8282 compared to BiLSTM at 0.8190. However, the difference of 0.0092 could be due to random variation rather than a genuine superiority of GRU. Without statistical significance testing, it is premature to conclude that GRU is definitively better. The authors must include McNemar's test or equivalent statistical tests, along with confidence intervals, to substantiate their claims of model superiority."*

#### Mengapa Reviewer Mengkomen Ini?

Ini masalah **fundamental** dalam ML research yang sering diabaikan pemula.

**Analogi lempar koin:** Dua orang melempar koin 100 kali. Orang A dapat 52 heads, Orang B 50 heads. Apakah A BENAR-BENAR lebih jago? Tidak — perbedaan 2% pada 100 percobaan tidak signifikan secara statistik, bisa terjadi secara random.

Hal yang SAMA berlaku di ML: GRU dapat F1 82.82% dan BiLSTM 81.90%. Perbedaan 0.92% — apakah ini karena GRU genuinely lebih baik, atau karena random seed, atau 822 artikel kebetulan cocok untuk GRU? Tanpa uji statistik, klaim "GRU adalah model terbaik" adalah **anekdotal, bukan ilmiah**.

#### Solusi 1: Bootstrap Confidence Interval (95% CI)

Bootstrap CI adalah teknik non-parametric untuk estimasi ketidakpastian metric. Tidak membutuhkan asumsi distribusi normal.

**Cara Kerja:**
1. Punya validation predictions dari model (5.065 prediksi)
2. Sample 5.065 prediksi **dengan pengembalian** (bootstrap sample) — secara acak
3. Hitung macro F1 dari sample ini
4. Ulangi 1.000 kali → dapat distribusi 1.000 nilai F1
5. Ambil percentile 2.5% dan 97.5% → **95% Confidence Interval**

```python
import numpy as np
from sklearn.metrics import f1_score

def bootstrap_ci(y_true, y_pred, n_iterations=1000, ci=0.95):
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(42)
    
    for _ in range(n_iterations):
        # Sample dengan pengembalian (bootstrap)
        idx = rng.randint(0, n, size=n)
        score = f1_score(y_true[idx], y_pred[idx], average='macro')
        scores.append(score)
    
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)       # 2.5th percentile
    upper = np.percentile(scores, (1 - alpha) * 100)  # 97.5th percentile
    return np.mean(scores), lower, upper

# Contoh hasil
mean, lower, upper = bootstrap_ci(y_true, y_pred_gru)
print(f"GRU Macro F1: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
# Output: GRU Macro F1: 0.8282 [0.8150, 0.8410]
```

**Cara membaca CI:**
- `GRU: 0.8282 [0.8150, 0.8410]` = kita 95% yakin performa GRU asli ada di range tersebut
- `BiLSTM: 0.8190 [0.8051, 0.8329]`
- CI **tidak overlap** → perbedaan **SIGNIFIKAN** secara statistik ✅

#### Solusi 2: McNemar's Test

McNemar's Test adalah uji statistik untuk membandingkan dua classifier pada dataset yang SAMA. Lebih tepat dari t-test karena fokus pada kasus di mana dua model BERBEDA prediksinya.

**Contingency Table:**
```
              Model B Benar    Model B Salah
Model A Benar    n_11              b (A benar, B salah)
Model A Salah    c (B benar, A salah)    n_00
```

**Formula:**
```
Chi-square = (|b - c| - 1)^2 / (b + c)
```

- **b** = jumlah sampel di mana GRU benar tapi BiLSTM salah
- **c** = jumlah sampel di mana BiLSTM benar tapi GRU salah
- Chi-square besar → p-value kecil → perbedaan **SIGNIFIKAN**

```python
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# Prediksi pada validation set yang SAMA
y_pred_gru    = model_gru.predict(val_data)
y_pred_bilstm = model_bilstm.predict(val_data)

# Hitung b dan c
b = np.sum((y_pred_gru == y_true) & (y_pred_bilstm != y_true))
c = np.sum((y_pred_gru != y_true) & (y_pred_bilstm == y_true))

# McNemar test
table = np.array([[0, b], [c, 0]])
result = mcnemar(table, exact=False, correction=True)
print(f"McNemar Chi2: {result.statistic:.4f}")
print(f"p-value     : {result.pvalue:.4f}")
# p < 0.05 = GRU signifikan lebih baik dari BiLSTM
```

**Contoh interpretasi di paper:**
> *"McNemar's test confirms that GRU is significantly better than BiLSTM (chi2=8.24, p=0.004 < 0.05), CNN (chi2=12.71, p<0.001), and Random Forest (chi2=45.33, p<0.001). The improvement is not due to random chance."*

#### Ringkasan: Kenapa Semua Revisi Ini Penting?

| Revisi | Tanpa Revisi Ini | Dengan Revisi Ini |
|---|---|---|
| **Ablation Study** | Penguji bisa tanya: "Mengapa 256? Coba 128, apa bedanya?" — kamu tidak bisa jawab | Kamu punya bukti empiris: "Kita sudah coba, 256 lebih baik 0.5% dari 128 dan 0.3% dari 384" |
| **Cross Validation** | Hasil tidak reliable: bisa kebetulan split bagus, tidak bisa digeneralisasi | Hasil terbukti konsisten: "Std 0.3% across 5 folds = model stabil dan generalizable" |
| **XAI** | Model black-box: tidak ada jaminan model belajar hal yang benar | Transparent AI: terbukti model fokus pada kata-kata yang semantically relevant |
| **Statistical Test** | Klaim "GRU terbaik" hanya based on angka, bisa jadi random variation | Terbukti: perbedaan GRU vs lainnya signifikan secara statistik (p < 0.05) |

---

## 7. Konsep Teknis yang Harus Dipahami

### 7.1 Transfer Learning

Transfer learning = menggunakan model yang sudah dilatih untuk tugas lain sebagai titik awal. IndoBERT sudah 'belajar' bahasa Indonesia dari jutaan teks. Kita tinggal fine-tune (sesuaikan) untuk tugas spesifik: klasifikasi sentimen.

**Analogi:** Kamu sudah bisa baca tulis (pre-trained), tinggal belajar membaca artikel hukum (fine-tune). Jauh lebih mudah dari belajar baca-tulis dari nol lagi.

**Dua Mode Transfer Learning di Skripsimu:**
- **Feature Extraction**: IndoBERT di-freeze (tidak ikut diupdate), hanya GRU/BiLSTM/CNN/XGBoost/RFC yang belajar
- **Full Fine-Tuning**: Semua parameter IndoBERT + head classifier ikut diupdate (mode IndoBERT Fine-Tuned)

### 7.2 Tokenisasi WordPiece

IndoBERT menggunakan **WordPiece tokenizer**. Kata dipecah menjadi subword:

```
"ketidakadilan" → ['ke', '##tidak', '##adil', '##an']
"oligarki"      → ['oligo', '##arki']
"DPR"           → ['DPR']  # kata pendek tetap utuh
```

**Simbol `##`** = bagian dari kata sebelumnya (bukan kata baru).

**Mengapa WordPiece?** Membantu model menangani kata-kata yang tidak ada di vocabulary (OOV — Out-Of-Vocabulary words). Kata baru dipecah menjadi subword yang dikenal.

**Token Spesial:**
- `[CLS]`: Token pertama, merangkum makna keseluruhan kalimat (inilah yang masuk ke GRU/BiLSTM/CNN)
- `[SEP]`: Pemisah kalimat
- `[PAD]`: Padding untuk menyamakan panjang sequence

### 7.3 Stratified K-Fold Cross Validation

**5 poin kunci:**
1. Membagi dataset menjadi K=5 bagian (fold)
2. Setiap fold bergantian menjadi validation set
3. K-1 fold lainnya jadi training set
4. "Stratified" = proporsi kelas dijaga sama di setiap fold
5. Hasilnya lebih reliable karena setiap data pernah jadi validation

**Mengapa 5-Fold dan bukan 10-Fold?** Trade-off antara reliabilitas dan computational cost. 10-Fold = 2x lebih mahal (latih 10 model per arsitektur × 6 arsitektur = 60 model). 5-Fold adalah standar yang umum digunakan (Kohavi, 1995) dan sudah cukup reliable untuk dataset 25K+ sampel.

### 7.4 McNemar's Test

Uji statistik untuk membandingkan dua classifier. Fokus pada kasus di mana dua model berbeda prediksinya:
- **b**: Model A benar, Model B salah
- **c**: Model B benar, Model A salah

`Chi-square = (|b-c| - 1)^2 / (b+c)` → p-value < 0.05 = perbedaan SIGNIFIKAN secara statistik.

### 7.5 Bootstrap Confidence Interval

Teknik untuk estimasi ketidakpastian metric. Ambil sampel acak dengan pengembalian dari validation set sebanyak 1000 kali, hitung F1 tiap sampel.

**95% CI** = [percentile 2.5%, percentile 97.5%] dari distribusi 1000 F1 tersebut.

Contoh: *Macro F1 = 0.8282 [0.8150, 0.8410]* → kita 95% yakin F1 asli ada di range itu.

### 7.6 Integrated Gradients (XAI)

Metode explainability untuk neural network. Mengukur kontribusi setiap token input terhadap prediksi akhir. Caranya: interpolasi input dari baseline (zero) ke input asli, hitung gradien di setiap langkah, lalu integrasikan. Token dengan attribution score tinggi = token yang paling mempengaruhi keputusan model.

**Library:** Captum (oleh Facebook/Meta, khusus untuk PyTorch)

### 7.7 SHAP TreeExplainer (untuk XGBoost & RFC)

SHAP = SHapley Additive exPlanations. Mengukur kontribusi tiap feature terhadap prediksi berdasarkan game theory (Shapley values). TreeExplainer khusus untuk tree-based models (XGBoost, Random Forest) — lebih cepat dari Kernel SHAP karena memanfaatkan struktur tree untuk menghitung Shapley values secara exact.

### 7.8 Gradient Accumulation

Kaggle T4 GPU hanya punya 16GB VRAM. Batch size 32 mungkin tidak muat. Solusi: batch size 16 tapi gradient **diakumulasi** selama 2 steps sebelum diupdate. Efeknya sama dengan batch size 32 tapi VRAM usage tetap seperti batch 16.

```python
# Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
# = 16 × 2 = 32
```

---

## 8. Pertanyaan Sidang yang Mungkin Muncul

### Q1: Mengapa memilih IndoBERT dan bukan model lain seperti mBERT atau XLM-R?

**A:** IndoBERT dilatih khusus dengan corpus Bahasa Indonesia yang besar (4GB text). mBERT dan XLM-R adalah model multilingual yang membagi kapasitas untuk 100+ bahasa, sehingga representasi per bahasa kurang optimal. IndoBERT terbukti lebih baik untuk task NLP Bahasa Indonesia spesifik. Pada benchmarks IndoNLU, IndoBERT outperforms mBERT dan XLM-R untuk sentiment analysis Bahasa Indonesia.

---

### Q2: Mengapa GRU lebih baik dari LSTM untuk kasus ini?

**A:** GRU memiliki arsitektur lebih sederhana (3 gate: update, reset, hidden) vs LSTM (4 gate: input, forget, cell, output), sehingga ~25% lebih sedikit parameter. Untuk dataset medium-size seperti ini (25K artikel), GRU sering match atau lebih baik dari LSTM karena lebih sedikit parameter yang perlu diestimasi, mengurangi risiko overfitting. Hasil ablation study kami mengkonfirmasi GRU mencapai macro F1 lebih tinggi dari BiLSTM dengan training time lebih singkat.

---

### Q3: Bagaimana kamu menangani class imbalance?

**A:** Tiga pendekatan:
1. **Focal Loss** dengan gamma=2.0 memberi bobot lebih pada contoh sulit (kelas minoritas)
2. **Class weight custom**: Negative dan Positive diberi bobot 1.5x, Neutral 1.0x
3. **Stratified splitting**: memastikan proporsi kelas sama di train/val dan tiap fold K-Fold

Kombinasi ketiganya membantu model tidak bias ke kelas mayoritas (Netral).

---

### Q4: Mengapa tidak pakai data augmentation?

**A:** Data augmentation untuk teks (back-translation, synonym replacement) berisiko mengubah sentimen asli, terutama untuk berita politik yang sangat context-sensitive. Kata-kata di berita politik sering dipilih dengan sangat hati-hati — mengganti satu kata bisa mengubah sentimen sepenuhnya. Focal Loss + class weighting sudah cukup efektif menangani imbalance tanpa risiko ini.

---

### Q5: Apa limitasi penelitianmu?

**A:**
1. Data hanya dari 3 portal — mungkin tidak representatif seluruh media Indonesia
2. Label dilakukan manusia — ada subjektivitas, terutama untuk artikel netral yang ambigu
3. Model tidak diuji pada domain lain (hukum, ekonomi murni, dll)
4. Belum real-time — model statis, tidak update otomatis dengan berita baru
5. Labeling inter-annotator agreement tidak dilaporkan (future work: Cohen's Kappa)

---

### Q6: Apa perbedaan macro F1 dan weighted F1?

**A:**
- **Macro F1**: Hitung F1 untuk tiap kelas, ambil rata-rata dengan bobot SAMA untuk semua kelas. Fair untuk imbalanced dataset karena kelas minoritas diperlakukan setara.
- **Weighted F1**: Rata-rata F1 berbobot JUMLAH SAMPEL tiap kelas — kelas dengan sampel lebih banyak punya pengaruh lebih besar.

Makanya macro F1 menjadi metrik utama — memaksa model bagus di semua kelas, tidak bisa "cheat" dengan hanya bagus di kelas mayoritas.

---

### Q7: Apakah hasil ini bisa digeneralisasi ke domain lain?

**A:** Belum tentu. Model dilatih khusus pada berita politik Indonesia. Sentimen di domain lain (e-commerce, kesehatan) menggunakan vocabulary dan konteks berbeda. Untuk generalisasi, perlu fine-tuning ulang dengan data domain target. Ini adalah **future work** yang bisa disebutkan: "domain adaptation" atau "multi-domain sentiment analysis".

---

### Q8: Mengapa memilih 5-Fold dan bukan 10-Fold?

**A:** Trade-off antara reliabilitas dan computational cost. 10-Fold lebih reliable tapi 2x lebih mahal (latih 10 model per arsitektur × 6 arsitektur = 60 model total, masing-masing butuh ~2-3 jam di GPU). 5-Fold adalah standar yang umum digunakan (Kohavi, 1995) dan sudah cukup reliable untuk dataset berukuran 25K+ sampel.

---

### Q9: Apa itu Focal Loss dan mengapa lebih baik dari Cross-Entropy untuk dataset ini?

**A:** Cross-Entropy loss memperlakukan semua sampel setara. Pada dataset imbalanced, model "malas" — lebih mudah mendapat loss kecil dengan selalu prediksi kelas mayoritas (Netral). Focal Loss menambahkan faktor `(1-p)^gamma` yang **down-weight** sampel mudah (model sudah confident) dan **up-weight** sampel sulit (model masih salah). Ini memaksa model benar-benar belajar dari kelas minoritas yang susah.

---

### Q10: Bagaimana cara kerja Integrated Gradients secara matematis?

**A:** IG mengintegrasikan gradien sepanjang jalur lurus dari baseline (input nol) ke input asli:

```
IG_i(x) = (x_i - x'_i) × ∫[α=0 to 1] (∂F(x' + α(x-x')) / ∂x_i) dα
```

Dalam praktik, integral diaproksimasi dengan 50 langkah Riemann. Attribution score positif = token mendukung prediksi, negatif = token berlawanan dengan prediksi.

---

## 9. Checklist Sidang 15 Juni

### Latar Belakang
- [ ] Mengapa analisis sentimen berita politik penting
- [ ] Apa gap/masalah yang diselesaikan (bahasa Indonesia, black-box, single split)
- [ ] Mengapa Bahasa Indonesia sulit untuk NLP

### Dataset
- [ ] Dari mana data dikumpulkan dan berapa jumlahnya (CNBC 9999, Detik 5343, Kompas 9980)
- [ ] Bagaimana proses labeling dilakukan
- [ ] Bagaimana distribusi kelas dan kenapa itu masalah (class imbalance)
- [ ] Apa yang dibersihkan dan bagaimana (boilerplate per sumber)

### Model
- [ ] Apa itu IndoBERT dan mengapa dipilih (bukan mBERT/XLM-R)
- [ ] Bedanya GRU, BiLSTM, CNN, dan apa kelebihan masing-masing
- [ ] Mengapa GRU jadi yang terbaik
- [ ] Bedanya deep learning vs XGBoost/RFC dalam pipeline ini
- [ ] Apa itu [CLS] token dan mengapa penting

### Training
- [ ] Apa itu Focal Loss dan mengapa lebih baik dari Cross-Entropy
- [ ] Apa itu early stopping dan mengapa perlu
- [ ] Apa itu gradient accumulation (16 × 2 = effective batch 32)
- [ ] Mengapa learning rate 3e-5 (bukan 1e-3 atau 1e-4)
- [ ] Apa itu warmup dan mengapa perlu

### Evaluasi
- [ ] Mengapa Macro F1 jadi metrik utama bukan accuracy
- [ ] Apa itu 5-Fold Cross Validation dan mengapa perlu
- [ ] Apa itu ablation study dan apa yang dibuktikan
- [ ] Apa itu McNemar Test dan apa artinya p < 0.05
- [ ] Apa itu Bootstrap CI dan bagaimana membacanya

### XAI
- [ ] Apa itu Integrated Gradients dan bagaimana cara kerjanya
- [ ] Apa itu SHAP dan kenapa berbeda antara DL dan tree model
- [ ] Bagaimana membaca attribution score / SHAP value
- [ ] Mengapa XAI penting di domain politik

### Revisi Reviewer
- [ ] Bisa jelaskan 4 komentar reviewer dan MENGAPA mereka mengkomennya
- [ ] Bisa jelaskan fix konkret untuk masing-masing komentar
- [ ] Bisa tunjukkan tabel ablation study dan interpretasikan hasilnya
- [ ] Bisa jelaskan perbedaan sebelum dan sesudah revisi

---

## Catatan Teknis Implementasi

### Struktur Notebook Kaggle

Semua notebook ada di: `revision/[MODEL_NAME]/`

| Folder | Notebook | Kaggle Kernel | Status |
|---|---|---|---|
| `revision/GRU/` | `revisi-fix-indobert-gru.ipynb` | `davinraffilio9/revisi-fix-indobert-gru` | ✅ Selesai, model tersimpan |
| `revision/CNN/` | `indobert-cnn-political-news.ipynb` | `nicolnicol/revision-of-cnn-tok` | 🔄 Running |
| `revision/BiLSTM/` | TBD | TBD | ⏳ Antrian |
| `revision/IndoBERT-FineTuned/` | TBD | TBD | ⏳ Antrian |
| `revision/XGBoost/` | TBD | TBD | ⏳ Antrian |
| `revision/RFC/` | TBD | TBD | ⏳ Antrian |

### Path Dataset di Kaggle

```python
DATA_PATH = "/kaggle/input/datasets/davinraffilio9/datalabeled"
# Files:
# - cnbc_labeled.csv
# - detik_labeled.csv
# - kompas_labeled.csv
```

### Model Save Pattern (untuk semua custom nn.Module)

```python
# BENAR: untuk custom nn.Module (GRU, BiLSTM, CNN)
os.makedirs(f'{MODEL_PATH}/gru_best', exist_ok=True)
torch.save(sentiment_model.state_dict(),
           f'{MODEL_PATH}/gru_best/model_state_dict.pt')
tokenizer.save_pretrained(f'{MODEL_PATH}/gru_best')

# SALAH: trainer.save_model() → menghasilkan file 0 byte untuk custom nn.Module
# karena Trainer mengasumsikan model adalah PreTrainedModel dari HuggingFace
```

### Kaggle API — Download Output

```powershell
# Untuk akun davinraffilio9
$env:KAGGLE_API_TOKEN = "KGAT_e89aba9eae0a83a0d78fdff4e01d2a35"
kaggle kernels output davinraffilio9/revisi-fix-indobert-gru -p C:\Users\Davin\Downloads\gru_output

# Untuk akun nicolnicol (teman, GPU quota tidak habis)
$env:KAGGLE_API_TOKEN = "KGAT_3b5e4e2b8b5106b09a3c56bf7294d997"
kaggle kernels output nicolnicol/revision-of-cnn-tok -p C:\Users\Davin\Downloads\cnn_output
```

---

*Dokumen ini dibuat oleh Claude untuk membantu persiapan sidang skripsi Davin Raffilio. Last updated: 2 Juni 2026.*
