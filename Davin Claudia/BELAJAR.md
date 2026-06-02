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

## SESI 1: IMPORT LIBRARY

### `torch` vs `torch.nn` vs `torch.nn.functional`

**Analogi: Dapur masak**

| Library | Analogi | Fungsi |
|---|---|---|
| `torch` | Dapur itu sendiri | Operasi dasar, bikin tensor, pindah ke GPU |
| `torch.nn` | Peralatan masak jadi | Layer siap pakai: GRU, Linear, Dropout |
| `torch.nn.functional` (F) | Resep masakan | Fungsi langsung dijalankan: relu, softmax, cross_entropy |

**Bedanya `nn` vs `F`:**
- `nn.ReLU()` = beli alat, simpen di model, pakai berkali-kali
- `F.relu(x)` = langsung jalankan fungsi sekarang, tidak perlu simpen

---

### `Dataset` vs `DataLoader`

**Analogi: Lemari kartu ujian vs Asisten**

| | Analogi | Fungsi |
|---|---|---|
| `Dataset` | Lemari kartu ujian | Simpan & akses data satu per satu. Wajib punya `__len__` dan `__getitem__` |
| `DataLoader` | Asisten yang ambil kartu batch | Ambil 16 artikel sekaligus, acak urutan, paralel loading |

**Kenapa dari torch?**
Karena semua komponen (Dataset → DataLoader → model → loss) harus dalam satu ekosistem PyTorch agar bisa ngobrol dalam format tensor yang sama. Tidak perlu konversi bolak-balik.

---

### `AutoTokenizer` vs `AutoModel`

**Analogi: Kamus vs Otak**

| | Analogi | Fungsi |
|---|---|---|
| `AutoTokenizer` | Kamus | Ubah teks → angka (token IDs) |
| `AutoModel` | Otak IndoBERT | Token IDs → vektor bermakna 768 dimensi |

**Kenapa prefix `Auto`?**
HuggingFace otomatis deteksi arsitektur yang tepat dari nama model. Tidak perlu tau persis kelas yang dipakai.

```python
# Contoh tokenizer:
tokenizer("korupsi merajalela")
# → {'input_ids': [101, 2983, 4521, 102], 'attention_mask': [1,1,1,1]}
```

---

## SESI 2: KONFIGURASI

### `SEED = 42` dan `set_seed()`

**Analogi: Mesin permen karet dengan nomor tombol**

Komputer tidak bisa random beneran — dia pakai rumus matematika yang butuh titik awal (= SEED). Seed sama → urutan random sama → hasil training selalu bisa direproduksi.

```python
# TANPA seed:
# Run Senin: F1 = 82%
# Run Selasa: F1 = 79%  ← beda! Tidak bisa direproduksi

# DENGAN set_seed(42):
# Run Senin: F1 = 82%
# Run Selasa: F1 = 82%  ← selalu sama! ✅
```

**`set_seed(42)` dari HuggingFace set 4 seed sekaligus:**
```python
random.seed(42)           # Python random
np.random.seed(42)        # NumPy random
torch.manual_seed(42)     # PyTorch CPU
torch.cuda.manual_seed(42) # PyTorch GPU
```

**Kenapa angka 42?**
Meme dari novel "Hitchhiker's Guide to the Galaxy" — 42 = jawaban atas segala pertanyaan di alam semesta 😄. Tidak ada alasan matematis khusus, angka apapun valid.

✅ **Yang dipakai:** `SEED = 42`
❌ **Alternatif:** Tidak pakai seed → hasil tidak reproducible

🎤 **Jawab dosen:**
> *"Kami menggunakan random seed 42 untuk memastikan reproducibility penelitian. Dengan seed yang sama, seluruh proses menghasilkan hasil yang identik setiap kali dijalankan. Ini penting untuk validasi dan replikasi hasil penelitian."*

---

### `MAX_LENGTH = 512`

**Analogi: Meja belajar yang cuma muat 512 buku**

512 adalah **hard limit arsitektur BERT** — bukan pilihan, tapi batasan teknis yang sudah dikunci sejak BERT dibuat. Tidak bisa diubah tanpa ganti arsitektur.

```python
# Artikel pendek (100 token) → aman, muat
# Artikel panjang (800 token) → dipotong jadi 512, sisanya DIBUANG ✂️
```

**Kenapa tidak pakai lebih kecil (misal 256)?**
```
MAX_LENGTH = 256 → teks terpotong di tengah kalimat
                 → model tidak baca konteks akhir artikel
                 → bisa salah prediksi sentimen
```

✅ **Yang dipakai:** 512 (maksimal yang tersedia)
❌ **Alternatif:**
- 256 → terlalu banyak konteks hilang
- 1024 → tidak bisa! BERT keras di 512
- Longformer (4096) → tidak ada versi IndoBERT, perlu retrain dari nol

🎤 **Jawab dosen:**
> *"MAX_LENGTH 512 adalah batas maksimal yang didukung arsitektur BERT. Kami menggunakan nilai maksimal ini untuk memastikan model membaca konteks artikel selengkap mungkin."*

**Kalau ditanya kenapa tidak pakai model yang support >512:**
> *"Ada model seperti Longformer yang support hingga 4096 token, namun tidak tersedia versi pre-trained untuk Bahasa Indonesia. Menggunakannya akan mengorbankan kualitas representasi Bahasa Indonesia yang sudah dioptimalkan IndoBERT."*

---

### `GRU_HIDDEN = 256`

**Analogi: Ukuran buku catatan GRU**

Hidden size = seberapa banyak informasi yang bisa "diingat" GRU dalam satu waktu.

```
Hidden = 128 → buku catatan kecil → underfitting (lupa terlalu banyak)
Hidden = 256 → buku catatan sedang → OPTIMAL ✅
Hidden = 384 → buku catatan tebal → overfitting (hafal, tidak generalisasi)
```

**Dibuktikan dengan ablation study:**
| Hidden Size | Macro F1 | Kesimpulan |
|---|---|---|
| 128 | lebih rendah | Underfitting |
| **256** | **tertinggi** | **Sweet spot ✅** |
| 384 | sedikit lebih rendah | Mulai overfitting |

✅ **Yang dipakai:** 256 (terbukti dari ablation study)
❌ **Alternatif:** 128 (underfitting), 384 (tidak signifikan lebih baik), 512 (terlalu besar untuk 25K dataset)

🎤 **Jawab dosen:**
> *"Pemilihan hidden size 256 didasarkan pada ablation study. Kami membandingkan 128, 256, dan 384. Hidden size 256 menghasilkan macro F1 tertinggi — 128 terlalu kecil sehingga underfitting, sedangkan 384 tidak memberikan peningkatan signifikan."*

---

## SESI 3: CARA KERJA GRU

**Analogi: Orang baca berita sambil pegang sticky note**

Setiap baca satu kata, sticky note (= hidden state) di-update:

```
Baca "Presiden"   → sticky note: "ada tokoh penting"
Baca "korupsi"    → sticky note: "ada hal negatif"
Baca "ditangkap"  → sticky note: "kasus hukum serius"
Baca "KPK"        → KESIMPULAN: NEGATIF! ✅
```

### 2 Tombol Pintar GRU:

**Tombol 1 — UPDATE GATE 🔄**
> "Informasi baru ini penting ga buat disimpen?"
```
Baca "korupsi"  → UPDATE GATE: "PENTING! Simpen!"
Baca "yang"     → UPDATE GATE: "Ga penting, skip."
```

**Tombol 2 — RESET GATE 🗑️**
> "Info lama yang ini masih relevan ga?"
```
Artikel ganti topik → RESET GATE: "Info lama hapus aja."
```

### GRU vs LSTM:

| | GRU | LSTM |
|---|---|---|
| Tombol (gate) | 2 (update + reset) | 4 (input + forget + cell + output) |
| Memori | 1 (hidden state) | 2 (hidden state + cell state) |
| Kecepatan | ⚡ Lebih cepat | 🐢 Lebih lambat |
| Parameter | Lebih sedikit | Lebih banyak |

**Kenapa GRU dipilih:** Dataset 25K = medium size. GRU cukup dan lebih efisien dari LSTM.

🎤 **Jawab dosen:**
> *"GRU memproses teks secara sekuensial, token per token. Setiap langkah, GRU memperbarui hidden state yang berfungsi seperti memori — menyimpan konteks dari kata-kata sebelumnya. GRU memiliki dua mekanisme gating: update gate yang memutuskan informasi baru mana yang disimpan, dan reset gate yang memutuskan informasi lama mana yang dihapus."*

---

## SESI 4: CARA MODEL BELAJAR (TRAINING LOOP)

**Analogi: Murid yang belajar dari koreksi guru**

Training = 3 langkah yang diulang ribuan kali:

```
STEP 1 — Forward Pass (tebak jawaban):
Artikel → IndoBERT → GRU → Classifier → prediksi: POSITIF

STEP 2 — Hitung Loss (ukur seberapa salah):
Prediksi: POSITIF
Jawaban benar: NEGATIF
Loss = 3.0  ← besar = salah banget!

STEP 3 — Backward Pass + Update (koreksi diri):
"Parameter mana yang bikin salah? Koreksi sedikit!"
optimizer.step()

Ulangi untuk 20.257 artikel × 15 epoch
```

### Early Stopping:
```python
EarlyStoppingCallback(early_stopping_patience=3)

# Artinya:
Epoch 3: F1 = 0.82 ↑ naik → lanjut
Epoch 4: F1 = 0.82 → tidak naik (1)
Epoch 5: F1 = 0.81 → tidak naik (2)
Epoch 6: F1 = 0.82 → tidak naik (3)
STOP! → load balik model epoch 3 yang terbaik
```

---

## SESI 5: LOSS FUNCTION

### Cross Entropy Loss (CE Loss)

**Rumus:** `Loss = -log(probabilitas jawaban yang benar)`

```
Model yakin & BENAR  (prob = 0.85): Loss = -log(0.85) = 0.16  ← kecil ✅
Model tidak yakin    (prob = 0.40): Loss = -log(0.40) = 0.92  ← sedang
Model yakin & SALAH  (prob = 0.05): Loss = -log(0.05) = 3.00  ← besar 💀
```

**Intinya:** Makin yakin & benar → loss makin kecil. Makin yakin & salah → loss makin besar.

---

### Class Weight — Atasi Class Imbalance

**Masalah:** Dataset tidak seimbang:
```
Netral  : 13.000 artikel ← BANYAK
Positif :  6.000 artikel
Negatif :  6.322 artikel
```

Tanpa class weight → model bias ke Netral karena paling sering muncul.

**Solusi di code:**
```python
# STEP 1: Hitung bobot otomatis (rumus bawaan sklearn)
class_weights = compute_class_weight('balanced', ...)
# Negative : 1.76
# Neutral  : 0.59
# Positive : 1.37

# STEP 2: Amplifikasi manual × [1.5, 1.0, 1.5]
class_weights = class_weights * np.array([1.5, 1.0, 1.5])
# Negative : 1.76 × 1.5 = 2.64  ← boost extra!
# Neutral  : 0.59 × 1.0 = 0.59  ← tetap
# Positive : 1.37 × 1.5 = 2.05  ← boost extra!

# STEP 3: Pindah ke GPU
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
```

**Kenapa dikali 1.5 lagi?**
Bobot dasar dari sklearn masih kurang signifikan karena Netral sangat dominan. Amplifikasi 1.5x membuat penalti untuk kesalahan di kelas minoritas lebih terasa.

**Efeknya:**
```
Salah prediksi Netral  : loss × 0.59 = dikecilkan
Salah prediksi Negatif : loss × 2.64 = dibesarkan 4.5× lebih berat!
```

✅ **Yang dipakai:** Class weight balanced + amplifikasi 1.5x
❌ **Alternatif:**
- Tidak pakai class weight → model bias ke Netral
- Oversampling (SMOTE) → berisiko untuk teks, duplikasi artikel bisa bikin model hafal
- Undersampling → buang data Netral yang banyak, info hilang

🎤 **Jawab dosen:**
> *"Bobot dasar yang dihitung otomatis menggunakan metode balanced dari sklearn sudah memberikan bobot lebih pada kelas minoritas. Namun karena distribusi kelas Netral yang sangat dominan, bobot dasar tersebut masih belum cukup signifikan. Oleh karena itu kami menambahkan amplifikasi 1.5x pada kelas Negatif dan Positif agar penalti untuk kesalahan prediksi di kelas minoritas lebih terasa."*

---

### CPU vs GPU — Kenapa `.to(device)`?

**Analogi: Gudang di lantai berbeda**
```
CPU = gudang lantai 1
GPU = gudang lantai 2

Data di lantai 1 TIDAK BISA langsung dipakai di lantai 2!
Harus dipindahin dulu dengan .to(device)!
```

```python
# TANPA .to(device) → ERROR!
class_weights = torch.FloatTensor([2.64, 0.59, 2.05])  # di CPU
loss = ce_loss * class_weights  # ce_loss di GPU → ERROR! ❌

# DENGAN .to(device) → AMAN!
class_weights = torch.FloatTensor([2.64, 0.59, 2.05]).to(device)  # pindah ke GPU
loss = ce_loss * class_weights  # keduanya di GPU → AMAN! ✅
```

| | CPU | GPU |
|---|---|---|
| Analogi | Kepala sekolah | Pasukan semut |
| Core | 8-32 | Ribuan |
| Kelebihan | Pinter, multitasking | Cepat untuk komputasi paralel |
| Di Kaggle | Prosesor biasa | Tesla T4 (gratis!) |

---

---

## SESI 6: FOCAL LOSS — DETAIL

### Masalah yang belum terselesaikan sama class weight

Meski class weight sudah ada, masih ada masalah:
```
13.000 artikel Netral → banyak yang MUDAH diprediksi
→ Model sudah yakin & benar untuk artikel-artikel ini
→ Tapi tetap mempengaruhi training dan buang waktu!
```

### Cara Kerja Focal Loss

```python
# Di code:
pt = torch.exp(-ce_loss)        # seberapa yakin model
focal_loss = ((1 - pt) ** 2.0 * ce_loss).mean()
```

**pt = seberapa yakin model terhadap jawaban yang benar**
```
Model yakin & benar  → pt TINGGI (0.90)
Model salah/ragu     → pt RENDAH (0.08)
```

**`(1 - pt)^2` = pengatur volume:**
```
Artikel MUDAH (pt=0.90): (1-0.90)^2 = 0.01 → loss × 0.01 = diabaikan! 🔇
Artikel SUSAH (pt=0.08): (1-0.08)^2 = 0.85 → loss × 0.85 = diperhatiin! 🔊
```

### Kenapa gamma = 2.0?
```
gamma = 0.0 → sama aja CE biasa, tidak ada efek
gamma = 1.0 → efek ringan
gamma = 2.0 → OPTIMAL (dari paper Lin et al. 2020) ✅
gamma = 3.0 → terlalu agresif, overfitting
```
Dibuktikan dengan ablation study — gamma 2.0 menghasilkan macro F1 dan F1 kelas minoritas tertinggi.

### Gabungan Class Weight + Focal Loss
```
Artikel Negatif SUSAH:
① class weight : loss × 2.64
② focal weight : × 0.85
FINAL           : loss × 2.24  ← double boost!

Artikel Netral MUDAH:
① class weight : loss × 0.59
② focal weight : × 0.01
FINAL           : loss × 0.006 ← nyaris nol!
```

### Contoh artikel MUDAH vs SUSAH

| Tipe | Contoh | Kenapa |
|---|---|---|
| Netral MUDAH | "Rapat kabinet membahas RAPBN hari ini" | Fakta polos, tidak ada kata emosional |
| Positif MUDAH | "Presiden umumkan kenaikan gaji karyawan" | Jelas positif, tidak ambigu |
| Negatif MUDAH | "Perang telah terjadi di Indonesia Timur" | Jelas negatif, kata 'perang' kuat |
| Negatif SUSAH | "Korupsi telah terjadi di Polri, namun presiden berkata kasus harus diusut tuntas" | Sinyal campur: korupsi (negatif) vs usut tuntas (positif) |

### Kenapa butuh KOMBINASI class weight + focal loss?

```
Class Weight = "SIAPA yang harus diperhatiin?"
               → Kelas Negatif & Positif dapat bobot lebih besar

Focal Loss   = "ARTIKEL MANA yang harus diperhatiin?"
               → Artikel susah dapat perhatian lebih
               → Artikel mudah diabaikan
```

**Analogi:**
```
Class Weight = guru kasih PR lebih banyak ke murid yang lemah
Focal Loss   = guru fokus ngajarin soal yang susah, skip yang mudah
```

Tanpa class weight → model anggap enteng Negatif & Positif
Tanpa focal loss → model buang waktu di artikel Netral yang mudah

✅ **Yang dipakai:** Class weight balanced × 1.5 + Focal Loss gamma=2.0
❌ **Alternatif:**
- CE biasa → bias ke Netral
- Class weight saja → masih buang waktu di artikel mudah
- Focal loss saja → kelas minoritas masih kurang dapat perhatian

🎤 **Jawab dosen (versi kamu sendiri — sudah bagus!):**
> *"Kami menggunakan kombinasi class weight dan focal loss untuk mengatasi class imbalance. Class weight memberikan bobot lebih besar pada kelas Negatif dan Positif karena jumlah datanya hampir setengah dari kelas Netral — selisihnya hampir 2x lipat. Sementara focal loss memastikan model tidak membuang waktu pada artikel yang sudah mudah diprediksi, sehingga training lebih fokus pada artikel yang sulit dan ambigu. Kombinasi keduanya memberikan double boost untuk kelas minoritas."*

---

---

## SESI 7: GRU_LAYERS = 2

**Analogi: 2 tim analis berlapis**
```
Layer 1 (Junior) → belajar pola level rendah (kata-kata)
                   "ada kata korupsi, ada kata pejabat"

Layer 2 (Senior) → belajar pola level tinggi (makna & konteks)
                   "oh ini kasus hukum serius!" → NEGATIF ✅
```

**Kenapa tidak 1 atau 3 layer?**
| Layer | Masalah |
|---|---|
| 1 | Kurang dalam, susah tangkap pola kompleks |
| **2** | **Sweet spot ✅** |
| 3+ | Overfitting, vanishing gradient |

**Ini bukan fitur GRU doang** — semua neural network bisa di-stack:
- GRU, LSTM → `num_layers=2`
- IndoBERT → sudah punya **12 layer Transformer** di dalamnya!
- "Deep" di Deep Learning = banyak layer ditumpuk

🎤 **Jawab dosen:**
> *"Dua layer GRU memungkinkan model mempelajari representasi hierarki — layer pertama menangkap pola level kata, layer kedua menangkap pola level makna dan konteks. Lebih dari 2 layer berisiko overfitting untuk dataset 25K."*

---

## SESI 8: DROPOUT = 0.2

**Analogi: 512 lampu yang random dimatiin saat training**

```
TANPA Dropout:
💡💡💡💡💡💡💡💡💡💡 → semua selalu nyala
→ tiap neuron spesialisasi 1 hal saja
→ ketemu kata baru → panik! → OVERFITTING

DENGAN Dropout 0.2:
Batch 1: 💡💡⬛💡💡⬛💡💡💡⬛ (20% = 102 neuron mati random)
Batch 2: ⬛💡💡💡⬛💡💡⬛💡💡 (beda lagi yang mati)
→ semua neuron terpaksa serba bisa
→ model lebih ROBUST & GENERAL ✅
```

**Kapan dropout ON/OFF?**
```
model.train() → dropout ON  (saat training)
model.eval()  → dropout OFF (saat evaluasi & predict)
```
Trainer ngatur ini otomatis — tidak perlu manual!

**Analogi ON/OFF:**
```
Training = latihan pakai rompi berat (dropout ON)
Test     = pertandingan sungguhan tanpa beban (dropout OFF)
```

**Kenapa 0.2?**
| Nilai | Efek |
|---|---|
| 0.0 | Tidak ada dropout → overfitting |
| 0.1 | Terlalu ringan |
| **0.2** | **Sweet spot untuk BERT-based model ✅** |
| 0.5 | Terlalu agresif, susah belajar |

**Kalau tidak pakai dropout → model OVERFITTING:**
```
Training accuracy  : 95% ← hafal data training
Validation accuracy: 72% ← jelek di data baru!
```

✅ **Yang dipakai:** Dropout 0.2
❌ **Alternatif:** Tidak pakai dropout (overfitting), dropout 0.5 (terlalu agresif)

🎤 **Jawab dosen:**
> *"Dropout adalah teknik regularisasi yang secara random mematikan 20% neuron saat training. Ini mencegah neuron terlalu bergantung satu sama lain sehingga model dipaksa belajar representasi yang lebih robust. Saat evaluasi, dropout dinonaktifkan otomatis sehingga semua neuron aktif."*

---

---

## SESI 9: DATA LOADING — Cell 7

```python
df1 = pd.read_csv(".../cnbc_labeled.csv")
df2 = pd.read_csv(".../detik_labeled.csv")
df3 = pd.read_csv(".../kompas_labeled.csv")

df_seed = pd.concat([_select_cols(df1), _select_cols(df2), _select_cols(df3)],
                     ignore_index=True)
```

**`_select_cols()`** = pilih 6 kolom yang relevan aja:
```
date, title, content, article_id, text, label
→ kolom lain dibuang, dataframe lebih ringan
```

**`ignore_index=True`** = reset nomor baris setelah concat:
```
TANPA: index duplikat (0,1,2...9998, 0,1,2...5342) ❌
DENGAN: index berurutan (0, 1, 2, ... 25321) ✅
```

---

## SESI 10: EDA & PREPROCESSING — Cell 9

**4 hal yang dilakukan sekaligus:**

### 1. Visualisasi Distribusi Label
```
Bar chart + pie chart → kenalan sama data dulu!
Konfirmasi class imbalance:
Netral  : 13.000 (51.3%) ← mayoritas!
Negatif :  6.322 (24.9%)
Positif :  5.000 (23.8%)
```

### 2. Train/Val Split (Stratified)
```python
train_df, val_df = train_test_split(
    df_seed, test_size=0.2,
    stratify=df_seed['label'],  # ← KUNCI!
    random_state=SEED
)
```

**Kenapa stratify?**
```
TANPA: proporsi kelas bisa beda antara train & val → tidak representatif ❌
DENGAN: proporsi kelas sama persis di train & val ✅

Train : 20.257 artikel (80%)
Val   :  5.065 artikel (20%)
```

### 3. Remap Label
```python
label_to_id = {-1: 0, 0: 1, 1: 2}
# PyTorch butuh label mulai dari 0 berurutan
# Label -1 tidak bisa dihandle PyTorch → ERROR!
```

### 4. Compute Class Weights
Sudah dibahas di sesi sebelumnya 😄

🎤 **Jawab dosen:**
> *"Preprocessing meliputi 4 tahap: EDA untuk konfirmasi class imbalance, stratified split 80:20, remapping label dari -1,0,1 ke 0,1,2 untuk PyTorch, dan menghitung class weights untuk loss function."*

---

## SESI 11: TOKENISASI & DATASET — Cell 11

### `attention_mask` — apaan?

Masalah: artikel dalam satu batch harus panjang sama → perlu padding (PAD token).
Tapi GRU tidak boleh baca PAD → merusak hidden state!

**Solusi: attention_mask**
```
attention_mask = 1 → token NYATA, harus dibaca
attention_mask = 0 → token PAD, SKIP!

Contoh:
input_ids      : [2983, 1234, 456,  0,   0,   0]
attention_mask : [  1,    1,   1,   0,   0,   0]
                  ↑baca  ↑baca ↑baca ↑skip ↑skip
```

### `padding=False` + `DataCollatorWithPadding`

```
padding='max_length' (BOROS):
→ Pad SETIAP artikel ke 512 token selalu
→ 16 artikel × 512 = 8.192 token per batch 😱

padding=False + DataCollator (HEMAT):
→ Pad ke panjang terpanjang DALAM BATCH itu aja
→ Kalau max artikel 12 token → 16 × 12 = 192 token ✅
→ Hemat VRAM & lebih cepat!
```

### `return_tensors=None`
```
None → tetap list Python dulu (fleksibel)
'pt' → langsung tensor (susah di-batch karena ukuran beda)

DataLoader yang convert ke tensor setelah tau ukuran batch!
```

🎤 **Jawab dosen:**
> *"Kami menggunakan `padding=False` dan `DataCollatorWithPadding` untuk efisiensi memori. Padding hanya dilakukan sampai panjang artikel terpanjang dalam satu batch, bukan sampai MAX_LENGTH 512. Attention mask memberitahu model token mana yang nyata dan mana yang padding."*

---

---

## SESI 12: ARSITEKTUR MODEL `IndoBertGRU` — Cell 13

### Komponen di `__init__()`:

| Komponen | Kode | Fungsi |
|---|---|---|
| **IndoBERT** | `self.bert = AutoModel.from_pretrained(model_name)` | Load 12 layer Transformer. Output: `[batch, seq_len, 768]` |
| **GRU** | `nn.GRU(input=768, hidden=256, layers=2, bidirectional=True)` | Layer GRU dua arah, 2 layer ditumpuk |
| **Dropout** | `nn.Dropout(0.2)` | Matiin 20% neuron saat training |
| **Classifier** | `nn.Linear(512, 3)` | 512 (= 256×2 BiGRU concat) → 3 kelas sentimen |

### Kenapa harus extend `nn.Module`?

Semua model PyTorch wajib extend `nn.Module` — kayak "KTP" biar PyTorch kenal & bisa manage otomatis (`.to(device)`, `.train()`, `.eval()`, hitung gradient, dll).

`super().__init__()` = daftar ke PyTorch dulu sebelum tambahin komponen.

### Alur Data Forward — Step by Step:

```
Input artikel
    ↓ Tokenizer
Token IDs [101, 2983, 1234, ...]
    ↓ IndoBERT (12 layer Transformer)
[batch, seq_len, 768]  ← setiap token dapat vektor 768 dimensi
    ↓ pack_padded_sequence (buang PAD)
sequence padat
    ↓ BiGRU (2 arah)
forward h_n[-2]  +  backward h_n[-1]
    ↓ torch.cat([h_n[-2], h_n[-1]], dim=1)
[batch, 512]
    ↓ Dropout (matiin 20%)
[batch, 512]
    ↓ Linear(512→3)
[batch, 3] = logits
    ↓ argmax
[-2.1, 0.3, 4.8] → POSITIF! ✅
```

---

## SESI 13: BATCH × TOKEN × FITUR — Bentuk Tensor

**`[batch, jumlah_token, 768]`** itu maksudnya:

```
Lapisan 1 = BATCH    → berapa artikel sekaligus (16 artikel/batch)
Lapisan 2 = TOKEN    → berapa token per artikel (max 512)
Lapisan 3 = FITUR    → 768 angka per token (dari IndoBERT)
```

Contoh dengan batch=2 artikel:
```
[
  [  ← Artikel 1
    [0.2, -0.8, 0.5, ... 768 angka],  ← token "Korupsi"
    [0.1, -0.3, 0.7, ... 768 angka],  ← token "pejabat"
  ],
  [  ← Artikel 2
    [0.3, -0.1, 0.8, ... 768 angka],  ← token "Presiden"
    [0.5, -0.4, 0.6, ... 768 angka],  ← token "umumkan"
  ]
]
```

Shape di kode kamu: `[16, 512, 768]` = 16 artikel × max 512 token × 768 dimensi

---

## SESI 14: PADDING & ATTENTION MASK

### Masalahnya:
Artikel dalam satu batch panjangnya beda. Tapi GPU butuh ukuran sama!

```
Artikel 1: "Korupsi pejabat ditangkap" → 3 token
Artikel 2: "Presiden umumkan kenaikan gaji karyawan" → 6 token
```

### Solusi: Padding
```
Artikel 1: [korupsi, pejabat, ditangkap, PAD, PAD, PAD]
Artikel 2: [presiden, umumkan, kenaikan, gaji, karyawan, PAD]
```

Semua jadi 6 token! ✅

### Tapi GRU ga boleh baca PAD!
PAD = token kosong, bisa bikin sticky note GRU jadi rusak.

### Solusi: `attention_mask`
```
attention_mask = 1 → "token NYATA, harus dibaca!"
attention_mask = 0 → "token PAD, SKIP!"

Artikel 1:
input_ids      : [2983, 1234, 456,  0,   0,   0]
attention_mask : [  1,    1,   1,   0,   0,   0]
```

### Mengapa PAD ada lalu dibuang lagi?
- **IndoBERT (GPU paralel)** butuh ukuran sama → harus pakai PAD
- **GRU (sequential)** ga butuh ukuran sama → buang PAD untuk efisiensi

Solusi: `pack_padded_sequence` — peras artikel, gabungin semua token nyata jadi satu sequence padat, buang PAD.

---

## SESI 15: GRU BIDIRECTIONAL & h_n

### `h_n` itu apa?
**Hidden state TERAKHIR** setelah GRU selesai baca seluruh artikel. Sticky note akhir!

```
Baca "Korupsi"   → sticky note update
Baca "pejabat"   → sticky note update
Baca "ditangkap" → sticky note update
Baca "KPK"       → sticky note FINAL ← ini h_n!
```

### Bentuk `h_n`:
```
h_n shape: [num_layers × 2, batch, hidden_size]
         = [2 × 2, batch, 256]
         = [4, batch, 256]
```

### Visualisasi sebagai gedung 4 lantai:

```
┌─────────────────────────────────┐
│ Lantai 3 = h_n[3] = h_n[-1]     │  ← Layer 2, Backward ✅
├─────────────────────────────────┤
│ Lantai 2 = h_n[2] = h_n[-2]     │  ← Layer 2, Forward  ✅
├─────────────────────────────────┤
│ Lantai 1 = h_n[1] = h_n[-3]     │  ← Layer 1, Backward
├─────────────────────────────────┤
│ Lantai 0 = h_n[0] = h_n[-4]     │  ← Layer 1, Forward
└─────────────────────────────────┘
```

### Kenapa `-2` dan `-1`?

Index negatif = hitung dari belakang → **selalu ambil 2 lantai teratas (layer paling dalam)**.

Kalau pakai index positif (2 dan 3) → harus ganti code kalau jumlah layer berubah.
Kalau pakai negatif (-2 dan -1) → otomatis bener berapapun jumlah layer.

### Forward vs Backward — 2 orang baca artikel:

```
Artikel: "Meski ekonomi lesu, pertumbuhan akhirnya meningkat pesat"

Orang 1 (Forward, kiri→kanan):
"awalnya negatif... ada pembalikan... akhirnya positif"
→ KESIMPULAN FORWARD = h_n[-2]  [256 angka]

Orang 2 (Backward, kanan→kiri):
"sangat positif... meski awalnya susah... akhirnya berhasil"
→ KESIMPULAN BACKWARD = h_n[-1]  [256 angka]

Gabung: torch.cat([h_n[-2], h_n[-1]], dim=1) → [batch, 512]
```

Dua perspektif = pemahaman lebih lengkap! Penting karena kata akhir kalimat sering mempengaruhi makna kata awal.

---

## SESI 16: LINEAR LAYER & LOGITS

### `nn.Linear(512, 3)` = perkalian matriks + bias

```
output = input × W + b

W = matriks bobot [512, 3]  ← 1.536 angka yang dipelajari
b = bias [3]                 ← 3 angka tambahan
```

### Cara hitung:
```
Skor Negatif = (a1×w1) + (a2×w2) + ... + (a512×w512) + b1
Skor Netral  = (a1×w3) + (a2×w4) + ... + (a512×w1024) + b2
Skor Positif = (a1×w5) + (a2×w6) + ... + (a512×w1536) + b3

Hasil: [-2.1, 0.3, 4.8] = logits
       Neg   Net   Pos

argmax → POSITIF! ✅
```

### Analogi 512 saksi:
Bayangin 512 saksi, masing-masing punya bobot kepercayaan berbeda untuk tiap vonis. Setiap saksi vote dengan bobotnya → semua suara dijumlahkan → vonis akhir.

### Output 3 karena 3 kelas:
Kalau task binary classification → output 2. Kalau 10 kelas → output 10.

### Logits ≠ Probabilitas
Logits = skor mentah (bisa negatif, bisa > 1). Trainer otomatis apply Softmax saat predict untuk dapat probabilitas.

---

## SESI 17: BAGAIMANA INDOBERT "BELAJAR" 768 DIMENSI?

### Tidak ada yang melabeli!
IndoBERT pakai **Self-Supervised Learning** dengan tugas **Masked Language Modeling (MLM)**.

### Cara kerja MLM:
```
Kalimat asli: "Presiden menandatangani keputusan penting"

IndoBERT dikasih: "Presiden [MASK] keputusan penting"
                              ↑ disembunyikan

IndoBERT harus tebak: "[MASK] = menandatangani"
```

Salah → koreksi bobot → coba lagi dengan jutaan kalimat lain.

### Data IndoBERT:
- Wikipedia Bahasa Indonesia
- Berita online
- Web crawl teks Indonesia
- Total: ~4GB teks!

### Hasilnya: representasi otomatis terbentuk
```
"korupsi" sering muncul dekat "pejabat", "ditangkap", "KPK"
→ 768 angkanya jadi mirip dengan kata-kata negatif

"berhasil" sering muncul dekat "pertumbuhan", "meningkat"
→ 768 angkanya jadi mirip dengan kata-kata positif
```

### Analogi belajar bahasa asing tanpa kamus:
```
Dengar 1000x:
"Il fait beau aujourd'hui" → orang senyum, pergi keluar
"Il fait froid aujourd'hui" → orang pakai jaket, menggigil

Tanpa kamus, kamu otomatis tau:
"beau" = bagus, "froid" = dingin
```

IndoBERT belajar dari **konteks**, bukan dari label!

---

## SESI 18: KLARIFIKASI PENTING — REPRESENTASI vs SENTIMEN

### 768 angka IndoBERT BUKAN label "positif/negatif"!

768 angka = **representasi MAKNA umum**, bukan sentimen!

```
"korupsi" → [0.2, -0.8, 0.5, ...]
            ↑ ini cuma bilang: "kata ini terkait tindakan ilegal,
              pejabat, hukum, uang"
            ← TIDAK ada info "ini negatif"!
```

### IndoBERT itu netral:
Bisa dipake untuk:
- Sentiment analysis ← skripsimu
- Ringkasan teks
- Deteksi hoax
- Question answering
- Dll.

### Yang nentuin sentimen = GRU + Classifier (yang KAMU tambah!)

```
IndoBERT = kamus → "korupsi = tindakan ilegal mengambil uang negara"
           ← belum tau ini sentimen apa

GRU + Classifier = kamu yang baca & simpulkan
                    "ada 'korupsi' → ini NEGATIF"
                    ← di sinilah sentimen ditentukan!
```

### Konteks-sensitif:
Kata yang sama bisa punya 768 angka berbeda tergantung kalimatnya!
```
"korupsi" di kalimat A → [0.2, -0.8, 0.5, ...]
"korupsi" di kalimat B → [0.1, -0.6, 0.4, ...]
                          ↑ beda! Karena konteksnya beda
```

Inilah kelebihan BERT vs Word2Vec/GloVe — **representasi kontekstual**, bukan fixed embedding.

### 🎤 Jawab dosen:
> *"IndoBERT menggunakan self-supervised learning dengan Masked Language Modeling pada corpus ~4GB teks Indonesia. Representasi 768 dimensi yang dihasilkan adalah representasi makna kontekstual — bukan label sentimen. Sentimen ditentukan oleh layer downstream (BiGRU + Linear classifier) yang dilatih khusus untuk task klasifikasi sentimen menggunakan dataset berlabel kami."*

---

---

# 📋 PROGRESS LENGKAP SKRIPSI — STATE SAAT INI

> Dokumen ini wajib dibaca dulu kalau pindah window/context baru.
> Tanggal sidang: **15 Juni 2026**

---

## 🎯 STATUS MODEL — 6 ARSITEKTUR

| Model | Macro F1 | Accuracy | Status | Lokasi Model |
|---|---|---|---|---|
| **IndoBERT + GRU** | **82.45%** | **83.51%** | ✅ SELESAI | `C:\Users\Davin\Downloads\gru_output\gru_best\model.safetensors` (485 MB) |
| **IndoBERT + CNN** | **82.55%** | **83.55%** | ✅ SELESAI | `C:\Users\Davin\Downloads\cnn_output\cnn_best\model_state_dict.pt` (479 MB) |
| IndoBERT + XGBoost | - | - | 🔄 Running di Kaggle nicolnicol | TBD |
| IndoBERT + BiLSTM | - | - | ⏳ Belum push | TBD |
| IndoBERT Fine-Tuned | - | - | ⏳ Belum push | TBD |
| IndoBERT + RFC | - | - | ⏳ Belum push | TBD |

### Hasil Detail GRU:
- Accuracy: 0.8351
- Macro F1: 0.8245
- Weighted F1: 0.8354
- F1-Negative: 0.8021
- F1-Neutral: 0.8545
- F1-Positive: 0.8170

### Hasil Detail CNN:
- Accuracy: 0.8355
- Macro F1: 0.8255
- Weighted F1: 0.8359
- F1-Negative: 0.7998
- F1-Neutral: 0.8533
- F1-Positive: 0.8233

---

## 🔑 API TOKENS KAGGLE

| Akun | Token | Status |
|---|---|---|
| davinraffilio9 (kamu) | `KGAT_e89aba9eae0a83a0d78fdff4e01d2a35` | Quota habis, reset Senin |
| nicolnicol (teman) | `KGAT_3b5e4e2b8b5106b09a3c56bf7294d997` | Aktif, dipakai untuk CNN & XGBoost |

### Kaggle Kernels:
- `davinraffilio9/revisi-fix-indobert-gru` — ✅ Selesai (GRU)
- `nicolnicol/revision-of-cnn-tok` — ✅ Selesai (CNN), CANCELED
- `nicolnicol/revision-indobert-xgboost` — 🔄 Running (kamu run manual T4)

### Cara Cek Status:
```powershell
$env:PYTHONUTF8 = "1"
$env:KAGGLE_API_TOKEN = "KGAT_3b5e4e2b8b5106b09a3c56bf7294d997"
kaggle kernels status nicolnicol/revision-indobert-xgboost
```

### Cara Download Output:
```powershell
kaggle kernels output nicolnicol/revision-indobert-xgboost -p "C:\Users\Davin\Downloads\xgboost_output"
```

---

## 📁 FILE PENTING

### Repo GitHub: `ininick/skripsis`
```
skripsis/
├── Davin Claudia/
│   ├── CLAUDE.md          ← panduan sidang lengkap
│   └── BELAJAR.md          ← FILE INI (catatan sesi belajar)
├── MODEL/
│   ├── README.md           ← status model + cara load
│   ├── GRU/                ← metadata GRU (tokenizer config + summary)
│   └── CNN/                ← metadata CNN (tokenizer config)
└── revision/
    ├── GRU/                 ← revisi-fix-indobert-gru.ipynb
    ├── CNN/                 ← indobert-cnn-political-news.ipynb
    ├── XGBoost/             ← indobert-xgboost.ipynb
    ├── BiLSTM/              ← (belum push ke Kaggle)
    ├── IndoBERT-FineTuned/  ← (belum push)
    └── RFC/                 ← (belum push)
```

### File di Downloads:
- `Panduan_Sidang_Skripsi_Davin.pdf` — panduan sidang lengkap (BAB 6 revisi reviewer detail)
- `Panduan_Cell_per_Cell_Notebook.pdf` — penjelasan tiap cell GRU & CNN
- `Revised_Paper_IEEE_XX.docx` — paper revisi format IEEE dengan placeholder kuning `[XX]`
- `gru_output/` — model GRU (485 MB)
- `cnn_output/` — model CNN (479 MB)

---

## 📊 DATASET

**Source:** Kaggle dataset `davinraffilio9/datalabeled`  
**Path Kaggle:** `/kaggle/input/datasets/davinraffilio9/datalabeled`

| File | Jumlah Artikel |
|---|---|
| `cnbc_labeled.csv` | 9.999 |
| `detik_labeled.csv` | 5.343 |
| `kompas_labeled.csv` | 9.980 |
| **TOTAL** | **25.322** |

**Split:** 80:20 stratified
- Train: 20.257 artikel
- Val: 5.065 artikel

**Distribusi:**
- Negatif: 4.792 (18.9%) — minoritas
- Netral: 14.358 (56.7%) — mayoritas
- Positif: 6.172 (24.4%)

---

## ⚙️ HYPERPARAMETER UTAMA

| Parameter | Nilai | Catatan |
|---|---|---|
| SEED | 42 | Reproducibility |
| MODEL_NAME | indobenchmark/indobert-base-p1 | IndoBERT base phase 1 |
| MAX_LENGTH | 512 | Hard limit BERT |
| GRU_HIDDEN | 256 | Dari ablation study |
| GRU_LAYERS | 2 | Layer GRU |
| DROPOUT | 0.2 | Regularisasi |
| Learning Rate | 3e-5 | Standar fine-tuning BERT |
| Batch Size (per device) | 16 | Train & eval |
| Gradient Accumulation | 2 steps | Effective batch = 32 |
| Epochs | 15 (max) | + early stopping patience=3 |
| LR Scheduler | Cosine | Smooth decay |
| Warmup Ratio | 0.1 | 10% pertama |
| Weight Decay | 0.01 | L2 regularisasi |
| fp16 | True | Mixed precision |
| Focal Loss gamma | 2.0 | Atasi imbalance |
| Class Weights | [2.64, 0.59, 2.05] | Balanced × [1.5, 1.0, 1.5] |

---

## 🎤 REVISI REVIEWER ICISS — 4 POIN

### R1: Ablation Study
**Komentar:** "Choices like hidden_size=256, bidirectional, kernel_sizes [3,4,5] need empirical justification."

**Fix:** 3-4 ablation per model
- GRU/BiLSTM: hidden size (128/256/384), uni vs bi, focal gamma (0/1/2/3)
- CNN: kernel sizes, num filters (64/128/256), pooling (max/avg/k-max)

### R2: Cross Validation
**Komentar:** "Single split insufficient — high variance, unclear generalization."

**Fix:** Stratified 5-Fold CV semua model, report mean ± std

### R3: Explainable AI (XAI)
**Komentar:** "Black-box models need interpretability for political domain."

**Fix:** 
- DL models → Integrated Gradients (Captum)
- XGBoost/RFC → SHAP TreeExplainer

### R4: Statistical Significance
**Komentar:** "Difference of 0.0092 could be random — need statistical tests."

**Fix:**
- Bootstrap CI 95% (n=1000)
- McNemar Test antara GRU (best) vs model lain

---

## 📚 PROGRESS BELAJAR — APA SAJA YANG SUDAH DIBAHAS

### ✅ Sudah Selesai (Sesi 1-18):
1. Import library (torch, nn, F, Dataset, DataLoader, AutoTokenizer, AutoModel)
2. Konfigurasi (SEED=42, MAX_LENGTH=512)
3. GRU_HIDDEN=256 (dari ablation)
4. Cara kerja GRU (sticky note + update/reset gate)
5. Training loop (forward, loss, backward)
6. Cross Entropy Loss (`-log(p)`)
7. Class Weight (×1.5 amplifikasi)
8. CPU vs GPU & `.to(device)`
9. Focal Loss & gamma=2.0
10. GRU_LAYERS=2 (2 tim analis berlapis)
11. DROPOUT=0.2 (lampu mati random)
12. Data Loading (3 CSV → concat → df_seed)
13. EDA & Preprocessing (stratified split, label remap, class weights)
14. Tokenisasi (`attention_mask`, `padding=False`, DataCollator)
15. Arsitektur IndoBertGRU (komponen + forward step-by-step)
16. Tensor shape `[batch, token, 768]`
17. Padding & PAD token (kenapa ada lalu dibuang)
18. BiGRU & h_n[-2], h_n[-1] (gedung 4 lantai)
19. Linear layer (matriks × bobot + bias)
20. IndoBERT belajar via Masked Language Modeling (~4GB teks Indonesia)
21. **KLARIFIKASI**: 768 dimensi = MAKNA bukan SENTIMEN

### ⏳ Yang Belum Dibahas:
- Cell 14-15: FocalLossTrainer & compute_metrics
- Cell 16-17: TrainingArguments detail (14 parameter)
- Cell 18: `trainer.train()` — apa yang terjadi di balik layar
- Cell 20-21: Evaluasi, classification_report, save model (kenapa `torch.save` bukan `trainer.save_model`)
- Cell 22-23: PySastrawi, Confusion matrix, Word Cloud
- Ablation Study (3 ablasi)
- K-Fold CV (StratifiedKFold)
- XAI dengan Captum (Integrated Gradients)
- Statistical Testing (Bootstrap CI + McNemar)
- Save predictions untuk McNemar di notebook lain

---

## 🚀 NEXT STEP UNTUK SESI BARU

1. **Cek status XGBoost di Kaggle** — `kaggle kernels status nicolnicol/revision-indobert-xgboost`
2. Kalau selesai → download output ke `C:\Users\Davin\Downloads\xgboost_output`
3. **Lanjut belajar dari Cell 14-15 (FocalLossTrainer & compute_metrics)**
4. Atau kalau mau, langsung lanjut ke Cell 16-18 (Training)

---

## 💡 CARA DAVIN MAU DIAJARIN (PENTING!)

- **Pelan-pelan**, jangan langsung teknis
- **Banyak contoh konkret** bahasa Indonesia (artikel politik)
- **Before vs After** kalau bisa
- **Analogi sehari-hari** (sticky note, gedung lantai, dapur, lampu, dll)
- **Sebutkan alternatif** yang tidak dipakai + alasannya (buat antisipasi pertanyaan dosen)
- **Kasih jawaban siap pakai buat sidang** dalam bahasa formal
- **Cecar dengan pertanyaan balik** setelah jelasin — buat latihan sidang
- Catat progress di **BELAJAR.md** secara berkala
- Pakai bahasa santai/ngobrol, jangan kaku

---

*Dokumen ini terus diupdate setiap sesi belajar.*
*Last updated: 3 Juni 2026 — End of Session*
