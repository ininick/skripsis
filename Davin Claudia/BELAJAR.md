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

*Dokumen ini terus diupdate setiap sesi belajar.*
*Last updated: 2 Juni 2026*
