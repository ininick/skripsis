# MODEL — Saved Models

Folder ini berisi metadata dan hasil dari model-model yang sudah selesai ditraining.

> **Note:** File weight model (.pt / .safetensors) tidak di-push ke GitHub karena ukurannya 479-485 MB per model. Weight tersimpan di lokal dan Kaggle output.

---

## Status Model

| Model | Macro F1 | Accuracy | Weighted F1 | Status |
|---|---|---|---|---|
| **IndoBERT + GRU** | **82.45%** | **83.51%** | **83.54%** | ✅ Selesai |
| IndoBERT + CNN | 82.55% | 83.55% | 83.59% | ✅ Selesai (model tersimpan) |
| IndoBERT + BiLSTM | TBD | TBD | TBD | ⏳ Antrian |
| IndoBERT Fine-Tuned | TBD | TBD | TBD | ⏳ Antrian |
| IndoBERT + XGBoost | TBD | TBD | TBD | ⏳ Antrian |
| IndoBERT + RFC | TBD | TBD | TBD | ⏳ Antrian |

---

## GRU — IndoBERT + GRU (Best Model)

**Lokasi weight:** `C:\Users\Davin\Downloads\gru_output\gru_best\model.safetensors` (485 MB)

**Arsitektur:** `[CLS] → BiGRU(hidden=256, layers=2) → Linear(512→3) → Softmax`

**Hasil:**
```json
{
  "accuracy": 0.8351,
  "macro_f1": 0.8245,
  "weighted_f1": 0.8354
}
```

**Kaggle Kernel:** `davinraffilio9/revisi-fix-indobert-gru`

---

## CNN — IndoBERT + CNN

**Lokasi weight:** `C:\Users\Davin\Downloads\cnn_output\cnn_best\model_state_dict.pt` (479 MB)

**Arsitektur:** `[CLS] → Conv1d(filters=128, kernels=[3,4,5]) → MaxPool → Linear(384→3) → Softmax`

**Hasil:**
```json
{
  "accuracy": 0.8355,
  "macro_f1": 0.8255,
  "weighted_f1": 0.8359
}
```

**Kaggle Kernel:** `nicolnicol/revision-of-cnn-tok`

---

## Cara Load Model (GRU)

```python
import torch
from transformers import AutoTokenizer
# pastikan class IndoBertGRU sudah di-import dari notebook

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/gru_best/")

# Load model
model = IndoBertGRU(model_name="indobenchmark/indobert-base-p1")
model.load_state_dict(torch.load("path/to/gru_best/model.safetensors"))
model.eval()
```

## Cara Load Model (CNN)

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/cnn_best/")

model = IndoBERT_CNN_Classifier(model_name="indobenchmark/indobert-base-p1", num_labels=3, ...)
model.load_state_dict(torch.load("path/to/cnn_best/model_state_dict.pt"))
model.eval()
```
