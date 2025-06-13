
# Emotion Classification for Journaling Feature

Proyek ini bertujuan untuk mengklasifikasikan emosi dalam opini publik berbahasa Indonesia menggunakan pendekatan deep learning berbasis LSTM dengan lapisan Attention. Dataset yang digunakan merupakan kumpulan opini publik yang diambil dari media sosial tweeter dan telah dilabeli berdasarkan kategori emosi.

---

## Dataset

Dataset yang digunakan berasal dari repositori GitHub:

[Emotion Dataset from Indonesian Public Opinion – by Ricco48](https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/tree/main/Emotion%20Dataset%20from%20Indonesian%20Public%20Opinion)

Dataset terdiri dari beberapa file terpisah berdasarkan label emosi (seperti `joy.csv`, `sadness.csv`, `anger.csv`, dll). Sebelum digunakan, **semua file tersebut telah digabungkan (merge) menjadi satu file utama** bernama `data_merge.csv`. Proses merging ini mencakup:

- Membaca setiap file emosi
- Menggabungkan seluruh data menjadi satu `DataFrame` gabungan
- Menyimpan hasilnya sebagai `data_merge.csv`

---

## Tujuan

Tujuan utama proyek ini adalah:
- Membangun model klasifikasi emosi berbasis teks bahasa Indonesia
- Menggunakan teknik deep learning (BiLSTM + Attention)
- Mengatasi ketidakseimbangan kelas dengan `class_weight`
- Menerapkan analisis dan evaluasi model melalui metrik klasifikasi

---

## Arsitektur Model

Model ini menggunakan arsitektur model tensorflow yang terdiri dari:

- **Embedding Layer**: Mengubah teks menjadi representasi vektor
- **Bidirectional LSTM**: Menangkap konteks dari dua arah
- **Custom Attention Layer**: Fokus pada kata-kata penting dalam kalimat
- **Dense + Softmax Layer**: Untuk klasifikasi akhir menjadi salah satu dari beberapa emosi


## Preprocessing & Tokenisasi

- Tokenisasi dilakukan menggunakan `Tokenizer` dari Keras
- Padding dilakukan dengan panjang tetap (`max_len = 20`)
- Label dikodekan menggunakan `LabelEncoder` dan `to_categorical`
- Stratified split digunakan untuk membagi data menjadi train/test

---

## Class Imbalance Handling

Menggunakan `compute_class_weight` dari Scikit-learn untuk mengatasi ketidakseimbangan jumlah data antar kelas emosi. Beberapa class weight juga dituning manual untuk hasil optimal:

```python
class_weight_dict[0] = 1.18
class_weight_dict[4] = 0.62
class_weight_dict[5] = 1.18
```

---

## Training

Model dilatih menggunakan:

- `Adam` optimizer
- `categorical_crossentropy` loss
- Early stopping untuk mencegah overfitting
- ModelCheckpoint untuk menyimpan model terbaik berdasarkan `val_loss`


---

## Evaluasi

Model dievaluasi dengan metrik:

- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Prediksi emosi dengan argmax pada hasil softmax output

---

## Hasil Evaluasi Model

| Label    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Anger    | 0.84      | 0.89   | 0.86     | 590     |
| Fear     | 0.86      | 0.92   | 0.89     | 590     |
| Joy      | 0.84      | 0.85   | 0.84     | 591     |
| Love     | 0.87      | 0.93   | 0.90     | 591     |
| Neutral  | 0.72      | 0.50   | 0.59     | 591     |
| Sad      | 0.82      | 0.90   | 0.86     | 590     |

**Overall Metrics**:

- **Accuracy**: 0.83 (3543 samples)
- **Macro Avg**: Precision = 0.83, Recall = 0.83, F1-Score = 0.82
- **Weighted Avg**: Precision = 0.83, Recall = 0.83, F1-Score = 0.82

---

## Tools & Library

- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Pandas, NumPy, Seaborn, Matplotlib
- Jupyter Notebook

---
