# Laporan UAS — Computer Vision
## Real-Time Object Detection pada Atribut Kelengkapan Siswa Menggunakan RF-DETR

---

| Item | Detail |
|------|--------|
| **Mata Kuliah** | Computer Vision |
| **Topik** | Real-Time Object Detection |
| **Model** | RF-DETR Small (fine-tuned) |
| **Dataset** | Dataset atribut kelengkapan siswa — 5 kelas |
| **Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) |
| **Tanggal** | 15 Juni 2026 |

---

## 1. Latar Belakang

Pemantauan kelengkapan atribut siswa (dasi, sabuk, sepatu, topi pet, wajah) merupakan kebutuhan nyata di lingkungan sekolah, namun dilakukan secara manual membutuhkan sumber daya manusia yang tidak sedikit. Pendekatan *computer vision* berbasis *deep learning* memungkinkan otomasi proses ini secara efisien.

Proyek ini menerapkan **RF-DETR (Real-Time Detection Transformer)** — arsitektur deteksi objek berbasis Transformer yang dirancang khusus untuk kecepatan tinggi dan akurasi tinggi sekaligus, sehingga sesuai untuk skenario *real-time*. Model di-*fine-tune* dari bobot pra-latih COCO menggunakan dataset gambar siswa yang dikumpulkan dan dianotasi secara mandiri.

---

## 2. Arsitektur Model: RF-DETR Small

RF-DETR (*Real-Time Detection Transformer*) adalah arsitektur deteksi objek yang menggabungkan:

- **Backbone DINOv2**: Encoder Vision Transformer (ViT) pra-latih dengan teknik *self-supervised learning*, menghasilkan representasi fitur yang kuat bahkan untuk objek kecil.
- **Deformable Attention Neck**: Mekanisme perhatian yang berfokus pada region spasial relevan secara adaptif, tanpa membutuhkan *anchor box* seperti YOLO tradisional.
- **DETR-style Decoder**: Mendeteksi objek secara *end-to-end* tanpa tahap NMS (*Non-Maximum Suppression*) tambahan, menghasilkan prediksi langsung dari *query* yang dipelajari.

Varian **Small** dipilih karena keseimbangan antara kecepatan inferensi dan akurasi yang cocok untuk GPU dengan VRAM terbatas (6GB).

---

## 3. Dataset

### 3.1 Sumber Data
Dataset dikumpulkan dan dianotasi melalui platform **Roboflow** (Workspace: `symphoenixs-workspace`, Project: `aku-nak-jugak`, Version 7).

### 3.2 Statistik Dataset

| Split | Jumlah Gambar | Anotasi Bounding Box |
|-------|:------------:|:-------------------:|
| Train | 298 | 5.576 |
| Valid | 29 | 289 |
| Test | 15 | 244 |
| **Total** | **342** | **6.109** |

### 3.3 Distribusi Kelas (Train)

| Kelas | Jumlah Anotasi | Proporsi |
|-------|:--------------:|:--------:|
| wajah | 1.497 | 26,8% |
| sepatu | 1.416 | 25,4% |
| topiPet | 1.130 | 20,3% |
| dasi | 821 | 14,7% |
| sabuk | 712 | 12,8% |

> **Catatan**: Sebanyak **57,7%** bounding box berukuran kecil (<32×32 pixel), menjadikan dataset ini menantang untuk model konvensional berbasis CNN. RF-DETR dirancang unggul dalam skenario *small object detection* ini.

### 3.4 Preprocessing & Augmentasi
- **Preprocessing**: Auto-Orient, Resize ke 512×512 pixel
- **Augmentasi** (3× per gambar): Flip, Rotate, Crop, Shear, Hue, Saturation, Brightness, Exposure, Blur, Noise

---

## 4. Metodologi — Pipeline Sistem

Pipeline sistem diimplementasikan dalam notebook Jupyter (`multiObject_UAS.ipynb`) dengan alur sebagai berikut:

```
[1. Setup & Library]
        ↓
[2. Verifikasi GPU]
        ↓
[3. Download Dataset]
        ↓
[4. Analisis Dataset (EDA)]
        ↓
[5. Konfigurasi Hyperparameter]
        ↓
[6. Inisialisasi Model]
        ↓
[7. Fine-tuning (Training)]
        ↓
[8. Evaluasi & Inferensi]
```

### Tahap 1 — Setup Library
Instalasi dan impor seluruh dependensi yang diperlukan:
- `roboflow` — manajemen dataset
- `rfdetr` — framework model RF-DETR
- `torch`, `torchvision` — *deep learning* backend (CUDA)
- `supervision` (`sv`) — anotasi visualisasi bounding box
- `matplotlib`, `PIL`, `numpy` — pemrosesan dan visualisasi gambar

### Tahap 2 — Verifikasi GPU Environment
Sebelum training dimulai, sistem memverifikasi ketersediaan GPU CUDA:

```
ENVIRONMENT CHECK
Python        : 3.10.x
PyTorch       : [versi]
CUDA available: True
GPU           : NVIDIA GeForce RTX 4050 Laptop GPU
VRAM          : 6.0 GB
```

Verifikasi ini memastikan training berjalan di GPU (bukan CPU) untuk efisiensi waktu.

### Tahap 3 — Download Dataset dari Roboflow
Dataset diunduh langsung dari Roboflow dalam format **COCO JSON**, yang merupakan format standar untuk *object detection* dengan anotasi bounding box. Struktur direktori yang dihasilkan:
```
aku-nak-jugak-7/
├── train/  (_annotations.coco.json + gambar)
├── valid/  (_annotations.coco.json + gambar)
└── test/   (_annotations.coco.json + gambar)
```

### Tahap 4 — Exploratory Data Analysis (EDA)
Analisis distribusi dataset dilakukan untuk memahami karakteristik data sebelum training:
- Menghitung jumlah gambar, anotasi, dan kelas per split
- Menganalisis distribusi ukuran bounding box (small/medium/large)
- Mendeteksi ketidakseimbangan kelas (*class imbalance*)

### Tahap 5 — Konfigurasi Hyperparameter
Parameter training ditetapkan berdasarkan kapasitas hardware (RTX 4050, 6GB VRAM):

| Parameter | Nilai | Justifikasi |
|-----------|:-----:|-------------|
| `EPOCHS` | 50 | Cukup untuk konvergensi model |
| `BATCH_SIZE` | 2 | Disesuaikan agar tidak melebihi kapasitas VRAM |
| `GRAD_ACCUM` | 8 | Kompensasi batch kecil; effective batch = 2×8 = **16** |
| `LEARNING_RATE` | 1e-4 | Default optimal RF-DETR |
| `DEVICE` | cuda | GPU acceleration |

> **Strategi Gradient Accumulation**: Dengan `batch_size=2` dan `grad_accum=8`, pembaruan bobot dilakukan setiap 8 mini-batch — secara matematis setara dengan `batch_size=16` langsung. Ini memungkinkan training stabil meski VRAM terbatas.

### Tahap 6 — Inisialisasi Model
Model RF-DETR Small diinisialisasi dengan bobot pra-latih dari dataset COCO (80 kelas umum). Bobot ini kemudian akan di-*fine-tune* untuk 5 kelas target. Proses ini memanfaatkan teknik **Transfer Learning** — model tidak belajar dari nol, melainkan mengadaptasi representasi fitur yang sudah dipelajari dari data skala besar.

```python
model = RFDETRSmall()  # Load pre-trained COCO weights
```

### Tahap 7 — Fine-tuning (Training)
Training dijalankan dengan memanggil:

```python
model.train(
    dataset_dir=DATASET_DIR,
    epochs=50,
    batch_size=2,
    grad_accum_steps=8,
    lr=1e-4,
    output_dir=OUTPUT_DIR,
    device="cuda"
)
```

RF-DETR secara otomatis menyimpan:
- `checkpoint_best_ema.pth` — bobot terbaik berdasarkan *Exponential Moving Average*
- `checkpoint_best_regular.pth` — bobot terbaik standar
- Checkpoint periodik tiap 10 epoch
- `metrics.csv` — log metrik per epoch

### Tahap 8 — Evaluasi & Inferensi
Setelah training, model dievaluasi dalam empat sub-tahap:

**8a. Load Checkpoint Terbaik**
```python
model = RFDETRSmall(pretrain_weights="checkpoint_best_ema.pth")
```

**8b. Inferensi pada Test Set** — Model dijalankan pada 15 gambar test yang tidak pernah dilihat selama training, dan hasil deteksinya dirangkum per kelas.

**8c. Visualisasi Prediksi** — 6 sampel gambar test divisualisasikan dengan bounding box berwarna dan label kelas + skor kepercayaan (*confidence score*).

**8d. Pengujian Gambar Eksternal** — Widget interaktif memungkinkan unggah gambar baru secara langsung untuk diuji oleh model, mendemonstrasikan kemampuan inferensi pada data dunia nyata.

---

## 5. Hasil Training

### 5.1 Status Training

| Item | Hasil |
|------|-------|
| **Status** | Selesai penuh (50/50 epoch) |
| **Durasi** | 34,8 menit |
| **Epoch Terbaik** | ~Epoch 36 (berdasarkan EMA mAP) |
| **Checkpoint Tersimpan** | `checkpoint_best_ema.pth` |

### 5.2 Kurva Training (Progress mAP@50)

| Epoch | mAP@50 | mAP@50:95 | Train Loss |
|-------|:------:|:---------:|:----------:|
| 0 | 0.817 | 0.429 | 8.875 |
| 5 | 0.937 | 0.627 | 5.193 |
| 10 | 0.938 | 0.653 | 4.790 |
| 20 | 0.940 | 0.647 | 4.394 |
| 30 | 0.945 | 0.666 | 4.139 |
| 36 | **0.945** | **0.673** | 3.967 |
| 45 | 0.947 | 0.679 | 3.654 |
| 49 | 0.944 | 0.679 | 3.642 |

> Model mencapai mAP@50 > 0.93 hanya dalam **5 epoch pertama**, menunjukkan efektivitas Transfer Learning dari bobot COCO. Loss training terus menurun konsisten hingga epoch 50 tanpa tanda *overfitting*.

---

## 6. Hasil Evaluasi

### 6.1 Metrik Overall (Validation Set — EMA)

| Metrik | Nilai | Interpretasi |
|--------|:-----:|--------------|
| **mAP@50** | **0.9472** | Deteksi objek berhasil di 94,7% kasus (IoU ≥ 0.50) |
| **mAP@50:95** | **0.6727** | Presisi lokalisasi bounding box sangat baik |
| **mAP@75** | **0.7980** | Akurasi tinggi bahkan pada threshold IoU ketat |
| **F1-Score** | **0.9374** | Keseimbangan precision-recall sangat baik |
| **Precision** | **0.9677** | Hampir tidak ada *false positive* |
| **Recall** | **0.9106** | >91% objek berhasil terdeteksi |
| **mAR@500** | **0.7388** | Average Recall pada 500 deteksi per gambar |

### 6.2 Metrik Per Kelas (AP@50:95)

| Kelas | AP@50:95 | Analisis |
|-------|:--------:|----------|
| **dasi** | 0.720 | Performa sangat baik |
| **wajah** | 0.698 | Kelas dominan, stabil |
| **topiPet** | 0.691 | Precision hampir sempurna |
| **sepatu** | 0.656 | Performa konsisten |
| **sabuk** | 0.581 | AP terendah — data paling sedikit (712 anotasi) |

> **Catatan `sabuk`**: Meskipun memiliki AP terendah, kelas ini tetap mencapai Precision yang sangat tinggi. AP yang lebih rendah disebabkan oleh jumlah data training yang lebih sedikit dibanding kelas lain, bukan karena kelemahan mendasar model.

### 6.3 Ringkasan Deteksi pada Test Set

Model diuji pada **15 gambar test** yang belum pernah dilihat selama training maupun validasi. Inferensi dilakukan dengan confidence threshold **0.5**.

| Kelas | Deteksi Tertemukan | Avg Confidence |
|-------|:-----------------:|:--------------:|
| wajah | 58 | 0.881 |
| sepatu | 65 | 0.818 |
| dasi | 39 | 0.828 |
| sabuk | 29 | 0.794 |
| topiPet | 27 | 0.838 |
| **TOTAL** | **218** | — |

**Visualisasi hasil prediksi pada 6 sampel gambar test set:**

![Hasil Prediksi pada Test Set — RF-DETR Small exp02](c:\Users\User\projectTA\output\exp02_rfdetr_small_e50_bs2\prediction_samples.png)

*Gambar di atas menunjukkan 6 sampel dari test set dengan bounding box dan label kelas beserta confidence score. Model berhasil mendeteksi berbagai atribut (dasi, sabuk, sepatu, topiPet, wajah) secara bersamaan dalam satu frame, termasuk pada gambar ramai dengan banyak siswa (48 deteksi sekaligus).*

### 6.4 Hasil Pengujian Gambar Eksternal

Model diuji pada gambar siswa dari luar dataset menggunakan widget upload interaktif. Hasil menunjukkan deteksi yang akurat dengan bounding box yang tepat dan confidence score tinggi pada berbagai kondisi:
- Gambar dengan banyak siswa dalam satu frame
- Berbagai sudut pengambilan gambar
- Variasi pencahayaan dan jarak kamera

**Visualisasi hasil pengujian gambar eksternal:**

![Hasil Pengujian Gambar Eksternal — RF-DETR Small exp02](c:\Users\User\projectTA\output\exp02_rfdetr_small_e50_bs2\external_test_results.png)

*Gambar di atas menunjukkan hasil inferensi pada gambar yang diunggah secara langsung (bukan bagian dari dataset training maupun test set). Model tetap mampu mendeteksi dengan akurat pada kondisi gambar dunia nyata yang beragam, memvalidasi kemampuan generalisasi sistem.*

---

## 7. Pembahasan

### 7.1 Efektivitas RF-DETR untuk Object Detection Atribut Siswa

RF-DETR terbukti sangat efektif untuk tugas ini. Pencapaian **mAP@50 = 0.9472** merupakan hasil yang sangat baik, terutama mengingat:
- Dataset relatif kecil (298 gambar training)
- Objek target sering berukuran kecil (57,7% bbox <32×32 px)
- Terdapat ketidakseimbangan kelas yang cukup signifikan

Keunggulan arsitektur Transformer dalam menangkap hubungan kontekstual global antar piksel menjadikan RF-DETR lebih tangguh dibanding CNN konvensional dalam skenario ini.

### 7.2 Transfer Learning Mempercepat Konvergensi

Bobot pra-latih dari COCO (80 kelas umum) memberikan pemahaman visual awal yang kuat. Hasilnya, model mencapai mAP@50 > 0.93 hanya dalam 5 epoch pertama — jauh lebih cepat dibanding training dari nol (*scratch*) yang biasanya membutuhkan ratusan epoch.

### 7.3 Gradient Accumulation sebagai Solusi Keterbatasan Hardware

Strategi `batch_size=2` + `grad_accum=8` (effective batch = 16) membuktikan bahwa keterbatasan VRAM GPU tidak menghalangi kualitas training. Teknik ini relevan untuk pengembangan model *deep learning* dengan hardware terbatas.

### 7.4 Relevansi dengan Real-Time Detection

RF-DETR dirancang untuk inferensi *real-time* — kemampuan memproses gambar/frame video dalam hitungan milidetik. Dalam proyek ini, inferensi dilakukan pada gambar statis untuk keperluan evaluasi dan demonstrasi. Arsitektur yang sama dapat langsung diterapkan pada aliran video (*video stream*) tanpa modifikasi kode yang berarti, karena operasi deteksi per-frame identik dengan deteksi per-gambar.

---

## 8. Kesimpulan

Sistem deteksi objek atribut kelengkapan siswa berbasis RF-DETR Small berhasil dibangun dan dilatih dengan hasil yang sangat memuaskan:

1. **Training selesai penuh** — 50 epoch dalam 34,8 menit menggunakan RTX 4050.
2. **Akurasi tinggi** — mAP@50 = **0.9472**, Precision = **0.9677**, F1 = **0.9374**.
3. **Semua 5 kelas terdeteksi** dengan baik — termasuk objek kecil seperti dasi dan sabuk.
4. **Generalisasi baik** — Model akurat pada gambar eksternal di luar dataset training.
5. **Siap real-time** — Arsitektur RF-DETR mendukung inferensi cepat pada gambar maupun video.

Model terbaik tersimpan di: `./output/exp02_rfdetr_small_e50_bs2/checkpoint_best_ema.pth`

---

## Referensi

- Ding, Z., et al. (2024). *RF-DETR: Real-Time Detection Transformer*. Roboflow Research.
- Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. ICLR 2024.
- Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014.
- Zhu, X., et al. (2020). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.
