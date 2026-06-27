# Perbandingan Novelty: Sistem Anda vs SiRapi + Gap Analysis

> **Konteks Penulis:** Mahasiswa Teknik Informatika semester 6, proposal Tugas Akhir tipe prototipe (R&D) — *Rancang Bangun Sistem Cerdas dengan Metode Face Recognition dan Multi-Object Detection untuk Verifikasi Kelengkapan Atribut Seragam Sekolah.*
>
> Dokumen ini membandingkan 7 novelty yang ditemukan dari research gap terhadap kelemahan aktual SiRapi (berdasarkan code review) dan merumuskan insight yang memperkuat posisi kompetitif.

---V

## Daftar Isi

1. [Pemetaan Novelty → Gap SiRapi](#1-pemetaan-novelty--gap-sirapi)
2. [Deep Comparison per Novelty](#2-deep-comparison-per-novelty)
   - Novelty 1 (+ Ekstensi: Dapodik Import, Continuous Learning Loop, Re-ID Future Work)
   - Novelty 2 (Asosiasi Spasial)
   - Novelty 3 (Rule Engine)
   - Novelty 4 (+ Ekstensi: Augmentation Strategy)
   - Novelty 5 (+ Ekstensi: Privacy-Preserving Data Flow, FL-Ready Architecture)
   - Novelty 6 (Cloud Bot)
   - Novelty 7 (+ Ekstensi: Positive Reinforcement / Gamification)
3. [Analisis Arsitektur: React SPA + FastAPI vs Next.js + Go Fiber](#3-analisis-arsitektur)
4. [Keunggulan Tersembunyi yang Belum Anda Sadari](#4-keunggulan-tersembunyi)
5. [Potensi Kelemahan & Mitigasi](#5-potensi-kelemahan--mitigasi)
6. [Matriks Kekuatan Kompetitif Final](#6-matriks-kekuatan-kompetitif-final)
7. [Rekomendasi Teknis untuk Proposal TA](#7-rekomendasi-teknis-untuk-proposal-ta)
   - 7.1 [Framing Novelty di Proposal](#71-framing-novelty-di-proposal)
   - 7.2 [Positioning terhadap SiRapi](#72-positioning-terhadap-sirapi-di-proposal)
   - 7.3 [Apa yang Diambil dari SiRapi + Inovasi Tambahan](#73-strategi-teknis-apa-yang-diambil-dari-sirapi--inovasi-tambahan)
   - 7.4 [Strategi Pemilihan Model (SLR-Driven)](#74-strategi-pemilihan-model-slr-driven)
   - 7.5 [Arsitektur Teknis yang Direkomendasikan](#75-arsitektur-teknis-yang-direkomendasikan)

---

## 1. Pemetaan Novelty → Gap SiRapi

Tabel ini menunjukkan bagaimana setiap novelty Anda secara langsung menjawab kelemahan SiRapi yang ditemukan dari code review.

| # | Novelty Anda | Gap SiRapi yang Dijawab | Impact |
|---|-------------|------------------------|--------|
| N1 | Integrated FR + Multi-Object Detection Pipeline (+ Student Enrollment via Dapodik, Continuous Learning Loop, Re-ID extension) | **G-CV2** (tidak ada face recognition), **G-CV4** (single-model), **G-AB1** (tidak ada modul absensi nyata), **G-AB6** (tidak ada student enrollment), **G-CV6** (tidak ada retraining pipeline), **G-AB2** (tidak ada integrasi sekolah — via Dapodik import), **G-CV3** (Re-ID sebagai future work extension) | **CRITICAL** — Menjawab 7 gap, termasuk 3 gap severity CRITICAL |
| N2 | Asosiasi Spasial (Jarak Centroid) | **gap implisit** yang bahkan belum teridentifikasi di analisis awal — SiRapi tidak pernah mempertimbangkan masalah "atribut milik siapa" | **HIGH** — Novelty paling orisinal, tidak ada di SiRapi bahkan di level konsep |
| N3 | Rule Engine Verifikasi Kelengkapan | **G-CV7** (tidak ada handling variasi seragam), **G-AB5** (tidak ada dispensasi), **G-AB7** (tidak ada jadwal integration) | **HIGH** — SiRapi punya rules_engine.py tapi untuk konteks HSE/PPE, bukan seragam sekolah |
| N4 | Dataset Realistis Sekolah Tunggal (+ Augmentation Strategy) | **G-CV1** (model belum custom-trained), **G-CV10** (tidak ada augmentation strategy) | **HIGH** — SiRapi literal menggunakan `yolov8n.pt` pre-trained tanpa dataset apapun |
| N5 | Edge Deployment (+ Privacy-Preserving Data Flow, FL-Ready Architecture) | **G-ED1** (belum ada actual deployment), **G-ED2** (tidak ada model optimization), **G-SC2** (tidak ada privacy compliance — privacy by design), **G-AR5** (tidak ada multi-node strategy — FL-ready) | **HIGH** — Menambah dimension privacy + extensibility yang SiRapi tidak punya |
| N6 | Cloud Bot (n8n + LLM + WA/Telegram) | **G-AB3** (tidak ada notifikasi orang tua), **gap arsitektur**: n8n sebagai orchestrator lebih fleksibel dari hardcoded Telegram bot | **MEDIUM** — Telegram SiRapi functional tapi limited audience |
| N7 | Dashboard Pelaporan Terintegrasi (+ LLM Bot, Positive Reinforcement/Gamification) | **G-UX2** (dashboard terlalu complex), **G-UX5** (tidak ada role guru/wali kelas), **insight baru**: gamification self-contained sebagai engagement tool | **MEDIUM-HIGH** — Relevansi + inovasi UX (LLM bot + gamification) yang SiRapi tidak punya |

**Insight Kunci:** Setelah ekspansi, 7 novelty Anda menjawab **20+ dari 37 gap** yang teridentifikasi, termasuk **semua 5 gap severity CRITICAL** dan sebagian besar gap HIGH di kategori CV, Absensi, Edge, dan Privacy. Ini menunjukkan bahwa research gap Anda tepat sasaran dan coverage diferensiasinya luas.

---

## 2. Deep Comparison per Novelty

### Novelty 1: Integrated FR + Multi-Object Detection Pipeline

#### Apa yang SiRapi Punya (Kode Aktual)

```python
# ai-engine/main.py, baris 14
model = YOLO("yolov8n.pt")  # Pre-trained COCO, bukan custom

# ai-engine/main.py, baris 67-71
# === LOGIKA DUMMY PELANGGARAN ===
# TODO: Nanti ganti TARGET_CLASS_ID ini menjadi class_id atribut seragam
TARGET_CLASS_ID = 0  # "person" dari COCO dataset
```

SiRapi secara literal punya **satu model YOLOv8 pre-trained** yang hanya mendeteksi "person" (COCO class 0). Tidak ada:
- Face recognition model
- Custom-trained attribute detector
- Pipeline yang menghubungkan keduanya
- Konsep "status kelengkapan per individu"

#### Apa yang Anda Tawarkan

**Arsitektur pipeline (Final — Hybrid Sejati berdasarkan eksperimen Opsi B v2):**

```
Frame
  ├────────────────────────────────────────────────────────┐
  ↓                                                        ↓
RF-DETR (Custom-trained)                                InsightFace (Full Frame)
5 kelas: topi, dasi, sabuk,                             SCRFD detect + 5-point alignment
         sepatu, WAJAH                                  → ArcFace embedding
  ↓                                                        ↓
BBox atribut                  BBox wajah (anchor)       Identitas + BBox wajah (SCRFD)
(topi, dasi, sabuk, sepatu)           │                    │
      │                               └─────────┬──────────┘
      │                                         ↓
      │                               Spatial Association 
      │                     Match wajah RF-DETR ↔ wajah InsightFace
      │                     (via IoU atau centroid distance)
      │                                         ↓
      └─────────────────────────────────────────┴── link identitas ke atribut terdekat
                                                    ↓
                                          Rule Engine Verifikasi
                                                    ↓
               Output: "Ahmad Rizki → dasi: ✓, topi: ✗, sabuk: ✓, sepatu: ✓"
```

**Dua engine yang berjalan paralel pada frame yang sama:**
1. **RF-DETR (Atribut + Anchor):** Bertugas mendeteksi atribut seragam (topi, dasi, sabuk, sepatu) dan mendeteksi wajah HANYA sebagai anchor/titik pusat spasial.
2. **InsightFace (Recognition):** Bertugas menjalankan pipeline face recognition end-to-end secara independen (deteksi SCRFD → alignment → ArcFace embedding → matching). InsightFace memproses frame utuh, bukan crop dari RF-DETR.

> **Catatan model detection — RF-DETR:** Model dilatih dengan 5 kelas: topi, dasi, sabuk, sepatu, dan **wajah**. Wajah dideteksi oleh RF-DETR bukan sebagai input untuk face recognition, melainkan murni sebagai **anchor spasial** untuk menghubungkan atribut di sekitarnya ke wajah/orang yang tepat.

> **Catatan face recognition — InsightFace (Opsi B v2):** Keputusan final berdasarkan eksperimen komparatif. ArcFace sangat bergantung pada orientasi wajah yang ternormalisasi (5-point landmark alignment dari SCRFD). Karena itu, InsightFace harus memproses frame secara utuh, bukan memproses crop wajah dari RF-DETR yang rentan memotong area di sekitar wajah dan menggagalkan proses ekstraksi landmark.

> **Spatial Association (Penggabungan):** Pipeline menghubungkan (link) hasil identitas dari InsightFace ke atribut dari RF-DETR dengan cara mencocokkan bounding box wajah dari kedua detektor tersebut (misal via IoU tertinggi). Setelah wajah ter-match, identitas tersebut dihubungkan ke atribut-atribut terdekat di sekitarnya.

#### Kekuatan Pembanding

| Aspek | SiRapi | Sistem Anda |
|-------|--------|-------------|
| Person detection | ✓ (COCO pre-trained) | ✓ (Custom-trained, model dari hasil SLR) |
| Attribute detection | ✗ (TODO) | ✓ (5 custom classes: topi, dasi, sabuk, sepatu, wajah) |
| Face recognition | ✗ (tidak ada) | ✓ (ArcFace via InsightFace, enrollment-based) |
| Per-student output | ✗ | ✓ |
| Single pipeline | N/A | ✓ (RF-DETR → ArcFace → Spatial Assoc → Rule Engine, in-process) |
| Model selection rigor | Asal pakai pre-trained | Berbasis SLR + eksperimen komparatif |

#### Verdict: **Keunggulan Absolut**

Ini bukan incremental improvement — ini **fundamental capability** yang SiRapi tidak punya sama sekali. SiRapi hanya bisa bilang "ada person terdeteksi di frame". Sistem Anda bisa bilang "Ahmad Rizki tidak memakai dasi."

---

#### Ekstensi 1.1: Student Enrollment via Dapodik Import (One-Way)

Pipeline face recognition membutuhkan database siswa yang enrolled. Alih-alih input manual, sistem memanfaatkan **Dapodik import one-way** untuk onboarding siswa yang efisien.

**Scope yang diimplementasikan:**
- Import data siswa dari ekspor Dapodik (format Excel/CSV): NIS, nama, kelas, foto
- Face enrollment: upload foto via dashboard atau capture langsung dari webcam
- Face embedding disimpan di database lokal, linked ke student record
- **TIDAK ada sinkronisasi dua arah** dengan sistem informasi pendidikan nasional — menghindari kompleksitas integrasi API dan concern politik/MoU

**Kenapa keputusan scope ini penting:**
- Menghindari dependency pada infrastruktur eksternal (API Dapodik, e-Rapor)
- Izin sekolah cukup untuk akses ekspor data (bukan akses langsung ke sistem)
- Fully operational meskipun Dapodik down / tidak tersedia
- Dapat di-adapt untuk sekolah yang belum menggunakan Dapodik (manual CSV)

**Keunggulan vs SiRapi:** SiRapi sama sekali tidak memiliki konsep student enrollment. Data siswa harus diinput manual satu per satu. Sistem Anda menyediakan bulk import yang practical.

---

#### Ekstensi 1.2: Continuous Learning Loop (Hybrid Edge-Offline)

Pipeline bukan sistem statis — dirancang untuk **continuously improve** seiring waktu berdasarkan feedback dari pengguna.

**Arsitektur loop:**

```
[Edge Device]
    ↓
Inference → Detection result
    ↓
Confidence < threshold?
    ├── Ya → Flag ke Review Queue (disimpan lokal)
    └── Tidak → Normal flow (log attendance/violation)
    ↓
[Dashboard Review UI]
    ↓
Guru/Admin label: Accept / Reject / Correct
    ↓
Labeled samples accumulated
    ↓
Trigger retraining (N samples collected)
    ↓
[Retraining di Server Sekolah / Laptop Peneliti]
    ↓
Model baru divalidasi (held-out test set)
    ↓
Deploy ulang ke edge device (OTA-lite)
```

**Pembagian beban kerja:**

| Tahap | Lokasi | Justifikasi |
|-------|--------|-------------|
| Inference | **Edge device** (CCTV-connected laptop/server) | Real-time, low-latency |
| Data collection + labeling UI | **Edge device** | Tidak perlu infrastruktur tambahan |
| Retraining | **Server sekolah / laptop peneliti** | Butuh GPU kuat + memory besar — tidak feasible di edge |
| Validation + deployment | **Server sekolah** | On-premise, data tidak keluar lingkungan sekolah |

**Keunggulan vs SiRapi:** SiRapi memiliki `AnnotationBacklog` dan `review_queue` (infrastruktur feedback), tapi **tidak ada pipeline yang menghubungkan feedback tersebut ke retraining.** Sistem Anda menutup loop-nya secara eksplisit, meskipun retraining tidak dilakukan di edge device itu sendiri.

**Framing akademis:** Ini bukan klaim "full MLOps di edge" — jujur diakui sebagai **hybrid continuous learning loop** di mana edge melakukan inference + data collection, dan retraining dilakukan on-premise di server yang capable. Data tidak pernah keluar lingkungan sekolah.

---

#### Ekstensi 1.3: Future Work — Temporal Deduplication, Attribute Consolidation & Cross-Camera Re-ID

Spatial association (Novelty 2) menyelesaikan masalah "atribut milik siapa" **dalam satu frame**. Extension natural berikutnya adalah **tracking temporal dan Re-ID lintas kamera** — yang di-scope out sebagai future work dan dipecah menjadi tiga sub-ekstensi berurutan berdasarkan kompleksitas.

**Batasan masalah yang relevan:**
> Sistem ini melakukan deteksi dan logging per-frame tanpa mekanisme tracking temporal. Jika satu siswa terekam melewati kamera berkali-kali, setiap kemunculan akan dicatat sebagai event terpisah. Deduplikasi temporal dan konsolidasi atribut lintas frame berada di luar scope penelitian ini.

---

##### Ekstensi 1.3a: Intra-Camera Temporal Deduplication

**Problem yang diselesaikan:** Satu siswa lewat berulang kali di kamera yang sama (misal masuk-keluar kelas) saat ini dicatat sebagai event terpisah, menyebabkan duplikasi log dan inflasi statistik.

**Pendekatan:**
- Integrasi lightweight tracker (**Deep SORT** / **ByteTrack**) untuk assign ID persisten ke siswa yang sama dalam satu kamera
- Satu siswa yang lewat berkali-kali → digabung sebagai satu "student session" selama window waktu tertentu
- Tidak memerlukan training tambahan — tracker bekerja di atas output detector yang sudah ada

**Feasibility:** Relatif ringan (tidak ada model baru yang perlu ditraining), sehingga berpotensi menjadi **prioritas implementasi tertinggi** jika ada waktu sisa.

---

##### Ekstensi 1.3b: Track-Level Attribute Consolidation

**Problem yang diselesaikan:** Kualitas deteksi atribut bervariasi antar frame karena pose, jarak, dan oklusi. Frame pertama mungkin hanya mendeteksi 2 atribut, frame kelima mendeteksi 4 atribut. Tanpa konsolidasi, output per-frame tidak merepresentasikan kelengkapan sebenarnya.

**Pendekatan:**
- Agregasi deteksi atribut lintas frame **dalam satu student session** (hasil dari Ekstensi 1.3a)
- Strategi eksperimen pembanding:
  1. **Single-frame (baseline)** — ambil frame pertama/terakhir saja
  2. **Max-attributes** — pilih frame dengan jumlah atribut terbanyak
  3. **Confidence-weighted voting** — atribut dinyatakan "ada" berdasarkan rata-rata confidence atau konsistensi deteksi across frames
- Evaluasi: akurasi konsolidasi vs ground truth kelengkapan seragam siswa

**Mengapa ini penting:** Mengurangi false negative (atribut terlewat karena satu frame buruk) tanpa mencemari hasil dengan false positive satu kali.

---

##### Ekstensi 1.3c: Cross-Camera Re-Identification

**Problem yang diselesaikan:** Siswa yang berpindah antar kamera (misal gerbang depan → gerbang samping) saat ini di-re-face-recognize dari awal. Untuk skenario multi-gate monitoring, ini boros komputasi dan rentan gagal kalau wajah tidak terlihat jelas di kamera kedua.

**Pendekatan:**
- **OSNet** / **TransReID** untuk embedding appearance-based yang identifikasi siswa yang sama di kamera berbeda tanpa perlu face recognition ulang
- Memory bank untuk menyimpan embedding siswa aktif dalam window waktu tertentu

**Mengapa kompleksitas paling tinggi:**
- State management lintas kamera (memory bank, synchronization)
- Evaluasi cross-camera memerlukan dataset multi-view yang tidak trivial untuk dikumpulkan
- Relevan untuk skenario multi-gate monitoring (future scaling), bukan absensi single-gate

---

**Kenapa semua di-scope out dari TA ini:**
- Kompleksitas tambahan yang tidak proporsional dengan problem statement utama (deteksi kelengkapan seragam di gerbang)
- Face recognition + spatial association sudah memberikan **intra-frame per-student output** yang sufficient untuk skenario absensi single-gate
- Re-ID dan tracking temporal lebih relevan untuk **evolusi sistem** ke arah multi-gate atau monitoring intra-sekolah

**Strategi kontingensi jika ada waktu sisa:**
- Prioritas 1: Ekstensi 1.3a (paling feasible, dampak paling besar pada kualitas data)
- Prioritas 2: Ekstensi 1.3b (build on top of 1.3a, menunjukkan depth eksperimen)
- Prioritas 3: Ekstensi 1.3c (memerlukan setup multi-kamera yang substansial)

**Framing akademis:** Roadmap pengembangan yang terstruktur menunjukkan **awareness terhadap limitasi** dan **rencana evolusi bertahap** — sikap yang dihargai di sidang TA dibanding over-promising fitur yang tidak deliverable. Pemisahan tiga sub-ekstensi juga memperlihatkan bahwa "future work" bukan sekadar wishlist, tapi sudah dipikirkan dari sisi feasibility dan prioritas.

---

### Novelty 2: Mekanisme Asosiasi Spasial (Jarak Centroid)

#### Apa yang SiRapi Punya

SiRapi punya `_calculate_centroid()` di `rules_engine.py`, tapi digunakan untuk **zone geofencing** (cek apakah objek di dalam safe zone), **BUKAN** untuk mengasosiasikan atribut ke individu:

```python
# rules_engine.py, baris 67-71
def _calculate_centroid(self, bbox: BoundingBox) -> tuple[int, int]:
    """Menghitung titik tengah dari bounding box."""
    center_x = int((bbox['x1'] + bbox['x2']) / 2)
    center_y = int((bbox['y1'] + bbox['y2']) / 2)
    return center_x, center_y
```

Fungsi ini hanya dipakai di `_check_zone()` untuk cek apakah centroid ada di dalam rectangle safe zone. Tidak pernah digunakan untuk menghubungkan atribut ke wajah.

#### Apa yang Anda Tawarkan

Algoritma eksplisit yang:
1. Hitung centroid setiap wajah terdeteksi (Face₁, Face₂, ...)
2. Hitung centroid setiap atribut terdeteksi (Dasi₁, Topi₁, ...)
3. Hitung jarak Euclidean antara setiap atribut dengan setiap wajah
4. Assign atribut ke wajah terdekat (nearest neighbor / Hungarian algorithm)
5. Handle edge case: dua siswa berdekatan, atribut yang ambigu

```
Contoh skenario:
Frame berisi 2 siswa berdekatan, 1 dasi terdeteksi.

Face_A centroid = (200, 150)
Face_B centroid = (350, 160)
Dasi_1 centroid = (190, 250)

Jarak Face_A-Dasi_1 = √((200-190)² + (150-250)²) = 100.5
Jarak Face_B-Dasi_1 = √((350-190)² + (160-250)²) = 184.4

→ Dasi_1 di-assign ke Face_A (lebih dekat)
→ Face_B: "dasi tidak terdeteksi"
```

#### Mengapa Ini Novelty Orisinal

**Ini adalah kontribusi algoritma yang genuine.** Kebanyakan penelitian CV untuk uniform detection hanya melakukan:
- "Ada dasi di frame" (frame-level) → SiRapi level
- "Person X memakai dasi" (person-level tanpa multi-person handling)

Sangat sedikit yang secara eksplisit menyelesaikan masalah **multi-person attribute assignment** dengan spatial reasoning. Ini bisa menjadi:
- **Kontribusi ilmiah** yang layak di bagian "Metodologi" proposal TA
- **Pembeda utama** yang sulit ditiru tanpa effort signifikan
- **Problem statement** yang kuat: "Bagaimana mengasosiasikan atribut seragam yang terdeteksi ke individu yang tepat dalam skenario multi-person?"

#### Taksonomi 4 Pendekatan Algoritmik Asosiasi Spasial

Pada tataran algoritmik, terdapat empat pendekatan dominan yang perlu dipahami agar pemilihan dapat dijustifikasi secara akademis:

| # | Pendekatan | Mekanisme | Kompleksitas | Kapan Digunakan |
|---|-----------|-----------|--------------|-----------------|
| 1 | **Euclidean Centroid Distance** | Mengukur kedekatan antar-objek melalui titik pusat geometrisnya (Zhou et al., 2020) | O(n·m) | Objek tidak saling overlap, distribusi spasial relatif terpisah |
| 2 | **Greedy Nearest-Neighbor** | Memasangkan setiap deteksi dengan tetangga terdekatnya secara iteratif (Papais et al., 2024) | O(n·m) | Real-time dengan jumlah objek terbatas — sangat ringan |
| 3 | **Hungarian Algorithm** (linear assignment) | Menghasilkan pemasangan optimal secara global atas matriks biaya; tulang punggung SORT dan DeepSORT (Bewley et al., 2016; Wojke et al., 2017) | O(n³) | Butuh assignment optimal global, jumlah objek moderat |
| 4 | **IoU-based Matching** | Memanfaatkan tumpang-tindih spasial alih-alih jarak titik; dipakai ByteTrack dan turunannya (Zhang et al., 2022; Du et al., 2023) | O(n·m) | Objek saling bersarang atau memiliki area overlap signifikan |

> **Pilihan untuk sistem ini:** **Euclidean Centroid + Hungarian Algorithm** paling sesuai untuk domain verifikasi seragam. Pendekatan berbasis IoU kurang cocok karena bounding box wajah dan atribut seragam (dasi di leher, topi di kepala, sepatu di kaki) **secara anatomis tidak saling tumpang-tindih** — sehingga IoU mendekati nol dan tidak informatif sebagai biaya assignment.

#### Justifikasi Pemilihan Centroid untuk Domain Seragam Sekolah

Ini adalah argumen akademis kunci untuk menjawab "mengapa centroid, bukan IoU?":

**Karakteristik spasial unik domain seragam:**
- Topi → **atas wajah** (y_topi < y_face)
- Wajah → **area kepala** (referensi anchor)
- Dasi → **leher/dada**, di bawah wajah (y_dasi > y_face)
- Sabuk → **pinggang**, jauh di bawah (y_sabuk >> y_face)
- Sepatu → **kaki**, paling bawah (y_sepatu >>> y_face)

Karena setiap atribut memiliki **lokasi anatomis yang relatif tetap dan tidak overlap** dengan bounding box wajah, jarak Euclidean antara centroid wajah dan centroid atribut adalah metrik yang **informatif dan diskriminatif**. Dua atribut dari dua siswa yang berdekatan akan memiliki jarak centroid yang berbeda secara signifikan ke wajah masing-masing.

Ini berbeda dari skenario tracking kendaraan (ByteTrack) di mana bounding box bisa saling overlap dan IoU menjadi metrik yang lebih relevan.

**Anatomical Prior sebagai hard constraint tambahan:**

Untuk meningkatkan robustness, anatomical prior bisa ditambahkan sebagai pre-filter sebelum jarak dihitung:
- Jika `face_centroid = (x_f, y_f)`, dasi seharusnya ada di sekitar `(x_f, y_f + Δ_dasi)` — offset vertikal yang dapat dikalibrasi per setup kamera
- Atribut yang centroid-nya di luar rentang anatomis yang masuk akal di-filter sebelum masuk assignment
- Ini mengurangi false assignment pada skenario crowded scene

#### Potensi Peningkatan & Mitigasi Mis-Association

**1. Eksperimen komparatif metode assignment:**
- Bandingkan: Euclidean centroid (greedy NN) vs Euclidean centroid (Hungarian) vs biaya komposit (centroid + IoU)
- Ukur akurasi assignment di skenario: 1 orang, 2 orang berdekatan, 3 orang antrian (sesuai batasan masalah)
- Metrik evaluasi: assignment precision (% atribut yang di-assign ke individu yang benar)

**2. Mitigasi mis-association (strategi berlapis):**
- **Gating radius (distance threshold):** Jika jarak centroid atribut ke wajah terdekat melebihi `d_max`, atribut tidak di-assign ke siapapun (dianggap noise/oklusi) — mencegah false assignment jarak jauh
- **Kendala posisi relatif antar-kelas:** Constraint anatomis sebagai hard filter — dasi harus di bawah wajah (`y_dasi > y_face`), topi harus di atas wajah (`y_topi < y_face`), dst.
- **Biaya komposit centroid + IoU:** Untuk skenario crowded, kombinasikan sebagai weighted cost matrix: `cost = α × dist_euclidean + (1-α) × (1 - IoU)` (Wang et al., 2022; Du et al., 2023)

**Keputusan algoritma untuk TA ini:**

> **Pilihan utama: Greedy Nearest-Neighbor + distance threshold.** Dipilih karena: (1) kompleksitas komputasi rendah O(n·m) — cocok untuk real-time, (2) sederhana dan transparan — mudah dijelaskan, di-debug, dan divalidasi, (3) sufficient untuk batasan skenario ≤3 siswa per frame di mana konflik assignment sangat jarang terjadi. Distance threshold (gating) ditambahkan sebagai mitigasi false assignment jarak jauh.
>
> **Hungarian Algorithm O(n³)** — menghasilkan assignment optimal secara global — tetap disebutkan sebagai potensi enhancement untuk skenario yang lebih padat (>3 siswa). Jika eksperimen komparatif menunjukkan Hungarian memberikan hasil yang signifikan lebih baik pada data nyata, algoritma ini dapat menggantikan greedy NN. Keputusan final berdasarkan hasil eksperimen.

**3. Evaluasi pada edge case:**
- Dua siswa berdekatan (< 50px antar centroid wajah)
- Atribut terpotong di tepi frame (centroid bisa offset dari posisi sebenarnya)
- Oklusi parsial (satu siswa tertutup sebagian oleh siswa lain)

---

### Novelty 3: Rule Engine Verifikasi Kelengkapan

#### Apa yang SiRapi Punya

`rules_engine.py` SiRapi adalah **SafetyLogicEngine** untuk konteks HSE (Health, Safety, Environment):

```python
# rules_engine.py — baris 128-170 (inti logika)
if not is_operational_hours:
    if has_ppe:
        violation_type = "UNAUTHORIZED_OVERTIME"  # Lembur tanpa izin
    else:
        violation_type = "SECURITY_BREACH"         # Penyusup
else:
    if not has_ppe:
        violation_type = "NO_PPE"                  # Tidak pakai APD
```

Perhatikan: logikanya **binary** (`has_ppe: true/false`). Tidak ada:
- Konsep multi-atribut (topi DAN dasi DAN sabuk DAN sepatu)
- Aturan yang berbeda per hari
- Konsep "lengkap" vs "tidak lengkap" dengan detail atribut yang hilang
- Dispensasi atau exception per siswa

#### Apa yang Anda Tawarkan

```python
# Konsep Rule Engine Anda (pseudocode)
rules = {
    "senin": {
        "seragam": "osis",
        "required": ["topi pet", "dasi_biru_muda", "sabuk_hitam", "sepatu_hitam"]
    },
    "selasa": {
        "seragam": "batik",
        "required": ["sepatu_hitam"]
    },
    "jumat": {
        "seragam": "pramuka",
        "required": ["topi_pramuka", "hasduk", "sabuk", "sepatu_hitam"]
    }
}

def verify(student_id, detected_attributes, day):
    required = rules[day]["required"]
    missing = [attr for attr in required if attr not in detected_attributes]
    
    # Cek dispensasi
    if has_dispensation(student_id, missing):
        missing = filter_dispensed(missing, student_id)
    
    status = "LENGKAP" if len(missing) == 0 else "TIDAK LENGKAP"
    return {
        "student_id": student_id,
        "status": status,
        "missing": missing,        # ["dasi_merah"]
        "detail": "Dasi merah tidak terdeteksi"
    }
```

#### Kekuatan Pembanding

| Aspek | SiRapi Rules Engine | Rule Engine Anda |
|-------|-------------------|-----------------|
| Domain | HSE/PPE (industrial) | Seragam sekolah |
| Logic | Binary (has_ppe: true/false) | Multi-attribute (per atribut) |
| Schedule-aware | Jam operasional saja | Per hari + per event + per semester |
| Output | "NO_PPE" | "Dasi merah tidak terdeteksi" |
| Per-student | Tidak | Ya (+ dispensasi) |
| Konfigurasi | Hardcoded | Configurable via dashboard |

#### Insight Kritis

SiRapi rules engine-nya **bukan untuk sekolah** — ini jelas di-copy/adapt dari proyek monitoring keselamatan kerja (HSE). Variabel-variabelnya: `has_ppe`, `UNAUTHORIZED_OVERTIME`, `SECURITY_BREACH`. Ini artinya:

1. **Domain mismatch:** SiRapi mencoba menerapkan framework industrial safety ke konteks sekolah tanpa adaptasi yang proper
2. **Opportunity:** Rule engine Anda yang dirancang khusus untuk seragam sekolah Indonesia (dengan variasi per hari, per sekolah) adalah **domain-native solution**
3. **Untuk proposal TA:** Ini poin yang kuat — Anda bisa argue bahwa existing work menggunakan pendekatan generik, sedangkan Anda merancang solusi domain-spesifik

---

### Novelty 4: Dataset Realistis dari Lingkungan Sekolah Tunggal

#### Apa yang SiRapi Punya

```python
# ai-engine/scripts/train_model.py (dari code review)
# Target classes: Hardhat, Mask, NO-Hardhat, NO-Mask, 
#                 NO-Safety Vest, Person, Safety Cone, 
#                 Safety Vest, machinery, vehicle
```

SiRapi punya script training tapi dataset-nya untuk **PPE detection** (helm, vest, masker) — bukan seragam sekolah. Model yang di-deploy (`yolov8n.pt`) bahkan bukan hasil custom training, melainkan pre-trained COCO.

**Tidak ada satu pun gambar seragam sekolah di repository ini.**

#### Apa yang Anda Tawarkan

- Dataset dikumpulkan langsung dari lingkungan sekolah mitra (SMK, lab indoor)
- **Posisi kamera:** Indoor, ditempatkan di sekitar pintu masuk/keluar lab, mengarah ke pintu — kondisi pencahayaan indoor terkontrol, jarak siswa ke kamera relatif konsisten
- Mencakup kondisi operasional nyata: antrian masuk lab, variasi pencahayaan indoor, pose berbeda (dari depan/samping)
- Fokus awal: seragam OSIS, **Senin–Selasa** (hari lain masih perlu konfirmasi dengan sekolah mitra) — scope terkontrol
- Rencana ekspansi: hari lain + 2-3 jenis seragam (batik, pramuka) — future work
- **Label classes (target deteksi):** wajah siswa, **topi, dasi, sabuk, sepatu** — 4 atribut kelengkapan seragam yang dapat divisualisasikan secara eksplisit oleh kamera
- *Catatan scope:* Kemeja putih dan celana/rok abu **tidak dideteksi** (sulit dibedakan dari pakaian lain secara visual dari kamera pintu + bukan prioritas differentiator) — keputusan ini documented sebagai batasan penelitian

#### Kekuatan Pembanding

| Aspek | SiRapi | Sistem Anda |
|-------|--------|-------------|
| Dataset | Tidak ada (PPE dataset dari internet) | Koleksi langsung dari sekolah |
| Relevansi | Industrial safety classes | Seragam sekolah Indonesia |
| Variasi kondisi | Tidak terdokumentasi | Terdokumentasi (pencahayaan, pose, antrian) |
| Reproducibility | Tidak jelas sumber data | Jelas: sekolah X, tanggal Y, kondisi Z |

#### Insight untuk Proposal TA

Pengumpulan dataset sendiri dari lingkungan nyata adalah **kekuatan metodologis** yang sangat dihargai di sidang TA:
- Menunjukkan rigor penelitian (bukan hanya download dari Kaggle/Roboflow)
- Memberikan validitas eksternal: model diuji di lingkungan yang sama dengan deployment
- Fokus pada 1 jenis seragam dulu (OSIS) adalah keputusan scope yang bijak — reviewer akan lebih menghargai "1 seragam yang benar-benar work" daripada "3 seragam yang setengah-setengah"

**Rekomendasi penguatan:**
- Dokumentasikan proses pengumpulan data: jumlah gambar, jumlah siswa, variasi kondisi, rasio train/val/test
- Sertakan contoh gambar di proposal (dengan blur wajah untuk privasi)
- Jelaskan annotation protocol: siapa yang label, tool apa (Roboflow/CVAT/LabelImg), berapa lama

---

#### Ekstensi 4.1: Augmentation Strategy (Tiered Approach)

Karena dataset koleksi dari satu sekolah akan terbatas dibanding dataset public, strategi augmentasi menjadi kritis. Diadopsi pendekatan **bertingkat** untuk memastikan feasibility:

**Level 1 — Traditional Augmentation (Core, Wajib):**
- **Geometric:** Flip horizontal, rotation (±15°), random crop, scale jitter
- **Color:** Brightness (±20%), contrast (±20%), saturation, hue shift
- **Noise & blur:** Gaussian noise, motion blur, JPEG compression artifact
- **Occlusion:** Cutout, random erasing (simulasi backpack/rambut menutupi atribut)
- Tool: **Albumentations** library (standar komunitas CV)

**Level 2 — Semi-Advanced Augmentation (Recommended):**
- **MixUp:** Blend 2 gambar untuk robustness
- **CutMix:** Paste patch dari gambar lain ke gambar target
- **Mosaic augmentation:** 4 gambar digabung jadi 1 (built-in di YOLO family, sangat efektif untuk small object detection)
- **Copy-paste augmentation:** Copy atribut (dasi) dari gambar satu, paste ke gambar lain (balancing class)

**Level 3 — Generative Augmentation (Eksplorasi Tambahan, Opsional):**
- Stable Diffusion + ControlNet untuk generate variasi seragam dalam kondisi yang underrepresented
- Input: foto siswa berseragam lengkap → generate variasi tanpa topi/dasi (sintesis pelanggaran tanpa perlu foto pelanggaran real)
- **Status:** Eksplorasi tambahan — tidak di-commit sebagai core methodology
- **Risk:** Kompleks, butuh GPU kuat, distribution gap antara sintetis dan real

**Keputusan scope:**
- **Level 1 + 2 = wajib deliverable** untuk TA
- **Level 3 = bonus exploration** jika resource memungkinkan
- Proposal tidak over-promise generative augmentation — dihindari risk evaluator challenge "mana buktinya bekerja?"

**Keunggulan vs SiRapi:** SiRapi tidak mendokumentasikan strategi augmentation sama sekali — gap G-CV10. Sistem Anda memiliki strategi eksplisit yang bertingkat, sesuai prinsip reproducibility penelitian.

---

### Novelty 5: Edge Deployment (CCTV + Server/Laptop)

#### Apa yang SiRapi Punya

SiRapi punya:
- `docker-compose.prod.yml` dengan resource limits
- Dokumentasi hardware spec (Jetson Orin Nano, spesifikasi lengkap)
- Multi-stage Dockerfile yang optimized

**Tapi:** Tidak ada actual Jetson build, tidak ada TensorRT export, tidak ada inference benchmark di edge device. Semua masih di level Docker di PC biasa.

#### Apa yang Anda Tawarkan

Deployment pragmatis (keputusan final untuk TA):
- **CCTV:** IP Camera RTSP yang **disediakan sendiri** (dual-purpose: kebutuhan rumah + TA) — terhubung ke laptop peneliti/server sekolah jika diizinkan  via **LAN lokal**, tanpa internet untuk video streaming (`rtsp://[IP_ADDRESS]/stream`)
- **Server/Edge:** Laptop peneliti — menjalankan AI pipeline + FastAPI backend (in-process, zero overhead)
- **Dashboard:** React SPA (static build) di-host di **cloud/CDN** (Vercel/Netlify, gratis) — diakses via browser mana saja; FastAPI di edge di-expose via **Cloudflare Tunnel**
- **Tidak over-promise:** Tidak claim Jetson Orin jika belum punya

#### Kekuatan Pembanding

Ironisnya, pendekatan Anda **lebih jujur dan realistis** dibanding SiRapi:

| Aspek | SiRapi | Sistem Anda |
|-------|--------|-------------|
| Claim | "Edge deployment di Jetson Orin" | "Laptop/server sekolah + CCTV" |
| Realitas kode | Docker di PC, belum ada Jetson build | Akan di-deploy actual di sekolah |
| Hardware assumption | Jetson Orin Nano 8GB ($249) | Laptop yang sudah ada |
| Feasibility | Unproven (dokumentasi saja) | Pragmatis dan executable |

#### Insight

Untuk konteks TA prototipe (keputusan deployment yang sudah dikonfirmasi):
- **RTSP via LAN — tidak butuh internet:** CCTV dan laptop peneliti/server sekolah jika diijinkan sekolah cukup di-connect ke router/switch yang sama; video stream pure lokal, tidak ada data keluar ke internet (`cv2.VideoCapture("rtsp://[IP_ADDRESS]/stream")`)
- Jika punya GPU di laptop (bahkan GTX 1650), YOLO inference sudah sangat cepat
- **Nilai plus jika bisa:** Export model ke ONNX Runtime untuk inference lebih cepat di CPU
- **Dashboard Opsi B (keputusan final):** React SPA di-host di Vercel/Netlify (gratis, static files)+ FastAPI di edge di-expose via **Cloudflare Tunnel** (gratis, tanpa port forwarding, tanpa buka port router) — guru/admin akses dashboard dari device mana saja
- Demo dengan IP Camera RTSP nyata yang disediakan sendiri sudah jauh melebihi SiRapi yang masih di level Docker localhost + webcam biasa
---

#### Ekstensi 5.1: Privacy-Preserving Data Flow (Privacy by Design)

Edge deployment secara inheren memberikan keuntungan privasi, tapi sistem ini mengeksplisitkan **data flow commitment** yang fundamental menjawab concern UU PDP Indonesia dan parental consent.

**Commitment data flow:**

| Data | Lokasi | Keluar dari edge? | Catatan |
|------|--------|-------------------|----------|
| Raw video stream | Edge device (LAN only) | ❌ Tidak pernah | Endpoint `/video_feed/*` dibatasi IP lokal — tidak bisa diakses via Cloudflare Tunnel |
| Processed video (overlay bounding box) | Edge → Browser (LAN only) | ❌ Hanya dalam LAN | Streaming real-time di LAN, tidak disimpan di luar edge |
| Foto wajah siswa (enrollment & evidence) | Edge device storage | ❌ Tidak pernah | Citra mentah disimpan lokal untuk enrollment; tidak pernah dikirim ke luar edge |
| Face embeddings (vektor ArcFace 512-dim) | Edge device database | ❌ Tidak pernah | Vektor numerik hasil InsightFace — digunakan hanya untuk matching lokal di edge; bukan citra, tidak pernah disinkronisasi ke cloud |
| **Metadata deteksi** (nama, status, timestamp, atribut terdeteksi) | Edge → n8n → cloud dashboard | ✅ Hanya text metadata | Tidak mengandung gambar, video, maupun embedding — murni data teks hasil klasifikasi |
| Aggregated statistics | Edge → cloud dashboard | ✅ Text numeric | Rekapitulasi angka (jumlah pelanggaran, dsb.) — tidak mengandung data personal biometrik |

> **Catatan penting:** Meskipun pipeline menghasilkan **face embedding ArcFace** (vektor 512-dimensi), embedding ini **bukan citra wajah** dan **tidak pernah meninggalkan edge device**. Embedding digunakan murni untuk pencocokan identitas lokal (inference matching) — hasilnya hanya berupa `nama_siswa` (text) yang kemudian dikirim sebagai bagian dari metadata deteksi. Ini adalah contoh **data minimization dalam alur keluar**: yang keluar hanya identitas yang sudah terverifikasi (nama), bukan representasi biometrik mentahnya.

**Perbedaan fundamental vs SiRapi:**

SiRapi menyimpan evidence image plain di filesystem `/evidence/` (G-SC3) dan laporan PDF mengandung foto siswa tanpa anonymization (G-SC6). Jika server diakses, semua foto bocor.

Sistem Anda:
- **Foto siswa dan face embedding TIDAK PERNAH keluar dari edge device**
- Notifikasi WA/TG hanya berisi text: *"Ahmad Rizki (10-A) tidak memakai dasi pukul 07:15"* — tidak ada foto terlampir
- Laporan PDF hanya mengandung data text + grafik aggregate (optional: foto dengan wajah di-blur jika diminta)
- n8n + LLM memproses hanya metadata text, bukan raw images maupun face embedding

**Framing akademis:**
Ini adalah **privacy by design, bukan privacy by policy.** Bukan sekadar menambahkan kebijakan "kami janji tidak share foto" — secara arsitektural foto dan embedding memang tidak bisa share karena tidak pernah keluar dari edge. Yang keluar ke cloud hanya **text hasil klasifikasi**: nama siswa (dari matching lokal), status kelengkapan, dan timestamp. Pendekatan ini aligned dengan prinsip **data minimization** di UU PDP Pasal 16.

---

#### Ekstensi 5.2: Extensibility — FL-Ready Architecture (Future Work)

Arsitektur edge dengan privacy-preserving data flow secara natural **FL-ready** (Federated Learning-ready) — artinya ketika nanti sistem di-deploy ke multiple sekolah, transisi ke federated learning tidak butuh redesign fundamental.

**Design principles yang FL-ready:**
1. **Model weights terpisah dari data:** Edge device menyimpan data training lokal, model weights dapat di-serialize independen
2. **Standard model format:** Menggunakan format umum (ONNX / PyTorch state_dict) yang bisa di-aggregate
3. **Versioning infrastructure:** Setiap model punya version hash → mudah track provenance dalam FL round
4. **Sinkronisasi via API standar:** Edge dapat publish & subscribe ke model registry (jika nanti dibuat)

**Status untuk TA:**
- **BUKAN di-deliver di TA** (hanya single school, tidak ada FL client kedua)
- Disebutkan sebagai **design consideration** yang membuat sistem scalable ke arah FL
- Future work section mencatat: "Multi-school deployment dengan federated learning untuk improvement model tanpa sharing foto siswa antar sekolah"

**Framing akademis:**
Berbeda dengan claim ambisius tanpa bukti, pendekatan ini **menunjukkan awareness terhadap scalability** tanpa over-promise. Reviewer dapat melihat jalur pengembangan yang realistis.

**Keunggulan vs SiRapi:** SiRapi tidak memiliki strategi multi-school deployment sama sekali (G-AR5), apalagi dengan framework privacy-preserving seperti FL.

---

### Novelty 6: Cloud Bot (n8n + LLM + WA/Telegram)

#### Apa yang SiRapi Punya

Telegram bot multi-chat yang cukup mature:
- 3 file Go: `telegram.go`, `telegram_bot.go`, `telegram_sender.go`
- Registration flow, subscription filtering, rate limiting, image attachment
- Message deduplication via `SentMessageLog`
- Multi-chat management (bisa kirim ke banyak group)

**Kekuatan SiRapi di sini:** Implementasi Telegram bot-nya salah satu bagian paling solid.

**Kelemahannya:** Hanya Telegram. Orang tua siswa Indonesia mayoritas pakai WhatsApp.

#### Apa yang Anda Tawarkan

```
n8n (Workflow Automation)
├── Trigger: Webhook dari FastAPI backend
├── Process: LLM formatting (natural language summary)
├── Output A: WhatsApp Business API / WA Gateway
├── Output B: Telegram Bot (fallback/free)
└── Output C: Email (optional)
```

#### Kekuatan Pembanding

| Aspek | SiRapi (Telegram Bot) | Sistem Anda (n8n + LLM + WA) |
|-------|---------------------|-------------------------------|
| Platform | Telegram only | WhatsApp (primary) + Telegram (free fallback) |
| Audience reach | Admin/guru tech-savvy | Orang tua, guru, wali kelas (mainstream) |
| Message format | Template hardcoded di Go | LLM-generated natural language |
| **Multi-chat** | **✓ (hardcoded di Go)** | **✓ (via n8n workflow — multi-recipient, multi-group)** |
| Flexibility | Perlu coding untuk ubah flow | n8n visual workflow (low-code) |
| Cost | Free (Telegram API) | WA: perlu cloud service / Telegram: free |
| Arsitektur | Tightly coupled di backend Go | Decoupled via n8n webhook |

#### Insight & Rekomendasi

**Kekuatan unik Anda:** Penggunaan **LLM untuk formatting pesan** adalah sentuhan yang membedakan. Alih-alih template kaku:
```
❌ "[ALERT] Pelanggaran: no_dasi. Camera: cam01. Time: 07:15"
```
LLM bisa generate:
```
✅ "Selamat pagi, Bapak/Ibu. Anak Anda, Ahmad Rizki (10-A), terdeteksi 
   tidak memakai dasi saat masuk sekolah pagi ini pukul 07:15. 
   Mohon diperhatikan untuk besok. Terima kasih."
```

**Pertimbangan biaya WA:**
- WhatsApp Business API resmi: butuh BSP (Business Solution Provider), ada biaya per pesan
- Alternatif murah: WA Gateway open-source (Baileys/whatsapp-web.js) — tapi grey area ToS
- **Rekomendasi untuk TA:** Demo dengan Telegram dulu (gratis, mudah), jelaskan di proposal bahwa arsitektur n8n memungkinkan switch ke WA tanpa ubah backend. Ini menunjukkan **arsitektur yang extensible**

**Multi-chat tetap achievable via n8n:**
- n8n secara native mendukung **multi-recipient** — satu webhook trigger bisa broadcast ke banyak chat/group sekaligus
- Workflow bisa di-setup: 1 detection event → n8n split → kirim ke group guru, kirim ke WA wali kelas, kirim ke admin
- Ini artinya fitur multi-chat SiRapi **bisa di-match** tanpa harus coding manual di backend — cukup konfigurasi visual di n8n
- Bahkan lebih fleksibel: SiRapi multi-chat hanya Telegram, n8n multi-chat bisa **cross-platform** (WA group A + Telegram group B + Email ke kepala sekolah)

**n8n sebagai differentiator arsitektur:**
- Self-hosted (gratis) di laptop/server yang sama
- Visual workflow = mudah di-demo ke penguji
- Bisa ditambah logic tanpa coding (filter by severity, schedule summary, laporan harian otomatis, dll)
- **Laporan per hari:** n8n punya built-in cron trigger → setiap pukul 16:00 otomatis generate summary harian → broadcast ke semua group/chat yang terdaftar

---

### Novelty 7: Dashboard Pelaporan Terintegrasi

#### Apa yang SiRapi Punya

Dashboard SiRapi sangat lengkap (28+ halaman):
- Next.js 14 App Router + TailwindCSS + Framer Motion
- Recharts untuk grafik, React-Leaflet untuk heatmap
- PDF generation (Puppeteer + jsPDF), Excel export
- Real-time via WebSocket
- Dark/light theme, responsive design

**Tapi:** Banyak halaman yang overkill untuk konteks sekolah (annotation backlog, triage rules, security audit, custom dashboards). Dan yang lebih penting — **dashboard-nya menampilkan data dummy** karena AI engine belum menghasilkan data real.

#### Apa yang Anda Tawarkan

React SPA + FastAPI — terintegrasi end-to-end dengan model AI yang benar-benar jalan, dengan jumlah halaman yang akan berkembang sesuai kebutuhan fungsional. Ditambah sentuhan inovasi:
- **LLM Bot terintegrasi di dashboard** — sebagai interactive user guide, membantu guru/admin memahami fitur tanpa perlu baca manual
- Halaman-halaman inti yang mengambil best practice dari SiRapi (KPI, monitoring, reports) + fitur yang SiRapi tidak punya (per-student status, attendance log, LLM assistant)
- Kuantitas halaman bukan target — **kualitas integrasi data real** adalah prioritas

#### Kekuatan Pembanding

| Aspek | SiRapi (Next.js 14) | Sistem Anda (React SPA + FastAPI) |
|-------|---------------------|----------------------------------|
| Jumlah halaman | 28+ | Adjustment sesuai kebutuhan (berkembang iteratif) |
| Data yang ditampilkan | Dummy/statis (AI belum jalan) | Real dari model yang working |
| Complexity | Over-engineered untuk konteks sekolah | Right-sized, bertumbuh sesuai kebutuhan |
| User target | Generic (admin-centric) | Role-specific (guru, wali kelas) |
| In-app assistance | Tidak ada | **LLM Bot** terintegrasi sebagai user guide interaktif |
| Fitur unik dashboard | Banyak tapi tidak relevan (triage, annotation) | Per-student status, attendance log, LLM assistant |
| Demo impression | "Wah banyak fiturnya" tapi data kosong | "Setiap halaman hidup dengan data real" |

**Insight kritis:**

> Yang membedakan bukan jumlah halaman, tapi **apakah data di dashboard itu real.** Dashboard yang menampilkan data real dari AI pipeline yang working secara fundamental lebih bernilai dari dashboard 28 halaman dengan data dummy. Jumlah halaman Anda akan bertambah seiring development — yang penting setiap halaman yang ada menampilkan data yang meaningful dan actionable.

**Inovasi LLM Bot di Dashboard:**

Ini adalah diferensiasi UX yang SiRapi tidak miliki. Contoh use case:
- Guru baru buka dashboard → LLM bot: *"Selamat pagi, Bu. Hari ini ada 12 siswa terdeteksi tidak lengkap. Mau saya tampilkan daftar per kelas?"*
- Admin bingung fitur → ketik di chat: *"Cara export laporan mingguan?"* → LLM jawab dengan langkah-langkah kontekstual
- Wali kelas: *"Siapa saja di kelas 10-A yang sering tidak pakai dasi?"* → LLM query data dan jawab langsung

---

#### Ekstensi 7.1: Positive Reinforcement & Gamification (Future Work)

Alih-alih sistem yang hanya fokus pada "menghukum" pelanggaran, sistem Anda menambahkan **elemen positive reinforcement** untuk meningkatkan motivasi kedisiplinan siswa.

**Fitur gamification yang di-deliver (self-contained, tanpa integrasi sistem sekolah):**

| Fitur | Deskripsi | Implementasi |
|-------|-----------|--------------|
| **Discipline Streak** | Counter hari berturut-turut seragam lengkap | Simple query ke attendance log |
| **Digital Badge** | Badge virtual untuk milestone (7 hari, 30 hari, 100 hari disiplin) | Badge disimpan di student record |
| **Class Leaderboard** | Ranking kelas berdasarkan compliance rate mingguan/bulanan | Aggregate query, ditampilkan di dashboard |
| **Personal Milestone Notification** | Notifikasi ke orang tua saat anak dapat badge | Via n8n workflow (LLM-formatted positive message) |
| **Compliance Heatmap Personal** | Kalender personal menunjukkan hari-hari disiplin (hijau) vs tidak (merah) | Visualisasi di halaman student detail |

**Batasan scope yang dijaga:**
- ❌ **TIDAK** integrasi dengan sistem poin sekolah (butuh MoU, API politik, dll)
- ❌ **TIDAK** reward berupa hadiah fisik (logistically kompleks, out of scope)
- ✅ Reward intrinsik (badge digital, streak, leaderboard) yang self-contained dalam sistem
- ✅ Parent engagement via notifikasi positif (bukan hanya notifikasi pelanggaran)

**Framing psikologis:**
Pendekatan ini aligned dengan teori **operant conditioning** (B.F. Skinner) dan **self-determination theory** — motivasi intrinsik (autonomy, competence, relatedness) lebih efektif dibanding hanya hukuman. Untuk anak sekolah, positive reinforcement terbukti lebih membentuk perilaku jangka panjang.

**Keunggulan vs SiRapi:**
SiRapi memiliki class leaderboard sebagai **display statis** (Top Violators), tapi tidak ada mekanisme reward atau positive framing. Sistem Anda mengubah paradigma dari "surveillance" menjadi "engagement" — perubahan fundamental dalam tone produk.

**Sinergi dengan LLM Bot:**
LLM bot bisa generate pesan positif yang personal:
- *"Selamat! Ananda Ahmad konsisten berseragam lengkap 7 hari berturut-turut. Badge 'Minggu Disiplin' telah diraih."*
- Bukan hanya push notifikasi template, tapi variasi bahasa natural yang terasa personal

---

## 3. Analisis Arsitektur

### SiRapi: 3 Service Terpisah

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  Next.js 14  │◄──►│  Go Fiber    │◄──►│ Python FastAPI │
│  (Frontend)  │    │  (Backend)   │    │ (AI Engine)    │
│  Port 3000   │    │  Port 8080   │    │ Port 5000      │
│              │    │  SQLite      │    │ YOLOv8         │
│  App Router  │    │  JWT Auth    │    │ MJPEG Stream   │
│  SSR/CSR     │    │  WebSocket   │    │                │
└─────────────┘    └──────────────┘    └───────────────┘
     Node.js           Go binary          Python process
```

**Implikasi:**
- 3 runtime berbeda (Node.js, Go, Python) = 3 proses yang harus dikelola
- Go Fiber sebagai "middleman" antara frontend dan AI engine — menambah latency
- SSR (Server-Side Rendering) Next.js butuh Node.js server berjalan = lebih berat di edge
- Deployment complexity tinggi (3 Dockerfile, docker-compose orchestration)

### Sistem Anda: 2 Service — Opsi B Split Deployment (Keputusan Final TA)

```
        ☁️ CLOUD                               🏫 EDGE (LAN Sekolah)
┌──────────────────────────┐        ┌──────────────────────────────────┐
│  React SPA (Vite)        │        │  FastAPI (Unified Backend)        │
│  [Vercel / Netlify]      │◄─HTTPS─►│  [Laptop Peneliti]                │
│                          │  REST  │                                  │
│  Static files            │◄─WSS──►│  ├── /api/auth/*                 │
│  (HTML / JS / CSS)       │        │  ├── /api/detections/*            │
│  TailwindCSS             │        │  ├── /api/students/*              │
│  Recharts                │        │  ├── /api/reports/*               │
│  TypeScript              │        │  ├── /ws (WebSocket)              │
└──────────────────────────┘        │  ├── /video_feed/*                │
  Akses dari browser mana saja      │  │                                │
  (guru, admin, wali kelas)         │  ├── AI Pipeline:                 │
                                    │  │   ├── Detection (SLR)          │
  📡 via Cloudflare Tunnel          │  │   ├── Face Recog (SLR)         │
  (gratis, tanpa port forwarding)   │  │   └── Spatial Assoc.           │
                                    │  ├── PostgreSQL                   │
  📷 IP Camera (LAN, RTSP)         │  ├── JWT Auth                     │
  rtsp://192.168.x.x:554 ──────────►│  └── n8n Webhook trigger         │
  (pure LAN, tanpa internet)        └──────────────────────────────────┘
                                           1 Python process
```

### Perbandingan Arsitektur

| Aspek | SiRapi (Next.js + Go + Python) | Anda (React SPA + FastAPI) |
|-------|-------------------------------|---------------------------|
| **Jumlah service** | 3 (Node + Go + Python) | 2 (Static files + Python) |
| **Runtime** | 3 runtime berbeda | 1 runtime (Python) + static files |
| **Deployment complexity** | Tinggi (docker-compose wajib) | Rendah (1 proses + file hosting) |
| **Edge feasibility** | Berat (3 proses di laptop) | Ringan (1 proses utama) |
| **AI ↔ Backend latency** | HTTP call antar service | In-process (zero network overhead) |
| **Memory footprint** | ~2-3 GB (3 service) | ~1-1.5 GB (1 service + static) |
| **Development speed** | Perlu kuasai 3 bahasa | Cukup Python + TypeScript |
| **Frontend hosting** | Node.js server (butuh VPS/server) | Vercel/Netlify gratis (static CDN) |
| **Backend accessibility** | LAN lokal saja | Cloudflare Tunnel → akses dari mana saja |
| **Video input** | Webcam/RTSP (tidak terdokumentasi) | CCTV IP Camera RTSP via LAN (disediakan sendiri) |
| **Suitability untuk TA** | Over-engineered | Right-sized + pragmatis |

### Rekomendasi Frontend: React + TypeScript + Vite

| Opsi | Pro | Kontra | Verdict |
|------|-----|--------|---------|
| **React + Vite + TS** | Fast build, hot reload, SPA murni, ekosistem besar | Perlu handle routing sendiri (react-router) | **Recommended** |
| **Next.js 14 (App Router)** | SSR, file-based routing, API routes built-in | Overkill untuk SPA, butuh Node.js server | Too heavy |
| **Vue 3 + Vite** | Lebih mudah dipelajari | Ekosistem lebih kecil, kurang familiar | Viable alternative |
| **React CRA** | Familiar | Deprecated, slow build | Avoid |

**Stack rekomendasi final untuk frontend:**
```
React 18 + TypeScript + Vite
├── react-router-dom (routing)
├── TailwindCSS (styling - proven di SiRapi)
├── Recharts (charts - sama seperti SiRapi)  
├── Axios (HTTP client)
├── React Query / TanStack Query (data fetching + caching)
└── Zustand (state management - lebih simple dari Redux)
```

**Mengapa ini lebih baik dari Next.js untuk kasus Anda:**
1. **SPA murni** = build jadi static HTML/JS/CSS → bisa di-serve dari FastAPI langsung (`app.mount("/", StaticFiles(...))`) → **tidak perlu Node.js server sama sekali**
2. **Satu `uvicorn` process** serve API + AI + frontend static files = deployment paling sederhana
3. Untuk demo TA: `python main.py` → buka browser → selesai. Tidak perlu `docker-compose up` dengan 3 service.

### Arsitektur Backend (FastAPI Unified)

```python
# Struktur direktori yang direkomendasikan
backend/
├── main.py                 # FastAPI app entry point
├── config.py               # Configuration
├── database.py             # SQLAlchemy / SQLite setup
│
├── api/                    # REST API routes
│   ├── auth.py             # Login, JWT
│   ├── students.py         # CRUD siswa
│   ├── detections.py       # Log deteksi
│   ├── attendance.py       # Rekap absensi
│   ├── reports.py          # Generate laporan
│   └── webhook.py          # n8n trigger endpoint
│
├── ai/                     # AI Pipeline
│   ├── detector.py         # Multi-object detection inference (model dari SLR)
│   ├── face_recognition.py # Face recognition (model dari SLR)
│   ├── spatial_assoc.py    # Centroid-based association
│   ├── rule_engine.py      # Uniform verification rules
│   └── pipeline.py         # Orchestrator (detector → FR → assoc → rules)
│
├── models/                 # SQLAlchemy models
│   ├── student.py
│   ├── attendance.py
│   ├── detection.py
│   └── user.py
│
├── services/               # Business logic
│   ├── attendance_service.py
│   ├── notification_service.py
│   └── report_service.py
│
├── ws/                     # WebSocket
│   └── manager.py          # Connection manager
│
└── static/                 # React build output (served by FastAPI)
    ├── index.html
    ├── assets/
    └── ...
```

**Keunggulan arsitektur ini:**
- **AI pipeline in-process:** Tidak ada HTTP overhead antara AI dan backend
- **Single deployment:** `uvicorn main:app` menjalankan segalanya
- **Familiar untuk mahasiswa informatika:** Python end-to-end
- **Cukup untuk prototipe TA:** Tidak perlu microservice orchestration

---

## 4. Keunggulan Tersembunyi yang Belum Anda Sadari

Dari perbandingan mendalam, ada beberapa keunggulan yang mungkin belum Anda artikulasikan:

### 4.1 Domain Authenticity vs Domain Mismatch

**SiRapi punya masalah identitas.** Dari code review:
- `rules_engine.py` menggunakan terminologi HSE: `has_ppe`, `UNAUTHORIZED_OVERTIME`, `SECURITY_BREACH`
- Training script targetnya: `Hardhat, Mask, Safety Vest, machinery, vehicle`
- Dokumentasi mencampur konteks "industrial safety" dan "school uniform"
- Nama folder masih ada referensi ke "SmartAPD" (Alat Pelindung Diri) — proyek lama yang di-rebrand

**Insight:** SiRapi kemungkinan adalah proyek monitoring K3 (Keselamatan dan Kesehatan Kerja) yang di-**rebrand** menjadi sistem kerapihan sekolah untuk kompetisi. Domain-nya **tidak otentik**.

**Sistem Anda** dirancang dari awal (from scratch) untuk konteks seragam sekolah Indonesia. Ini memberikan:
- **Legitimasi akademis:** Research gap Anda benar-benar dari literatur uniform detection, bukan adaptasi dari PPE detection
- **Model yang relevan:** Training data dari sekolah nyata, bukan dataset industri
- **Rule engine yang native:** Dirancang untuk variasi seragam sekolah, bukan hardhat

### 4.2 Kontribusi Algoritmik (Asosiasi Spasial)

Novelty #2 (asosiasi spasial centroid) adalah **kontribusi metode** yang bisa dipublikasikan, bukan sekadar fitur produk. Ini bisa Anda framing sebagai:
- **Problem formulation:** "Multi-person attribute assignment in school uniform verification"
- **Proposed method:** "Centroid-based spatial association with distance threshold"
- **Evaluation metrics:** Precision, recall, F1-score pada skenario 1-person, 2-person, crowd

Ini sangat kuat untuk proposal TA karena dosen penguji mencari **kontribusi metode**, bukan hanya "saya pakai model X dan hasilnya bagus."

### 4.3 End-to-End Functional Pipeline vs Beautiful Empty Dashboard

SiRapi telah menginvestasikan effort besar di:
- 28 halaman dashboard dengan animasi Framer Motion
- Security audit trail yang comprehensive
- Multi-channel notification system

Tapi ketika Anda buka kode AI engine-nya:
```python
TARGET_CLASS_ID = 0  # TODO: ganti ini
```

**Pelajaran strategis:** Prioritas utama adalah **pipeline AI yang end-to-end functional.** Dashboard akan berkembang sesuai kebutuhan — yang krusial adalah data yang ditampilkan itu real. Jika demo Anda bisa:
1. Kamera menangkap frame siswa masuk gerbang
2. Model mendeteksi wajah + atribut seragam
3. Sistem mengidentifikasi siswa
4. Status "Ahmad Rizki — dasi: MISSING" muncul di dashboard
5. Notifikasi terkirim
6. LLM bot di dashboard menjawab pertanyaan guru tentang data

...maka itu sudah **menang telak** dari SiRapi yang dashboard-nya 28 halaman tapi data-nya dummy.

### 4.4 Arsitektur yang Jujur

SiRapi mengklaim:
- Edge deployment (Jetson) — belum ada build
- Absensi terintegrasi — tidak ada model Attendance
- AI detection — model belum trained

Sistem Anda jujur tentang scope:
- Laptop + CCTV (pragmatis, achievable)
- Fokus 1 jenis seragam dulu (OSIS) — realistic
- WhatsApp mungkin berbayar, jadi Telegram sebagai fallback

**Kejujuran scope = credibility di sidang TA.** Penguji lebih menghargai prototipe yang working dengan scope terbatas daripada proposal ambisius yang belum jalan.

---

## 5. Potensi Kelemahan & Mitigasi

Fairness analysis — aspek di mana SiRapi masih unggul dan bagaimana Anda bisa mitigasi.

| Aspek SiRapi Unggul | Detail | Mitigasi untuk Anda |
|---------------------|--------|---------------------|
| **Backend maturity** | Go Fiber backend sangat solid: JWT refresh token, session management, rate limiting, audit log, 60+ endpoints | FastAPI punya library untuk semua ini (python-jose, slowapi, dll). Untuk TA, tidak perlu 60 endpoints — 15-20 sudah cukup. Fokus endpoint yang dipakai. |
| **WebSocket implementation** | Real-time detection feed yang proven | FastAPI WebSocket support native. Implementasi lebih sederhana tapi functional. |
| **Telegram bot** | Multi-chat, subscription filtering, deduplication — sangat polished | n8n juga mendukung **multi-chat natively** (multi-recipient, multi-group, bahkan cross-platform WA+TG sekaligus). Feature-parity tercapai, bahkan lebih fleksibel karena low-code. Ditambah LLM formatting dan reach WA yang lebih luas. |
| **Documentation volume** | 19 file dokumentasi, 12,000+ baris | Untuk TA, Anda punya buku proposal. Dokumentasi kode cukup README + docstring. |
| **Docker deployment** | Multi-stage builds, health checks, resource limits | Untuk demo TA, `python main.py` sudah cukup. Docker bisa jadi "future work." |
| **Dashboard polish** | 28 halaman, Framer Motion, dark mode, Leaflet maps | Gunakan component library (shadcn/ui atau Ant Design) untuk polish cepat. Jumlah halaman akan berkembang sesuai kebutuhan. Keunggulan diferensiasi: **LLM bot terintegrasi** sebagai user guide — fitur yang SiRapi tidak punya. |
| **Security features** | Rate limiting, audit log, CORS, bcrypt, session revocation | Implementasi core security saja: JWT + bcrypt + CORS. Audit log dan rate limiting bisa simplified. Ini TA prototipe, bukan production deployment. |

### Risiko Teknis yang Perlu Diantisipasi

| Risiko | Probabilitas | Impact | Mitigasi |
|--------|-------------|--------|----------|
| **Face recognition accuracy rendah di kondisi nyata** | Medium | High | Kontrol kondisi: pencahayaan cukup, jarak kamera optimal (2-3m), enrollment foto berkualitas. Sediakan fallback manual. |
| **Spatial association gagal di kerumunan (>5 orang)** | Medium | Medium | Set threshold jarak maksimum. Jika terlalu crowded, fallback ke frame-level detection. Dokumentasikan sebagai limitation. |
| **CCTV sekolah tidak support RTSP** | Medium | High | Sediakan kamera sendiri (webcam USB / IP camera murah). Jangan bergantung 100% pada infrastruktur sekolah. |
| **Dataset terlalu kecil untuk training** | Medium | High | Augmentasi agresif (flip, rotate, brightness, crop). Minimum 500-1000 gambar per class. Transfer learning dari COCO weights. |
| **Demo di sidang gagal karena hardware** | Low | Critical | Test berulang kali di laptop yang sama. Siapkan video rekaman demo sebagai backup. Siapkan dataset test untuk offline demo. |
| **FastAPI performance bottleneck (AI + API di 1 proses)** | Low | Medium | Gunakan background worker (asyncio) untuk inference. Proses 1 frame per 200-500ms sudah cukup untuk gate monitoring. |

---

## 6. Matriks Kekuatan Kompetitif Final

### Scoring (1-5): 1=Tidak ada, 2=Minimal, 3=Cukup, 4=Baik, 5=Excellent

| Dimensi | SiRapi | Anda | Catatan |
|---------|:------:|:----:|---------|
| **AI Pipeline (core)** | 1 | 4 | SiRapi: model belum trained, logic dummy. Anda: custom model (SLR) + FR + spatial association + Dapodik enrollment + continuous learning loop |
| **Face Recognition** | 1 | 4 | SiRapi: tidak ada. Anda: model dipilih via SLR & eksperimen untuk presisi terbaik |
| **Student Identification** | 1 | 4 | SiRapi: anonymous detection. Anda: named per-student via Dapodik import |
| **Spatial Association** | 1 | 4 | SiRapi: centroid hanya untuk zone check. Anda: attribute-to-face assignment |
| **Continuous Learning Loop** | 2 | 4 | SiRapi: infra ada tapi tidak connected. Anda: closed-loop hybrid edge-offline retraining |
| **Rule Engine (domain fit)** | 2 | 4 | SiRapi: HSE rules di-rebrand. Anda: native school uniform rules |
| **Dataset Quality & Augmentation** | 1 | 4 | SiRapi: no dataset, no augmentation strategy. Anda: real school env + tiered augmentation strategy |
| **Attendance Integration** | 1 | 3 | SiRapi: no Attendance model. Anda: linked to FR pipeline |
| **Notification Reach** | 3 | 4 | SiRapi: Telegram multi-chat. Anda: n8n multi-chat cross-platform (WA+TG) + LLM formatting + positive reinforcement |
| **Dashboard Completeness** | 5 | 4 | SiRapi: 28 pages tapi data dummy. Anda: focused + LLM bot + gamification + data real |
| **Backend Maturity** | 5 | 3 | SiRapi: Go Fiber, 60+ endpoints. Anda: FastAPI, right-sized |
| **Security Features** | 4 | 2 | SiRapi: comprehensive. Anda: core security (target production-grade di final) |
| **Real-time (WebSocket)** | 4 | 3 | SiRapi: proven. Anda: simpler but functional |
| **Edge Deployment** | 2 | 4 | SiRapi: docs only. Anda: pragmatic deployment + privacy-preserving data flow + FL-ready architecture |
| **Privacy Compliance** | 1 | 4 | SiRapi: evidence image plain, no framework. Anda: privacy by design, foto tidak keluar edge |
| **Docker/DevOps** | 4 | 2 | SiRapi: multi-stage, health checks. Anda: simple uvicorn (target mature di final) |
| **Documentation** | 4 | 2 | SiRapi: 19 docs. Anda: proposal TA + README |
| **Domain Authenticity** | 2 | 5 | SiRapi: rebrand dari HSE. Anda: native school context |
| **User Engagement (Gamification)** | 2 | 4 | SiRapi: leaderboard statis. Anda: streak + badge + positive reinforcement notification |
| **Academic Contribution** | 2 | 4 | SiRapi: engineering showcase. Anda: algorithmic novelty (spatial assoc) + SLR-backed model selection + continuous learning framework |
| **Feasibility (as prototype)** | 3 | 4 | SiRapi: over-engineered. Anda: right-sized for TA dengan roadmap jelas |
| | | | |
| **TOTAL** | **53/105** | **76/105** | **Anda unggul +23 poin (~22% lead)** |

### Radar Overview (Dimensi Inti)

```
                   AI Pipeline
                        5
                       /|\
                      / | \
     Privacy    4    /  |  \   4  Face Recog
     Compliance ----/   |   \----
               /   /    |    \   \
              4   /     |     \   4  Edge Deploy
     Gamifica  ----/    |    \----
     tion     /   /     |     \   \
             4   /      |      \   4  Continuous
     Domain    ---/  SiRapi|Anda \--   Learning Loop
     Authen   /   /   (---)|(---)  \
             5   /        |        \  4  Dataset+
              \   \       |       /     Augmentation
               \   \      |      /
                4  \      |     /  4  Notification
                 \   \    |    /
        Academic  ----\   |   /----
        Contrib    4    \ | /    4  Rule Engine
                          \|/
                           5
                     Spatial Assoc
```

**Interpretasi:** Setelah ekspansi 7 poin tambahan, kompetitif matrix berubah signifikan:

1. **Core intelligence dominance diperkuat:** AI Pipeline, Face Recognition, Spatial Association, Continuous Learning, dan Dataset+Augmentation semua di atas SiRapi dengan margin besar (3-4 point gap per dimensi).

2. **Privacy & engagement jadi dimension baru:** Privacy Compliance (+3 gap) dan User Engagement/Gamification (+2 gap) menambah dua aksis baru di mana Anda unggul, padahal dimension ini penting untuk konteks sekolah + UU PDP.

3. **Edge deployment flipped:** Dari sebelumnya 3 vs 2 (+1), sekarang 4 vs 2 (+2) karena privacy-preserving data flow + FL-ready architecture.

4. **SiRapi masih unggul di engineering peripherals:** Backend maturity, security features, Docker/DevOps, documentation — tapi ini peripherals, bukan core. Untuk TA R&D, core intelligence + novelty jauh lebih menentukan evaluasi.

5. **Academic contribution gap melebar:** Dari 2 vs 4 menjadi semakin kuat dengan adanya kontribusi metodologis (spatial association), framework (continuous learning loop), dan design principle (privacy by design, FL-ready).

**Takeaway strategis:**
- Anda unggul di **dimensi yang matter untuk penelitian akademis**: kontribusi metode, rigor SLR, domain authenticity, privacy framework
- SiRapi unggul di **dimensi yang matter untuk produk komersial matang**: polish, security enterprise, devops
- Konteks TA prototipe R&D = dimensi akademis > dimensi produk komersial

---

## 7. Rekomendasi Teknis untuk Proposal TA

### 7.1 Framing Novelty di Proposal

Gunakan framework ini untuk menulis bagian "Novelty" atau "Kebaruan Penelitian" di proposal:

> **Kebaruan 1 (Arsitektur Pipeline):**
> Penelitian ini mengintegrasikan face recognition dan multi-object detection dalam satu pipeline terpadu, di mana output akhir berupa status kelengkapan atribut per individu siswa — bukan laporan deteksi terpisah yang tidak terasosiasi dengan identitas. Berbeda dengan existing work [SiRapi] yang hanya melakukan person detection tanpa identifikasi individu, sistem yang diusulkan mampu menghasilkan output: "Siswa X — atribut Y tidak terdeteksi." Pemilihan model detection dan face recognition didasarkan pada Systematic Literature Review (SLR) dan eksperimen komparatif untuk mendapatkan presisi terbaik pada domain seragam sekolah.
>
> **Kebaruan 2 (Metode Asosiasi Spasial):**
> Penelitian ini mengusulkan mekanisme asosiasi spasial berbasis jarak centroid untuk menyelesaikan masalah "atribut milik siapa" dalam skenario multi-person. Algoritma ini menghitung jarak Euclidean antara centroid bounding box setiap atribut terdeteksi dengan centroid bounding box wajah, kemudian melakukan assignment menggunakan nearest-neighbor matching dengan distance threshold. Pendekatan ini belum ditemukan pada existing work yang dianalisis.
>
> **Kebaruan 3 (Rule Engine Domain-Specific):**
> Penelitian ini merancang rule engine yang sadar konteks jadwal sekolah — di mana aturan kelengkapan atribut berubah sesuai hari (OSIS, batik, pramuka) dan dapat mengakomodasi exception (dispensasi medis, event khusus). Existing work [SiRapi] menggunakan rule engine yang diadaptasi dari konteks K3 (Keselamatan dan Kesehatan Kerja) dengan logika binary (has_ppe: true/false), yang tidak sesuai untuk domain seragam sekolah yang memiliki multi-atribut dengan aturan bervariasi.

### 7.2 Positioning terhadap SiRapi di Proposal

Dalam bagian "Tinjauan Pustaka" atau "Penelitian Terkait":

> [SiRapi] membangun sistem monitoring kerapihan seragam berbasis YOLOv8 dengan arsitektur 3-tier (Next.js, Go Fiber, FastAPI). Sistem tersebut memiliki dashboard yang komprehensif (28 halaman) dan infrastruktur notifikasi multi-channel. **Namun**, berdasarkan analisis terhadap implementasi aktual, ditemukan beberapa keterbatasan signifikan:
>
> 1. Model deteksi masih menggunakan pre-trained weights (COCO/PPE), belum dilatih khusus untuk atribut seragam sekolah
> 2. Tidak memiliki face recognition sehingga tidak dapat mengidentifikasi individu siswa
> 3. Tidak memiliki mekanisme asosiasi atribut-ke-individu untuk skenario multi-person
> 4. Rule engine diadaptasi dari konteks industrial safety (K3) dengan logika binary, bukan dirancang untuk variasi seragam sekolah
> 5. Modul absensi yang terintegrasi dengan deteksi seragam belum diimplementasikan
>
> Penelitian ini bertujuan mengatasi kelima keterbatasan tersebut melalui [metode yang diusulkan].

### 7.3 Strategi Teknis: Apa yang Diambil dari SiRapi + Inovasi Tambahan

Berikut adalah fitur-fitur **yang layak diadopsi** dari SiRapi (best practice) beserta **inovasi tambahan** yang menjadi pembeda:

#### Halaman Dashboard yang Layak Diadopsi dari Kompetitor: SiRapi

| Halaman SiRapi | Relevansi | Adopsi / Adaptasi untuk Sistem Anda |
|---------------|-----------|-------------------------------------|
| **Dashboard KPI** (stats, trend, violation breakdown) | Tinggi | ✅ **Adopsi** — tambah: per-student stats, attendance rate, LLM bot widget |
| **CCTV Monitoring Grid** (live feed + status) | Tinggi | ✅ **Adopsi — Local-Only:** processed video feed (bounding box + label nama + status atribut) hanya dapat diakses dari LAN sekolah; remote dashboard hanya tampilkan metadata. Endpoint `/video_feed/*` dibatasi IP lokal via middleware |
| **Analytics/Reports** (trend chart, export PDF/Excel) | Tinggi | ✅ **Adopsi** — tambah: laporan absensi per siswa, format compatible Dapodik |
| **Alert Management** (severity, acknowledge workflow) | Sedang | **Simplifikasi** — integrasi dengan n8n, alert langsung ke WA/TG |
| **Student Directory** (violation history per siswa) | Tinggi | ✅ **Adopsi + Enhance** — gabung dengan attendance record + face enrollment data |
| **Settings/Config** (camera, notification, rules) | Sedang | **Adopsi — Simplified:** konfigurasi kamera (RTSP URL), notifikasi (token/recipient), threshold confidence. Konfigurasi jadwal multi-seragam & dispensasi → **Future Work** (scope TA: OSIS Senin–Selasa, aturan hardcoded) |
| Security Audit (login activity, session mgmt) | Rendah (untuk prototipe) | **Skip dulu** — implementasi di fase production maturity |
| Annotation Backlog, Triage Rules, Custom Dashboard | Rendah | **Skip** — enterprise features yang tidak relevan untuk konteks sekolah |
| Heatmap/Map (Leaflet) | Rendah-Sedang | **Optional** — implementasi jika ada multi-zone di sekolah |

#### Inovasi Dashboard yang TIDAK Ada di SiRapi

| Inovasi | Deskripsi | Differentiator |
|---------|-----------|----------------|
| **LLM Bot di Dashboard** | Chat widget terintegrasi — guru bisa tanya data/fitur dalam bahasa natural | UX yang SiRapi tidak punya; mengurangi learning curve drastis |
| **Per-Student Status View** | Halaman detail per siswa: foto, riwayat absensi, riwayat pelanggaran, trend, streak, badge | SiRapi tidak punya karena tidak ada face recognition |
| **Attendance Log** | Log absensi real-time (siapa masuk jam berapa, seragam lengkap/tidak) | SiRapi tidak punya model Attendance sama sekali |
| **Face Enrollment Page** | Interface untuk daftarkan wajah siswa baru (upload foto / capture dari webcam) + Dapodik bulk import | SiRapi tidak memiliki konsep ini |
| **Schedule-Aware Config** | Konfigurasi aturan seragam per hari langsung dari dashboard | SiRapi rules engine hardcoded untuk HSE — **untuk TA ini adalah Future Work**; scope dibatasi OSIS Senin–Selasa dengan rule hardcoded |
| **Live Detection Overlay** | Video feed + bounding box real-time + label nama siswa + status atribut | SiRapi hanya menampilkan raw YOLO bounding box tanpa identitas |
| **Review Queue (Connected to Retraining)** | UI untuk review low-confidence detection → label → trigger retraining | SiRapi punya backlog tapi tidak connected ke retraining |
| **Gamification Dashboard** | Personal streak counter, badge collection, class leaderboard, compliance heatmap | SiRapi hanya punya leaderboard statis tanpa reward mechanism |
| **Positive Reinforcement Feed** | Timeline notifikasi positif (badge achievement, streak milestone) | Paradigma fundamental berbeda: surveillance → engagement |

### 7.4 Strategi Pemilihan Model (SLR-Driven)

Karena model detection dan face recognition **belum ditentukan** dan akan dipilih via SLR + eksperimen, berikut kerangka teknis untuk evaluasi:

#### Kriteria Evaluasi Multi-Object Detection

| Kriteria | Metrik | Minimum Target | Catatan |
|----------|--------|---------------|---------|
| Akurasi deteksi | mAP@0.5 | >0.75 | Pada dataset seragam sekolah |
| Akurasi deteksi ketat | mAP@0.5:0.95 | >0.50 | Untuk fine-grained attribute |
| Inference speed | FPS | >15 FPS | Pada hardware target (laptop/edge) |
| Model size | Parameter count / MB | <50 MB | Untuk feasibility edge deployment |
| Small object detection | AP-small | >0.40 | Penting untuk atribut kecil (dasi, sabuk) |

#### Kriteria Evaluasi Face Recognition

| Kriteria | Metrik | Minimum Target | Catatan |
|----------|--------|---------------|---------|
| Verifikasi akurasi | Accuracy / TAR@FAR=0.01 | >95% | Pada dataset siswa enrolled |
| Kecepatan embedding | ms/face | <50ms | Per wajah per frame |
| Robustness | Accuracy drop di variasi kondisi | <5% degradation | Pencahayaan, angle, masker |
| Database scalability | Matching speed di N siswa | <10ms untuk N<1000 | Cukup untuk 1 sekolah |

#### Status Model (Hasil SLR + Eksperimen Awal)

**Detection — RF-DETR (Keputusan Sementara, Masih Eksperimen):**
- **RF-DETR Medium** — transformer-based detector dari Roboflow, dilatih + dievaluasi langsung di platform Roboflow(tapi sepertinya hanya awal saja, selanjutnya pretrain sendiri )
- Hasil eksperimen awal: beberapa metrik **di atas 95%** pada dataset seragam sekolah
- Ukuran model estimasi **< 50 MB** — memenuhi kriteria edge deployment
- **Masih akan dieksplorasi:** RF-DETR Small — jika akurasi masih memadai, ukuran lebih kecil = lebih ringan di edge
- Platform anotasi + training: **Roboflow** (anotasi, augmentasi, tapi selanjutnya pretrain sendiri )
- *Keputusan final (Medium vs Small) ditentukan setelah eksperimen perbandingan selesai*

**Face Recognition — ArcFace (Rencana Utama, Sedang Proses):**
- **ArcFace** (via InsightFace library) — state-of-the-art additive angular margin loss, TAR@FAR=1e-4 tinggi
- Dipilih berdasarkan dominasi di SLR: konsisten unggul di LFW, IJB-C, MegaFace benchmarks
- Implementasi: sedang dalam proses eksplorasi dan integrasi
- Kandidat fallback: FaceNet (jika ArcFace terlalu berat di edge), DeepFace (unified wrapper)
- *Model final dikonfirmasi setelah eksperimen pada dataset wajah siswa sekolah mitra*

> **Kekuatan pendekatan ini vs SiRapi:** SiRapi langsung memakai YOLOv8n tanpa justifikasi akademis (bahkan model-nya pre-trained COCO, bukan custom). Sistem Anda memilih model berdasarkan **evidence dari SLR dan eksperimen nyata** (RF-DETR sudah mencapai >95% pada dataset domain) — ini adalah standar rigor penelitian yang diharapkan di TA.

### 7.5 Arsitektur Teknis yang Direkomendasikan

#### Unified FastAPI Backend — Keunggulan vs SiRapi 3-Tier

```
┌─────────────────────────────────────────────────┐
│              FastAPI Unified Backend              │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ REST API    │  │ AI Pipeline │  │ WebSocket│ │
│  │             │  │             │  │          │ │
│  │ /api/auth   │  │ Detection   │  │ Live     │ │
│  │ /api/student│  │ (SLR model) │  │ detection│ │
│  │ /api/attend │  │     ↓       │  │ feed     │ │
│  │ /api/detect │  │ Face Recog  │  │          │ │
│  │ /api/report │  │ (SLR model) │  │          │ │
│  │ /api/config │  │     ↓       │  │          │ │
│  │             │  │ Spatial     │  │          │ │
│  │  → n8n      │  │ Association │  │          │ │
│  │  webhook    │  │     ↓       │  │          │ │
│  │             │  │ Rule Engine │  │          │ │
│  └─────────────┘  └─────────────┘  └──────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │ SQLite/PostgreSQL  │  Static Files (React)  │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
               1 Python process (uvicorn)
```

**Keunggulan arsitektur ini:**
- **AI pipeline in-process:** Zero network overhead antara detection, FR, dan backend logic
- **Single deployment:** `uvicorn main:app` — bandingkan dengan SiRapi yang butuh 3 service
- **Model-agnostic:** Pipeline menerima output dari model apapun — ganti model cukup swap di `detector.py` dan `face_recognition.py`
- **Scalable ke production:** Bisa dipecah jadi microservice nanti jika dibutuhkan, tapi untuk prototipe TA satu proses sudah optimal

#### Tech Stack Rekomendasi

**Backend:**
```
FastAPI + Uvicorn
├── SQLAlchemy (ORM) + Alembic (migration)
├── python-jose (JWT auth)
├── WebSocket native FastAPI
├── httpx (async HTTP untuk n8n webhook)
└── Model inference (library sesuai hasil SLR)
```

**Frontend:**
```
React 18 + TypeScript + Vite
├── react-router-dom (routing)
├── TailwindCSS (styling — proven di SiRapi, ekosistem besar)
├── Recharts (charts — proven di SiRapi)
├── Axios / React Query (data fetching + caching)
├── Zustand (state management — simpler than Redux)
└── LLM chat widget (custom component → call backend → LLM API)
```

**Notification & Automation:**
```
n8n (self-hosted)
├── Webhook trigger dari FastAPI
├── LLM node untuk formatting pesan natural
├── Multi-output: WA + Telegram + Email
├── Cron trigger untuk laporan harian
└── Visual workflow — mudah di-demo & di-configure
```

---

*Dokumen ini dibuat berdasarkan code review komprehensif terhadap SiRapi dan analisis novelty peneliti, April 2026.*
