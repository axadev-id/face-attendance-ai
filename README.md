# Face Recognition dengan Transfer Learning menggunakan InceptionResnetV1 dan MobileNetV2

## 👥 Anggota Kelompok

**Nama Kelompok:** kata-ichsan-nama-kelompoknya-bebas

**Anggota:**
1. [Fajrul Ramadhana Aqsa] - [122140118]
2. [Ichsan Kuntadi Baskara] - [122140117]
3. [Mychael Daniel N] - [122140104]

---


## 📋 Deskripsi Singkat

Proyek ini mengimplementasikan sistem **Face Recognition** menggunakan teknik **Transfer Learning** dengan dua arsitektur deep learning state-of-the-art: **InceptionResnetV1** (pretrained CASIA-WebFace) dan **MobileNetV2** (pretrained ImageNet). Sistem ini menggunakan **MTCNN** untuk deteksi wajah otomatis, **K-Fold Cross Validation** untuk evaluasi yang robust, dan berbagai teknik augmentasi data untuk meningkatkan generalisasi model.

**Dataset:** 70 kelas wajah dengan multiple images per kelas  
**Best Model:** InceptionResnetV1 - Validation Accuracy **98.12%**  
**Framework:** PyTorch, FaceNet-PyTorch, Scikit-learn

---

## 🔗 Link Penting

- **Notebook Jupyter:** `Bebas.ipynb`
- **Model Terbaik InceptionResnetV1:** [final_best_model.pth](https://github.com/axadev-id/face-attendance-ai/blob/main/final_best_model.pth)
- **Model Terbaik MobileNetV2:** [final_best_model_mobilenetv2.pth](https://github.com/axadev-id/face-attendance-ai/blob/main/final_best_model_mobilenetv2.pth)
- **🌐 Web Application (Demo):** [Face Recognition Attendance System](https://huggingface.co/spaces/axadragon/face-recognition-attendance)
- ** https://huggingface.co/spaces/axadragon/face-recognition-attendance **

---

## 📊 Dataset

### Struktur Dataset
```
dataset/
├── Train/                  # Dataset asli (beragam format: JPG, HEIC, WEBP)
│   ├── person_1/
│   ├── person_2/
│   └── ... (70 kelas)
└── Data_Cropped/          # Dataset hasil preprocessing
    ├── person_1/
    ├── person_2/
    └── ... (70 kelas)
```

### Preprocessing Pipeline
1. **Konversi Format:**
   - HEIC → JPG
   - WEBP → JPG
   - Mempertahankan kualitas dengan quality=95

2. **Face Detection & Cropping (MTCNN):**
   - Deteksi wajah menggunakan MTCNN (Multi-task Cascaded Convolutional Networks)
   - Confidence thresholds: [0.4, 0.5, 0.5]
   - Minimum face size: 20 pixels
   - Padding ratio: 20% untuk konteks wajah
   - Resize ke: **160x160 pixels**

3. **Data Splitting:**
   - **Train:** ~70% (untuk K-Fold CV)
   - **Validation:** Per fold dalam K-Fold
   - **Test:** Set terpisah (~10%)
   - Menggunakan **Stratified Split** untuk menjaga distribusi kelas

### Karakteristik Dataset
- **Jumlah Kelas:** 70 identitas wajah
- **Format Gambar:** RGB (3 channels)
- **Ukuran Target:** 160x160 pixels
- **Normalisasi:** Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]

---

## 🔬 Metodologi

### 1. Transfer Learning Strategy
**Transfer Learning** digunakan untuk memanfaatkan pengetahuan dari model pretrained:
- **Frozen Backbone:** Ekstraksi fitur dari model pretrained tetap dipertahankan
- **Fine-tuned Classifier:** Head layer baru dilatih dari scratch untuk kelas spesifik

### 2. Data Augmentation
Tiga level augmentasi tersedia (proyek ini menggunakan **Moderate Augmentation**):

**Moderate Augmentation (Recommended):**
- RandomResizedCrop (scale: 0.7-1.0)
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- RandomRotation (±10°)
- RandomErasing (p=0.1, area kecil)

### 3. Training Configuration
```python
BATCH_SIZE = 16
NUM_EPOCHS = 80 (max)
EARLY_STOPPING_PATIENCE = 10
K_FOLDS = 3

Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
Scheduler: CosineAnnealingLR (eta_min=1e-6)
Loss: CrossEntropyLoss (label_smoothing=0.1)

Gradient Accumulation: 2 steps
Warmup Epochs: 5
Gradient Clipping: max_norm=1.0
```

### 4. K-Fold Cross Validation
- **3-Fold Stratified K-Fold** untuk evaluasi robust
- Setiap fold dilatih dengan model fresh (no weight leakage)
- Model terbaik dipilih berdasarkan validation loss terendah

---

## 🏗️ Arsitektur Model

### 6.1 InceptionResnetV1

**Pretrained:** CASIA-WebFace (dataset wajah Asia ~500K images, 10K identitas)

#### Architecture Overview
```
InceptionResnetV1 Backbone (Frozen)
├── Stem (Conv + BN + ReLU)
├── Inception-Resnet Blocks A, B, C
└── Global Average Pooling (512 features)

Custom Classifier Head (Trainable)
├── BatchNorm1d(512)
├── Dropout(p=0.6)
├── Linear(512 → 512)
├── BatchNorm1d(512)
├── ReLU()
├── Dropout(p=0.4)
└── Linear(512 → 70 classes)
```

#### Statistik Model
- **Total Parameters:** ~23M
- **Trainable Parameters:** ~300K (untuk 70 kelas)
- **Input Size:** 160x160x3
- **Output:** 70 classes (softmax)

#### Keunggulan
- ✅ Pretrained pada dataset wajah (domain-specific)
- ✅ Proven architecture untuk face recognition
- ✅ Feature extraction sangat kuat
- ✅ High accuracy pada validation (98.12%)

---

### 6.2 MobileNetV2

**Pretrained:** ImageNet (dataset general ~1.2M images, 1K classes)

#### Architecture Overview
```
MobileNetV2 Backbone (Frozen)
├── Initial Conv Layer
├── Inverted Residual Blocks (Depthwise Separable Conv)
└── Conv 1x1 + Global Average Pooling (1280 features)

Custom Classifier Head (Trainable)
├── Dropout(p=0.5)
├── Linear(1280 → 512)
├── ReLU()
├── Dropout(p=0.4)
└── Linear(512 → 70 classes)
```

#### Statistik Model
- **Total Parameters:** ~2.9M
- **Trainable Parameters:** ~690K (untuk 70 kelas)
- **Input Size:** 160x160x3
- **Output:** 70 classes (softmax)

#### Keunggulan
- ✅ Model ringan dan efisien (~87% lebih kecil dari InceptionResnetV1)
- ✅ Training lebih cepat dengan resource lebih sedikit
- ✅ Cocok untuk deployment di mobile/edge devices
- ✅ Competitive accuracy (96.78%)

---

## 📈 Hasil Evaluasi

### Perbandingan Model Terbaik

| Metric | InceptionResnetV1 | MobileNetV2 |
|--------|-------------------|-------------|
| **Pretrained On** | CASIA-WebFace (Asian Faces) | ImageNet (General) |
| **Best Fold** | Fold 1 | Fold 3 |
| **Train Loss** | 0.863650  | 1.399578 |
| **Train Accuracy** | 98.67% | 40.43% |
| **Validation Loss** | 1.391406 | 2.8626 |
| **Validation Accuracy** | 98.12% | 96.78% |
| **Total Parameters** | 23M | 2,9M |
| **Trainable Parameters** | 300K | 690K |
| **Training Time/Epoch** | ~45s | ~28s |

### 🏆 Model Terbaik: **InceptionResnetV1**
- Validation Accuracy: **98.12%**
- Validation Loss: **1.391406**
- File: `final_best_model.pth`

---

## 🔍 Analisis Kinerja Model

### InceptionResnetV1 Performance
**Kelebihan:**
- ✅ Akurasi tertinggi (98.12% validation accuracy)
- ✅ Pretrained pada dataset wajah Asia (domain-specific advantage)
- ✅ Ekstraksi fitur wajah sangat kuat
- ✅ Robust terhadap variasi pose, lighting, dan ekspresi

**Kekurangan:**
- ⚠️ Model lebih besar (23M parameters)
- ⚠️ Training time lebih lama (~45s/epoch)
- ⚠️ Membutuhkan resource komputasi lebih banyak

### MobileNetV2 Performance
**Kelebihan:**
- ✅ Model ringan dan efisien (3.5M parameters)
- ✅ Training cepat (~28s/epoch)
- ✅ Cocok untuk deployment di edge devices
- ✅ Competitive accuracy (96.78%)

**Kekurangan:**
- ⚠️ Accuracy sedikit lebih rendah dari InceptionResnetV1
- ⚠️ Pretrained pada ImageNet (general domain, bukan face-specific)
- ⚠️ Feature extraction kurang optimal untuk wajah

### Rekomendasi Penggunaan
- **High Accuracy Required:** Gunakan **InceptionResnetV1** (server deployment)
- **Mobile/Edge Deployment:** Gunakan **MobileNetV2** (mobile apps, IoT devices)
- **Balanced:** Tergantung trade-off accuracy vs efficiency

---

## 📊 Visualisasi

### 9.1 Kurva Pembelajaran (Learning Curves)

#### InceptionResnetV1
**Training Progress (3 Folds):**
- Loss menurun stabil dari ~2.5-3.0 ke ~0.86 (train) dan ~1.39 (validation)
- Validation accuracy mencapai 98.12% di fold terbaik (Fold 1)
- Tidak ada overfitting signifikan (gap Train-Val kecil)
- Early stopping triggered pada epoch bervariasi per fold

**Karakteristik:**
- Konvergensi cepat dalam 10-20 epoch pertama
- Plateau pada epoch 30-50
- Warmup LR (5 epochs) membantu stabilitas training

#### MobileNetV2
**Training Progress (3 Folds):**
- Loss menurun dari ~3.5-4.0 ke ~1.40 (train) dan ~2.86 (validation)
- Validation accuracy mencapai 96.78% di fold terbaik (Fold 3)
- Training lebih cepat (parameter lebih sedikit)
- Convergence lebih halus dengan gradient accumulation

---

### 9.2 Confusion Matrix

#### InceptionResnetV1 Confusion Matrix
**Observasi:**
- Diagonal dominan (prediksi benar tinggi ~98%)
- Misclassification sangat minimal (~1-2%)
- Beberapa kelas dengan similarity tinggi sedikit ter-confused
- Overall precision/recall sangat baik (98.12%)

#### MobileNetV2 Confusion Matrix
**Observasi:**
- Diagonal masih dominan (~96.78% correct)
- Slight increase dalam misclassification vs InceptionResnetV1 (~3-4%)
- Kelas dengan variasi tinggi lebih prone to error
- Masih excellent performance untuk mobile model

---

## 📄 Laporan Klasifikasi Lengkap

### Classification Report Summary

**InceptionResnetV1 (Aggregated 3-Folds):**
```
Overall Metrics:
- Accuracy: 98.12%
- Precision: ~98%
- Recall: ~98%
- F1-Score: ~98%
- Support: Total validation samples dari semua folds

Per-Class Performance:
- Mayoritas kelas: F1-Score > 95%
- Top performing classes: F1-Score = 100%
- Low-performing classes: Classes dengan data sangat sedikit
```

**MobileNetV2 (Aggregated 3-Folds):**
```
Overall Metrics:
- Accuracy: 96.78%
- Precision: ~96-97%
- Recall: ~96-97%
- F1-Score: ~96-97%
- Support: Total validation samples dari semua folds

Per-Class Performance:
- Mayoritas kelas: F1-Score > 93%
- Top performing classes: F1-Score > 98%
- Low-performing classes: Similar pattern dengan InceptionResnetV1
```

---

## 🚀 Cara Menjalankan Proyek

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (untuk GPU acceleration)
```

### Installation
```bash
# Clone atau download repository
cd end

# Install dependencies
pip install torch torchvision
pip install facenet-pytorch
pip install pillow pillow-heif
pip install matplotlib seaborn
pip install scikit-learn
pip install tqdm pandas numpy
```

### Struktur Direktori Required
```
end/
├── dataset/
│   └── Train/          # Masukkan dataset di sini (per kelas)
│       ├── person_1/
│       ├── person_2/
│       └── ...
├── Bebas.ipynb         # Main notebook
└── README.md
```

### Menjalankan Training
1. **Open Notebook:**
   ```bash
   jupyter notebook Bebas.ipynb
   ```

2. **Execute Cells Secara Berurutan:**
   - **Cell 1-4:** Preprocessing (konversi format, face detection, cropping)
   - **Cell 5-9:** Data splitting & augmentation setup
   - **Cell 10-17:** Training InceptionResnetV1 (K-Fold CV)
   - **Cell 18-24:** Training MobileNetV2 (K-Fold CV)
   - **Cell 25:** Perbandingan hasil kedua model

3. **Load Pretrained Model untuk Inference:**
   ```python
   # Load InceptionResnetV1 terbaik
   from facenet_pytorch import InceptionResnetV1
   import torch
   
   model = InceptionResnetV1(classify=True, num_classes=70)
   model.load_state_dict(torch.load('final_best_model.pth'))
   model.eval()
   
   # Inference
   # ... (preprocessing image → model prediction)
   ```

### Menjalankan Inference
```python
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(classify=True, num_classes=70)
model.load_state_dict(torch.load('final_best_model.pth'))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load & predict
img = Image.open('path/to/image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Predicted class: {predicted.item()}")
```

---

## 📁 Struktur Direktori

```
end/
├── dataset/
│   ├── Train/                      # Dataset asli (raw images)
│   │   ├── person_1/
│   │   ├── person_2/
│   │   └── ... (70 folders)
│   └── Data_Cropped/               # Dataset hasil preprocessing
│       ├── person_1/
│       ├── person_2/
│       └── ... (70 folders)
│
├── Bebas.ipynb                     # Main Jupyter notebook
├── README.md                       # Dokumentasi lengkap (this file)
│
├── final_best_model.pth            # InceptionResnetV1 terbaik
├── final_best_model_mobilenetv2.pth # MobileNetV2 terbaik
├── model_info.json                 # Metadata InceptionResnetV1
├── model_info_mobilenetv2.json     # Metadata MobileNetV2
│
├── best_model_fold_1.pth           # Checkpoint per fold
├── best_model_fold_2.pth
└── best_model_fold_3.pth
```

---

## 💡 Kesimpulan

### Hasil Utama
1. **InceptionResnetV1** mencapai **98.12% validation accuracy** (validation loss: 1.391406), menjadikannya model terbaik untuk face recognition di dataset ini
2. **MobileNetV2** mencapai **96.78% accuracy** (validation loss: 2.8626) dengan efisiensi parameter **~87% lebih rendah** (3.5M vs 23M), excellent untuk deployment
3. **Transfer Learning** terbukti sangat efektif - pretrained pada CASIA-WebFace memberikan keunggulan signifikan untuk face recognition
4. **K-Fold Cross Validation** memastikan evaluasi robust dan menghindari overfitting


---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:
1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add some improvement'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Buat Pull Request


## 📝 Lisensi

Proyek ini dibuat untuk tujuan **akademik** (Tugas Kuliah - Deep Learning Semester 7).

**Dependencies Licenses:**
- PyTorch: BSD License
- FaceNet-PyTorch: MIT License
- Scikit-learn: BSD License

**Pretrained Models:**
- InceptionResnetV1 (CASIA-WebFace): Research purpose only
- MobileNetV2 (ImageNet): Apache 2.0

---


<div align="center">

**⭐ Jika proyek ini bermanfaat, jangan lupa berikan star! ⭐**

Made with ❤️ for Deep Learning Course - Semester 7

</div>
