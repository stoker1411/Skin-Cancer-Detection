# ATLAS: Accurate Transfer Learning Analysis of Skin Lesions

**Skin Cancer Detection with Hybrid CNN-ML Classifiers**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

ATLAS is a computer vision research project that develops and evaluates a **hybrid approach** for skin cancer detection from dermatoscopic images. Instead of relying solely on computationally expensive end-to-end deep learning models, we combine **CNN feature extraction** with **traditional machine learning classifiers** to achieve high accuracy while maintaining interpretability and reducing resource requirements.

This work bridges the gap between deep learning performance and traditional ML explainability for practical clinical deployment in resource-constrained settings.

---

## 🎯 Key Objectives

- **Systematically compare** 3 CNN architectures (VGG16, ResNet50, EfficientNet) × 3 ML classifiers (SVM, Random Forest, XGBoost) = **9 hybrid combinations**
- **Achieve clinical-grade performance**: ≥85% accuracy, ≥80% sensitivity (minimize false negatives)
- **Reduce computational cost**: 50–70% training time reduction vs. end-to-end CNNs
- **Improve interpretability**: Feature importance analysis and explainability for clinical trust
- **Enable deployment**: Lightweight models suitable for edge/mobile devices

---

## 📊 Project Structure

```
atlas-skin-cancer/
├── README.md                    # This file
├── ROADMAP.md                   # Weekly milestones & task assignments
├── requirements.txt             # Python dependencies
├── data/
│   ├── ham10000/               # HAM10000 dataset (10,015 images, 7 classes)
│   └── preprocessing/          # Data augmentation & normalization scripts
├── src/
│   ├── feature_extraction.py   # CNN feature extraction (VGG16, ResNet50, EfficientNet)
│   ├── classifiers.py          # SVM, Random Forest, XGBoost training
│   ├── preprocessing.py        # Image resizing, normalization, hair removal
│   └── utils.py                # Helper functions (metrics, visualization)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline_cnn.ipynb   # End-to-end CNN baseline
│   └── 03_hybrid_pipeline.ipynb # Full hybrid model training
├── results/
│   ├── models/                 # Trained classifiers (.pkl, .h5)
│   └── metrics/                # Comparison tables, confusion matrices
├── docs/
│   └── proposal.pdf            # Midterm proposal
└── .gitignore
```

---

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone repository
git clone https://github.com/[team-org]/atlas-skin-cancer.git
cd atlas-skin-cancer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Download Data**
```bash
# HAM10000 dataset (automatic or manual download)
# https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
python src/download_data.py
```

### 3. **Run Baseline (End-to-End CNN)**
```bash
python src/train_baseline_cnn.py --epochs 20 --batch-size 32
```

### 4. **Feature Extraction**
```bash
python src/feature_extraction.py --cnn vgg16 --layer pool5
python src/feature_extraction.py --cnn resnet50 --layer avg_pool
python src/feature_extraction.py --cnn efficientnet --layer global_avg
```

### 5. **Train Hybrid Models**
```bash
python src/train_classifiers.py --features vgg16 --classifier svm
python src/train_classifiers.py --features resnet50 --classifier xgboost
python src/train_classifiers.py --features efficientnet --classifier rf
```

### 6. **Evaluate & Compare**
```bash
python src/evaluate.py --output results/comparison_table.csv
```

---

## 📈 Expected Results

| CNN | Classifier | Accuracy | Sensitivity | Specificity | Training Time | Inference Speed |
|-----|-----------|----------|-------------|------------|---------------|-----------------|
| VGG16 | SVM | ~85% | ~82% | ~87% | ~5 min | 45 img/s |
| ResNet50 | XGBoost | ~88% | ~85% | ~89% | ~8 min | 38 img/s |
| EfficientNet | RF | ~84% | ~80% | ~86% | ~12 min | 52 img/s |

*(Preliminary targets; actual results depend on implementation and hyperparameter tuning)*

---

## 📋 Dataset

**HAM10000** (Human Against Machine with 10,000 training images)
- **10,015 dermatoscopic images** from multiple sources
- **7 diagnostic categories**: Melanoma, Nevi, Basal Cell Carcinoma, Actinic Keratoses, Benign Keratosis, Dermatofibroma, Vascular Lesions
- **Class imbalance**: Addressed via SMOTE, class weights, stratified sampling
- **License**: Public domain; de-identified, ethics-approved

**Data Split**: 70% training, 15% validation, 15% test (stratified)

---

## 🔧 Technologies

- **Deep Learning**: TensorFlow 2.15, Keras
- **ML Classifiers**: scikit-learn 1.3, XGBoost 2.0
- **Data Processing**: NumPy, Pandas, OpenCV, imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Compute**: Google Colab Pro (Tesla T4/P100 GPU) or local GPU (8GB+ VRAM)

---

## 📅 Timeline

| Week | Milestone | Owner | Due Date |
|------|-----------|-------|----------|
| W1–W2 | Setup, dataset access, environment | All | Oct 28 |
| W3 | Baseline CNN on 10% data | Abdulaziz | Nov 4 |
| W4 | Feature extraction (3 CNNs) | Azizbek | Nov 11 |
| W5 | Classifier training & grid search | Abdulaziz, Azizbek | Nov 18 |
| W6 | Evaluation metrics & stats | Azizjon | Nov 25 |
| W7 | Ablation studies & interpretability | Azizjon | Dec 2 |
| W8 | Report, demo, code freeze | All | Dec 9 |

---

## 👥 Team

- **Abdulaziz Suvonkulov** (220456@centralasaian.uz) – Feature extraction & CNN training
- **Azizbek Yoqubov** (220517@centralasaian.uz) – **Coordinator** – Overall project management, classifier pipeline
- **Azizjon Qahhorov** (220536@centralasian.uz) – Evaluation metrics & interpretability analysis

---

## 📖 Documentation

- **Proposal**: See `docs/proposal.pdf` for full research rationale and methodology
- **ROADMAP**: See `ROADMAP.md` for weekly task assignments and progress tracking
- **Notebooks**: See `notebooks/` for step-by-step tutorials and exploratory analysis

---

## ✅ Contributions & Citation

This project is part of a Computer Vision course (CAU) research requirement. 

**To cite this work:**
```bibtex
@misc{atlas2024,
  author = {Suvonkulov, Abdulaziz and Yoqubov, Azizbek and Qahhorov, Azizjon},
  title = {ATLAS: Accurate Transfer Learning Analysis of Skin Lesions},
  year = {2024},
  howpublished = {\url{https://github.com/[team-org]/atlas-skin-cancer}}
}
```

---

## 📜 License

This project is released under the **MIT License**. See LICENSE file for details.

All datasets used are publicly available and properly cited. No sensitive personal information is collected or retained.

---

## 🤝 Contact & Issues

For questions, issues, or contributions:
- **Open an issue** on GitHub
- **Contact coordinator**: Azizbek Yoqubov (220517@centralasaian.uz)

---

## 🙏 Acknowledgments

- HAM10000 dataset creators: Tschandl, Rosendahl, Kittler
- Course instructors: Dr. I. Atadjanov & Dr. B. Kiani
- References: Esteva et al., Mahbod et al., Hosny et al., Khan et al.

---

**Last Updated**: October 2024 | **Status**: In Progress 🚧
