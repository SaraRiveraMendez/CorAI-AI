# CorAI-AI
# AFDB Feature Extraction

This repository contains a lightweight and memory-efficient Python script for extracting features from the **MIT-BIH Atrial Fibrillation Database (AFDB)** directly from [PhysioNet](https://physionet.org/content/afdb/1.0.0/).

The script downloads and processes ECG signals in time blocks, automatically detecting the **MLII (Lead II)** channel and computing basic signal statistics for further analysis or model training.

---

## ✨ Features

- ✅ **Automatic MLII (Lead II) channel detection** — no need to specify channel indices manually.  
- ⚙️ **Block-based processing** — handles large PhysioNet records without consuming excessive memory.  
- 📊 **Feature extraction** — mean, standard deviation, RMS, zero-crossing rate, and heart-rate estimate per block.  
- 💾 **Incremental CSV output** — appends results while processing, suitable for multi-user environments.  
- 🧩 **WFDB-based PhysioNet access** — no local downloads required.

---

## 📦 Installation

Make sure you have Python ≥ 3.8 installed. Then install the required dependencies:

```bash
pip install wfdb numpy pandas scipy

