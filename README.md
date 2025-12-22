```markdown
# CorAI-AI: AFDB Supervised Rhythm Classification

This repository provides a complete pipeline for supervised classification of cardiac arrhythmias using the **MIT-BIH Atrial Fibrillation Database (AFDB)** from [PhysioNet](https://physionet.org/content/afdb/1.0.0/).

The system extracts advanced ECG features and trains a machine learning classifier to distinguish between different cardiac rhythms using real annotations from the AFDB database.

## Overview

The pipeline consists of two main components:

1. **Feature Extraction** (`afdb_dataset_loader.py`) - Extracts comprehensive ECG features from PhysioNet signals
2. **Supervised Classification** (`afdb_supervised_classification.py`) - Trains and evaluates a RandomForest classifier using AFDB rhythm annotations

## Cardiac Rhythms Classified

The system classifies four main rhythm types based on AFDB annotations:

- **AFIB**: Atrial Fibrillation
- **AFL**: Atrial Flutter
- **J**: AV Junctional Rhythm
- **N**: Other rhythms (primarily normal sinus rhythm)

## Features

### Data Processing
- Streaming mode for direct PhysioNet access without local downloads
- Block-based signal processing (default 60-second windows)
- Automatic handling of multi-channel ECG recordings
- Support for both streaming and local file modes

### Feature Extraction
- **Temporal features**: mean, standard deviation, median, min, max, RMS, zero-crossing rate
- **Heart rate variability (HRV)**: RR interval statistics (mean, std, SDNN, RMSSD, pNN50)
- **QRS morphology**: width, amplitude, and area measurements
- **Wavelet features**: multi-level decomposition with energy and statistical properties
- **R-peak detection**: simplified Pan-Tompkins algorithm

### Classification
- RandomForest classifier with balanced class weights
- Stratified train-test split (75/25)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Feature importance analysis
- PCA visualization of feature space

## Installation

Requirements: Python ≥ 3.8

Install dependencies:

```bash
pip install wfdb numpy pandas scipy scikit-learn matplotlib seaborn pywt joblib
```

## Usage

### Basic Training

Train a classifier using AFDB records with streaming mode:

```bash
python afdb_supervised_classification.py \
    --records 04015 04043 04048 \
    --pn-dir afdb \
    --block-sec 60 \
    --channel-idx 1 \
    --save-model \
    --results-dir results_supervised
```

### Command-Line Arguments

- `--records`: Space-separated list of AFDB record IDs (required)
- `--pn-dir`: Database directory or 'afdb' for streaming mode (default: 'afdb')
- `--block-sec`: Duration of each analysis block in seconds (default: 60)
- `--channel-idx`: ECG channel index to process, typically 0 or 1 (default: 1)
- `--save-model`: Save trained model and preprocessing objects
- `--results-dir`: Output directory for results (default: 'results_supervised')

### Using Local Files

If you have pre-downloaded AFDB files:

```bash
python afdb_supervised_classification.py \
    --records 04015 04043 \
    --pn-dir ./path/to/local/afdb \
    --channel-idx 1 \
    --save-model
```

## Output Files

The training process generates several files in the results directory:

### Model Files (when `--save-model` is used)

- `afdb_rhythm_classifier.joblib` - Trained RandomForest classifier
- `afdb_scaler.joblib` - StandardScaler for feature normalization
- `afdb_label_encoder.joblib` - LabelEncoder for rhythm class mapping

These files can be loaded for inference on new data:

```python
import joblib
classifier = joblib.load('results_supervised/afdb_rhythm_classifier.joblib')
scaler = joblib.load('results_supervised/afdb_scaler.joblib')
encoder = joblib.load('results_supervised/afdb_label_encoder.joblib')
```

### Visualization Files

- `confusion_matrix.png` - Confusion matrix showing prediction accuracy per class
- `feature_importance.png` - Top 20 most important features for classification
- `pca_visualization.png` - 2D PCA projection of feature space colored by rhythm type
- `class_distribution.png` - Bar chart of class distribution in the dataset

### Report Files

- `metrics_TIMESTAMP.json` - Complete metrics in JSON format including:
  - Dataset information (samples, features, classes)
  - Class distribution
  - Performance metrics (accuracy, F1-score, precision, recall)
  - Timestamp and configuration parameters

- `report_TIMESTAMP.txt` - Human-readable text report containing:
  - Dataset summary
  - Model performance metrics
  - Detailed classification report per class

- `training.log` - Complete training log with diagnostic information

## Architecture

### Feature Extraction Pipeline

1. **Signal Loading**: Streams ECG signals from PhysioNet or loads from local files
2. **Annotation Loading**: Retrieves rhythm annotations from AFDB .atr files
3. **Block Segmentation**: Divides signals into fixed-duration blocks
4. **R-peak Detection**: Applies Pan-Tompkins algorithm for QRS detection
5. **Feature Computation**: Extracts 30+ features per block including:
   - Basic statistics (7 features)
   - RR interval metrics (8 features)
   - QRS morphology (6 features)
   - Wavelet decomposition (15 features across 5 levels)

### Classification Pipeline

1. **Data Preparation**:
   - Loads features with corresponding rhythm labels
   - Removes metadata columns
   - Handles missing values with median imputation

2. **Preprocessing**:
   - StandardScaler normalization
   - Label encoding for rhythm classes

3. **Training**:
   - RandomForest with 300 trees
   - Balanced class weights for imbalanced data
   - Stratified train-test split

4. **Evaluation**:
   - Per-class metrics (precision, recall, F1-score)
   - Overall accuracy
   - Confusion matrix analysis

## Performance Considerations

### Memory Efficiency
- Block-based processing prevents loading entire records into memory
- Streaming mode eliminates need for local storage
- Feature extraction processes one block at a time

### Computational Requirements
- Feature extraction: ~1-2 seconds per 60-second block
- Model training: Scales with number of blocks and features
- Recommended: Multi-core CPU for parallel RandomForest training (n_jobs=-1)

### Class Imbalance
- AFDB contains unbalanced rhythm distributions
- System uses balanced class weights in RandomForest
- Metrics computed with zero_division=0 for robust reporting

## Known Limitations

1. **Annotation Dependencies**: Requires valid .atr annotation files from AFDB
2. **Channel Selection**: Assumes ECG channels follow AFDB naming conventions (MLII, ECG1, ECG2)
3. **Block Boundaries**: Rhythm changes within blocks are assigned based on majority voting
4. **Minority Classes**: Some rhythm types may have insufficient samples for reliable prediction

## Troubleshooting

### No Files Saved
- Check directory write permissions: `os.access(results_dir, os.W_OK)`
- Verify absolute path is used for results directory
- Check available disk space

### Missing Classes in Predictions
- Indicates insufficient training samples for minority classes
- Consider collecting more data or adjusting class_weight parameter
- Check class distribution in output logs

### Feature Extraction Errors
- Verify WFDB can access PhysioNet (requires internet connection)
- Check record IDs are valid AFDB records
- Ensure channel_idx exists in the record


```

