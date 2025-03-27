# Emotion Classification Model (Fear vs. Excitement)

This project implements a machine learning model to distinguish between fear and excitement using physiological data from the DECEiVeR dataset.

## Project Structure
```
.
├── data/
│   ├── raw/                 # Original DECEiVeR dataset
│   ├── processed/           # Cleaned and preprocessed data
│   └── features/           # Extracted features
├── src/
│   ├── data_processing/    # Data preprocessing and feature extraction
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                # Unit tests
└── requirements.txt      # Project dependencies
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Data Preprocessing:
```bash
python src/data_processing/preprocess.py
```

2. Feature Extraction:
```bash
python src/data_processing/feature_extraction.py
```

3. Model Training:
```bash
python src/models/train.py
```

4. Real-time Prediction:
```bash
python src/models/predict.py
```

## Project Components

### 1. Data Processing
- Data cleaning and normalization
- Outlier detection and removal
- Signal alignment and synchronization
- Feature extraction from physiological signals

### 2. Model Architecture
- Binary classification (Fear vs. Excitement)
- Support for both batch and real-time processing
- Multiple model implementations (Random Forest, SVM, LSTM)

### 3. Features
- EDA (Electrodermal Activity) analysis
- EMG1/EMG2 (Muscle tension) patterns
- ACC (X, Y, Z) movement analysis
- Statistical and frequency domain features
- Cross-signal correlations

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC curves
- Confusion matrices
- Real-time performance metrics

## Dependencies
- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- TensorFlow (for deep learning models)
- Matplotlib, Seaborn
- SciPy 