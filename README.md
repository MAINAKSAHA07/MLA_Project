# Emotion Classification: Fear vs. Excitement Detection

This project implements a machine learning system to distinguish between fear and excitement using physiological data from the DECEiVeR dataset. The system analyzes multiple physiological signals including Electrodermal Activity (EDA), muscle tension (EMG), and accelerometer data to classify emotional states.

## Project Overview

The system processes physiological signals to extract meaningful features and uses machine learning to classify between fear and excitement states. This has applications in:
- Emotion recognition systems
- Mental health monitoring
- Human-computer interaction
- Affective computing research

## Project Structure
```
.
├── data/
│   ├── raw/                 # Original DECEiVeR dataset
│   ├── cleaned_data/        # Preprocessed and cleaned data
│   └── data_quality_checks/ # Data quality visualizations and summaries
├── src/
│   ├── emotion_classifier.py    # Main classification implementation
│   ├── preprocess_data.py       # Data preprocessing pipeline
│   └── analyze_data.py          # Data analysis utilities
├── models/                  # Trained model files
├── analysis_results_csv/    # Analysis results and metrics
├── preprocessed_data/       # Intermediate processed data
├── tests/                   # Unit tests
└── requirements.txt         # Project dependencies
```

## Features

### Signal Processing
- EDA (Electrodermal Activity) analysis
- EMG1/EMG2 (Muscle tension) monitoring
- 3-axis Accelerometer (ACC) data processing
- Signal filtering and noise reduction
- Outlier detection and removal

### Feature Extraction
- Statistical features (mean, std, skewness, kurtosis)
- Signal-specific features:
  - EDA: Mean difference, standard deviation of differences
  - EMG: Mean absolute value, total variation
  - ACC: Movement intensity, acceleration patterns
- Cross-signal correlations
- Time-domain and frequency-domain features

### Machine Learning
- Random Forest Classifier implementation
- Feature importance analysis
- Cross-validation with subject-wise splitting
- Performance metrics visualization

## Setup and Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python preprocess_data.py
```
This will:
- Clean and normalize the raw data
- Remove outliers
- Apply signal filtering
- Generate data quality visualizations
- Save processed data to `cleaned_data/`

2. Model Training and Evaluation:
```bash
python emotion_classifier.py
```
This will:
- Load preprocessed data
- Extract features
- Train the Random Forest classifier
- Generate performance metrics
- Create visualizations of results

## Model Performance

The system evaluates performance using:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization
- Feature importance analysis
- Cross-validation scores

## Dependencies
- Python 3.8+
- NumPy >= 1.24.3
- Pandas >= 2.0.0
- Scikit-learn >= 1.2.2
- SciPy >= 1.10.1
- Matplotlib >= 3.7.1
- Seaborn >= 0.12.2
- TensorFlow >= 2.12.0 (for potential deep learning extensions)

## Data Quality Checks

The preprocessing pipeline includes:
- Distribution analysis of signals
- Time series visualization
- Summary statistics generation
- Subject-wise sample counts
- Signal quality metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DECEiVeR dataset providers
- Contributors and maintainers
- Research community in affective computing 
