import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy import signal

def load_cleaned_data():
    """
    Load the cleaned data for both emotions
    """
    print("Loading cleaned data...")
    tension_data = pd.read_csv("cleaned_data/tension_cleaned.csv")
    excite_data = pd.read_csv("cleaned_data/excite_cleaned.csv")
    
    # Combine the data
    data = pd.concat([tension_data, excite_data], axis=0)
    return data

def extract_features(data, window_size=100):
    """
    Extract statistical and physiological features from the signals using sliding windows
    """
    features = []
    labels = []
    subject_ids = []
    
    # Signals we're interested in
    signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ']
    
    # Process each subject separately
    for subject_id in data['subject_id'].unique():
        subject_data = data[data['subject_id'] == subject_id]
        
        # Slide through the data with overlap
        for start in range(0, len(subject_data) - window_size, window_size // 2):
            window = subject_data.iloc[start:start + window_size]
            
            window_features = []
            
            # Extract features for each signal
            for signal_name in signals:
                signal_data = window[signal_name].values
                
                # Basic statistical features
                window_features.extend([
                    np.mean(signal_data),      # Mean
                    np.std(signal_data),       # Standard deviation
                    skew(signal_data),         # Skewness
                    kurtosis(signal_data),     # Kurtosis
                    np.max(signal_data),       # Maximum
                    np.min(signal_data),       # Minimum
                    np.percentile(signal_data, 75) - np.percentile(signal_data, 25)  # IQR
                ])
                
                # Additional physiological features
                if signal_name == 'EDA':
                    # EDA-specific features
                    window_features.extend([
                        np.mean(np.diff(signal_data)),  # Mean of first differences
                        np.std(np.diff(signal_data)),   # Standard deviation of first differences
                        np.max(np.abs(np.diff(signal_data)))  # Maximum absolute change
                    ])
                elif signal_name.startswith('EMG'):
                    # EMG-specific features
                    window_features.extend([
                        np.mean(np.abs(signal_data)),  # Mean absolute value
                        np.sum(np.abs(np.diff(signal_data))),  # Total variation
                        np.max(np.abs(signal_data))  # Maximum absolute value
                    ])
                elif signal_name.startswith('ACC'):
                    # Accelerometer-specific features
                    window_features.extend([
                        np.sum(np.abs(np.diff(signal_data))),  # Movement intensity
                        np.mean(np.abs(signal_data)),  # Mean absolute acceleration
                        np.max(np.abs(signal_data))  # Maximum absolute acceleration
                    ])
            
            # Cross-signal features
            # EDA-EMG correlation
            window_features.append(np.corrcoef(window['EDA'], window['EMG1'])[0,1])
            window_features.append(np.corrcoef(window['EDA'], window['EMG2'])[0,1])
            
            # Movement-EMG correlation
            for acc in ['ACCX', 'ACCY', 'ACCZ']:
                window_features.append(np.corrcoef(window[acc], window['EMG1'])[0,1])
                window_features.append(np.corrcoef(window[acc], window['EMG2'])[0,1])
            
            features.append(window_features)
            labels.append(window['emotion'].iloc[0])
            subject_ids.append(subject_id)
    
    return pd.DataFrame(features), pd.Series(labels), pd.Series(subject_ids)

def train_emotion_classifier():
    # Load cleaned data
    data = load_cleaned_data()
    
    # Extract features
    print("\nExtracting features...")
    X, y, subject_ids = extract_features(data)
    
    # Split data ensuring subjects don't overlap between train and test
    unique_subjects = subject_ids.unique()
    train_subjects = np.random.choice(unique_subjects, size=int(0.8 * len(unique_subjects)), replace=False)
    
    # Create train/test masks
    train_mask = subject_ids.isin(train_subjects)
    test_mask = ~train_mask
    
    # Split features and labels
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Fear vs Excitement')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    feature_names = []
    signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ']
    stats = ['mean', 'std', 'skew', 'kurtosis', 'max', 'min', 'iqr']
    
    # Basic statistical features
    for signal in signals:
        for stat in stats:
            feature_names.append(f"{signal}_{stat}")
    
    # Physiological features
    for signal in signals:
        if signal == 'EDA':
            feature_names.extend([f"{signal}_mean_diff", f"{signal}_std_diff", f"{signal}_max_diff"])
        elif signal.startswith('EMG'):
            feature_names.extend([f"{signal}_mean_abs", f"{signal}_total_var", f"{signal}_max_abs"])
        elif signal.startswith('ACC'):
            feature_names.extend([f"{signal}_movement", f"{signal}_mean_abs", f"{signal}_max_abs"])
    
    # Cross-signal features
    feature_names.extend([
        "EDA_EMG1_corr", "EDA_EMG2_corr",
        "ACCX_EMG1_corr", "ACCX_EMG2_corr",
        "ACCY_EMG1_corr", "ACCY_EMG2_corr",
        "ACCZ_EMG1_corr", "ACCZ_EMG2_corr"
    ])
    
    # Plot feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return clf, scaler

if __name__ == "__main__":
    model, scaler = train_emotion_classifier() 