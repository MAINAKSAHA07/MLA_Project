import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from pathlib import Path

def load_and_clean_file(file_path):
    """
    Load and clean a single data file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract metadata from filename
        file_name = os.path.basename(file_path)
        subject_id = int(file_name.split('_')[0][2:])
        session = int(file_name.split('_')[1])
        emotion = file_name.split('_')[3]
        
        # Add metadata columns
        df['subject_id'] = subject_id
        df['session'] = session
        df['emotion'] = emotion
        
        # Select only the columns we need
        columns_of_interest = ['TIMESTAMP', 'EMG1', 'EMG2', 'EDA', 'ACCX', 'ACCY', 'ACCZ', 
                             'subject_id', 'session', 'emotion']
        df = df[columns_of_interest]
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Remove outliers using IQR method for each signal
        signals = ['EMG1', 'EMG2', 'EDA', 'ACCX', 'ACCY', 'ACCZ']
        for col in signals:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)))]
        
        # Apply a low-pass filter to remove high-frequency noise
        for col in signals:
            # Butterworth low-pass filter
            b, a = signal.butter(4, 0.1, 'low')
            df[col] = signal.filtfilt(b, a, df[col])
        
        return df
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_emotion_data(emotion_code):
    """
    Process all files for a specific emotion
    """
    base_dir = "DECEiVeR_DATASET/01_DECEiVeR_raw"
    emotion_dir = os.path.join(base_dir, emotion_code)
    all_data = []
    
    print(f"\nProcessing {emotion_code} data...")
    
    for file in os.listdir(emotion_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(emotion_dir, file)
            print(f"Processing {file}...")
            df = load_and_clean_file(file_path)
            if df is not None:
                all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def create_data_visualizations(data, emotion):
    """
    Create visualizations for the cleaned data
    """
    output_dir = "data_quality_checks"
    os.makedirs(output_dir, exist_ok=True)
    
    signals = ['EMG1', 'EMG2', 'EDA', 'ACCX', 'ACCY', 'ACCZ']
    
    # Distribution plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(signals, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=data, x=col, bins=50)
        plt.title(f'{col} Distribution - {emotion}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{emotion}_distributions.png'))
    plt.close()
    
    # Time series plot for a sample subject
    sample_subject = data['subject_id'].iloc[0]
    sample_data = data[data['subject_id'] == sample_subject].head(1000)
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(signals, 1):
        plt.subplot(2, 3, i)
        plt.plot(sample_data['TIMESTAMP'], sample_data[col])
        plt.title(f'{col} Time Series - Subject {sample_subject}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{emotion}_timeseries_sample.png'))
    plt.close()

def generate_data_summary(data, emotion):
    """
    Generate summary statistics for the cleaned data
    """
    summary = data.describe()
    summary.to_csv(f'data_quality_checks/{emotion}_summary_stats.csv')
    
    # Count number of samples per subject
    subject_counts = data.groupby('subject_id').size()
    subject_counts.to_csv(f'data_quality_checks/{emotion}_samples_per_subject.csv')

def main():
    # Create output directories
    os.makedirs("cleaned_data", exist_ok=True)
    os.makedirs("data_quality_checks", exist_ok=True)
    
    # Process tension (fear) data
    tension_data = process_emotion_data("04_TENSION")
    if tension_data is not None:
        tension_data.to_csv("cleaned_data/tension_cleaned.csv", index=False)
        create_data_visualizations(tension_data, "tension")
        generate_data_summary(tension_data, "tension")
    
    # Process excitement data
    excite_data = process_emotion_data("05_EXCITE")
    if excite_data is not None:
        excite_data.to_csv("cleaned_data/excite_cleaned.csv", index=False)
        create_data_visualizations(excite_data, "excite")
        generate_data_summary(excite_data, "excite")
    
    print("\nData preprocessing completed!")
    print("Cleaned data saved in 'cleaned_data' directory")
    print("Data quality visualizations and summaries saved in 'data_quality_checks' directory")

if __name__ == "__main__":
    main() 