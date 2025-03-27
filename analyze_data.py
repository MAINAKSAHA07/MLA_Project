import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_dataset():
    # Base directory
    base_dir = "DECEiVeR_DATASET/01_DECEiVeR_raw"
    
    # Emotion directories
    emotions = ['01_NEUTRAL', '02_CALM', '03_TIRED', '04_TENSION', '05_EXCITE']
    
    # Analyze first file from each emotion
    for emotion in emotions:
        emotion_dir = os.path.join(base_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Directory not found: {emotion_dir}")
            continue
            
        # Get first CSV file in the directory
        csv_files = [f for f in os.listdir(emotion_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {emotion_dir}")
            continue
            
        first_file = csv_files[0]
        file_path = os.path.join(emotion_dir, first_file)
        
        print(f"\nAnalyzing {emotion}:")
        print(f"File: {first_file}")
        
        # Read first few rows to understand structure
        try:
            df = pd.read_csv(file_path, nrows=5)
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nData types:")
            print(df.dtypes)
            print("\nSample data:")
            print(df.head())
            print("\nShape:", df.shape)
            print("-" * 80)
        except Exception as e:
            print(f"Error reading file: {e}")

if __name__ == "__main__":
    analyze_dataset() 