import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_dir, output_dir, window_size=100):
        """
        Initialize the data preprocessor
        
        Args:
            input_dir (str): Directory containing raw data
            output_dir (str): Directory to save processed data
            window_size (int): Size of sliding window for processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Define signals of interest
        self.signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ']
        
        # Define arousal-specific signals
        self.arousal_signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ', 'HR', 'RESP']
        
    def load_data(self, emotion_code):
        """
        Load all files for a specific emotion
        
        Args:
            emotion_code (str): Code for the emotion (e.g., '04_TENSION')
            
        Returns:
            pd.DataFrame: Combined data for the emotion
        """
        emotion_dir = self.input_dir / emotion_code
        all_data = []
        
        logger.info(f"Loading {emotion_code} data...")
        
        for file in tqdm(list(emotion_dir.glob('*.csv'))):
            try:
                df = pd.read_csv(file)
                
                # Extract metadata from filename
                subject_id = int(file.stem.split('_')[0][2:])
                session = int(file.stem.split('_')[1])
                
                # Add metadata columns
                df['subject_id'] = subject_id
                df['session'] = session
                df['emotion'] = emotion_code
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else None
    
    def remove_outliers(self, data, signal, n_std=3):
        """
        Remove outliers using the z-score method
        
        Args:
            data (pd.DataFrame): Input data
            signal (str): Signal column name
            n_std (float): Number of standard deviations for outlier threshold
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        z_scores = np.abs((data[signal] - data[signal].mean()) / data[signal].std())
        return data[z_scores < n_std]
    
    def apply_filter(self, data, signal, cutoff_freq=0.1, order=4):
        """
        Apply Butterworth low-pass filter
        
        Args:
            data (pd.DataFrame): Input data
            signal (str): Signal column name
            cutoff_freq (float): Cutoff frequency
            order (int): Filter order
            
        Returns:
            pd.Series: Filtered signal
        """
        b, a = signal.butter(order, cutoff_freq, 'low')
        return signal.filtfilt(b, a, data[signal])
    
    def process_signals(self, data):
        """
        Process all signals in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Processed data
        """
        processed_data = data.copy()
        
        for signal in self.signals:
            # Remove outliers
            processed_data = self.remove_outliers(processed_data, signal)
            
            # Apply low-pass filter
            processed_data[signal] = self.apply_filter(processed_data, signal)
        
        return processed_data
    
    def create_visualizations(self, data, emotion):
        """
        Create and save visualizations for the processed data
        
        Args:
            data (pd.DataFrame): Processed data
            emotion (str): Emotion code
        """
        # Distribution plots
        plt.figure(figsize=(15, 10))
        for i, signal in enumerate(self.signals, 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=data, x=signal, bins=50)
            plt.title(f'{signal} Distribution - {emotion}')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / f'{emotion}_distributions.png')
        plt.close()
        
        # Time series plot for a sample subject
        sample_subject = data['subject_id'].iloc[0]
        sample_data = data[data['subject_id'] == sample_subject].head(1000)
        
        plt.figure(figsize=(15, 10))
        for i, signal in enumerate(self.signals, 1):
            plt.subplot(2, 3, i)
            plt.plot(sample_data['TIMESTAMP'], sample_data[signal])
            plt.title(f'{signal} Time Series - Subject {sample_subject}')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / f'{emotion}_timeseries_sample.png')
        plt.close()
    
    def generate_summary(self, data, emotion):
        """
        Generate and save summary statistics
        
        Args:
            data (pd.DataFrame): Processed data
            emotion (str): Emotion code
        """
        # Summary statistics
        summary = data.describe()
        summary.to_csv(self.output_dir / 'processed' / f'{emotion}_summary_stats.csv')
        
        # Sample counts per subject
        subject_counts = data.groupby('subject_id').size()
        subject_counts.to_csv(self.output_dir / 'processed' / f'{emotion}_samples_per_subject.csv')
    
    def process_emotion_data(self, emotion_code):
        """
        Process all data for a specific emotion
        
        Args:
            emotion_code (str): Code for the emotion
        """
        # Load data
        data = self.load_data(emotion_code)
        if data is None:
            logger.error(f"No data found for {emotion_code}")
            return
        
        # Process signals
        processed_data = self.process_signals(data)
        
        # Create visualizations
        self.create_visualizations(processed_data, emotion_code)
        
        # Generate summary
        self.generate_summary(processed_data, emotion_code)
        
        # Save processed data
        processed_data.to_csv(
            self.output_dir / 'processed' / f'{emotion_code}_cleaned.csv',
            index=False
        )
        
        logger.info(f"Completed processing {emotion_code} data")
    
    def load_arousal_data(self):
        """
        Load arousal data from the DECEiVeR dataset
        
        Returns:
            pd.DataFrame: Combined arousal data
        """
        arousal_dir = self.input_dir / '04_DECEiVeR_arouval'
        all_data = []
        
        logger.info("Loading arousal data...")
        
        for file in tqdm(list(arousal_dir.glob('*.csv'))):
            try:
                df = pd.read_csv(file)
                
                # Extract metadata from filename
                subject_id = int(file.stem.split('_')[0][2:])
                session = int(file.stem.split('_')[1])
                
                # Add metadata columns
                df['subject_id'] = subject_id
                df['session'] = session
                df['data_type'] = 'arousal'
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing arousal file {file}: {str(e)}")
                continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else None
    
    def process_arousal_data(self):
        """
        Process arousal data with specific methods for arousal-related signals
        """
        # Load arousal data
        data = self.load_arousal_data()
        if data is None:
            logger.error("No arousal data found")
            return
        
        # Process signals
        processed_data = self.process_signals(data)
        
        # Additional arousal-specific processing
        processed_data = self.process_arousal_signals(processed_data)
        
        # Create visualizations
        self.create_arousal_visualizations(processed_data)
        
        # Generate summary
        self.generate_arousal_summary(processed_data)
        
        # Save processed data
        processed_data.to_csv(
            self.output_dir / 'processed' / 'arousal_data_cleaned.csv',
            index=False
        )
        
        logger.info("Completed processing arousal data")
    
    def process_arousal_signals(self, data):
        """
        Apply arousal-specific signal processing
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Processed data with arousal-specific features
        """
        processed_data = data.copy()
        
        # Process HR (Heart Rate) if available
        if 'HR' in data.columns:
            processed_data['HR'] = self.apply_filter(data['HR'], cutoff_freq=0.1)
            processed_data['HR_derivative'] = np.gradient(processed_data['HR'])
        
        # Process RESP (Respiration) if available
        if 'RESP' in data.columns:
            processed_data['RESP'] = self.apply_filter(data['RESP'], cutoff_freq=0.1)
            processed_data['RESP_derivative'] = np.gradient(processed_data['RESP'])
        
        # Calculate arousal-specific features
        for signal in self.arousal_signals:
            if signal in data.columns:
                # Calculate rate of change
                processed_data[f'{signal}_rate_of_change'] = np.gradient(data[signal])
                
                # Calculate peak-to-peak amplitude
                peaks, _ = signal.find_peaks(np.abs(data[signal]))
                processed_data[f'{signal}_peak_amplitude'] = np.mean(np.abs(data[signal].iloc[peaks]))
        
        return processed_data
    
    def create_arousal_visualizations(self, data):
        """
        Create visualizations specific to arousal data
        
        Args:
            data (pd.DataFrame): Processed arousal data
        """
        # Distribution plots for arousal signals
        plt.figure(figsize=(15, 10))
        for i, signal in enumerate(self.arousal_signals, 1):
            if signal in data.columns:
                plt.subplot(2, 4, i)
                sns.histplot(data=data, x=signal, bins=50)
                plt.title(f'{signal} Distribution - Arousal')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'arousal_distributions.png')
        plt.close()
        
        # Time series plot for a sample subject
        sample_subject = data['subject_id'].iloc[0]
        sample_data = data[data['subject_id'] == sample_subject].head(1000)
        
        plt.figure(figsize=(15, 10))
        for i, signal in enumerate(self.arousal_signals, 1):
            if signal in data.columns:
                plt.subplot(2, 4, i)
                plt.plot(sample_data['TIMESTAMP'], sample_data[signal])
                plt.title(f'{signal} Time Series - Subject {sample_subject}')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'arousal_timeseries_sample.png')
        plt.close()
    
    def generate_arousal_summary(self, data):
        """
        Generate and save summary statistics for arousal data
        
        Args:
            data (pd.DataFrame): Processed arousal data
        """
        # Summary statistics
        summary = data.describe()
        summary.to_csv(self.output_dir / 'processed' / 'arousal_summary_stats.csv')
        
        # Sample counts per subject
        subject_counts = data.groupby('subject_id').size()
        subject_counts.to_csv(self.output_dir / 'processed' / 'arousal_samples_per_subject.csv')

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        input_dir="DECEiVeR_DATASET/01_DECEiVeR_raw",
        output_dir="data/processed"
    )
    
    # Process both emotions
    preprocessor.process_emotion_data("04_TENSION")
    preprocessor.process_emotion_data("05_EXCITE")
    
    # Process arousal data
    preprocessor.process_arousal_data()
    
    logger.info("Data preprocessing completed!")

if __name__ == "__main__":
    main() 