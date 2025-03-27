import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, input_dir, output_dir, window_size=100):
        """
        Initialize the feature extractor
        
        Args:
            input_dir (str): Directory containing processed data
            output_dir (str): Directory to save extracted features
            window_size (int): Size of sliding window for feature extraction
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define signals of interest
        self.signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ']
        
        # Define arousal-specific signals
        self.arousal_signals = ['EDA', 'EMG1', 'EMG2', 'ACCX', 'ACCY', 'ACCZ', 'HR', 'RESP']
    
    def extract_statistical_features(self, signal_data):
        """
        Extract statistical features from a signal
        
        Args:
            signal_data (np.array): Signal data
            
        Returns:
            list: List of statistical features
        """
        return [
            np.mean(signal_data),      # Mean
            np.std(signal_data),       # Standard deviation
            skew(signal_data),         # Skewness
            kurtosis(signal_data),     # Kurtosis
            np.max(signal_data),       # Maximum
            np.min(signal_data),       # Minimum
            np.percentile(signal_data, 75) - np.percentile(signal_data, 25)  # IQR
        ]
    
    def extract_eda_features(self, signal_data):
        """
        Extract EDA-specific features
        
        Args:
            signal_data (np.array): EDA signal data
            
        Returns:
            list: List of EDA-specific features
        """
        # Calculate first differences
        diff = np.diff(signal_data)
        
        return [
            np.mean(diff),            # Mean of first differences
            np.std(diff),             # Standard deviation of first differences
            np.max(np.abs(diff)),     # Maximum absolute change
            np.sum(np.abs(diff)),     # Total variation
            np.mean(np.abs(signal_data)),  # Mean absolute value
            np.max(signal_data) - np.min(signal_data)  # Range
        ]
    
    def extract_emg_features(self, signal_data):
        """
        Extract EMG-specific features
        
        Args:
            signal_data (np.array): EMG signal data
            
        Returns:
            list: List of EMG-specific features
        """
        return [
            np.mean(np.abs(signal_data)),  # Mean absolute value
            np.sum(np.abs(np.diff(signal_data))),  # Total variation
            np.max(np.abs(signal_data)),  # Maximum absolute value
            np.std(np.abs(signal_data)),  # Standard deviation of absolute values
            np.percentile(np.abs(signal_data), 90)  # 90th percentile of absolute values
        ]
    
    def extract_acc_features(self, signal_data):
        """
        Extract accelerometer-specific features
        
        Args:
            signal_data (np.array): Accelerometer signal data
            
        Returns:
            list: List of accelerometer-specific features
        """
        return [
            np.sum(np.abs(np.diff(signal_data))),  # Movement intensity
            np.mean(np.abs(signal_data)),  # Mean absolute acceleration
            np.max(np.abs(signal_data)),  # Maximum absolute acceleration
            np.std(np.abs(signal_data)),  # Standard deviation of absolute values
            np.percentile(np.abs(signal_data), 90)  # 90th percentile of absolute values
        ]
    
    def extract_frequency_features(self, signal_data, sampling_rate=100):
        """
        Extract frequency domain features
        
        Args:
            signal_data (np.array): Signal data
            sampling_rate (int): Sampling rate in Hz
            
        Returns:
            list: List of frequency domain features
        """
        # Compute FFT
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft)
        
        # Extract features
        return [
            np.mean(magnitude),  # Mean magnitude
            np.std(magnitude),   # Standard deviation of magnitude
            np.max(magnitude),   # Maximum magnitude
            np.sum(magnitude),   # Total power
            np.mean(magnitude[:len(magnitude)//2])  # Mean of positive frequencies
        ]
    
    def extract_cross_features(self, window_data):
        """
        Extract cross-signal features
        
        Args:
            window_data (pd.DataFrame): Window of data containing all signals
            
        Returns:
            list: List of cross-signal features
        """
        features = []
        
        # EDA-EMG correlations
        for emg in ['EMG1', 'EMG2']:
            features.append(np.corrcoef(window_data['EDA'], window_data[emg])[0,1])
        
        # Movement-EMG correlations
        for acc in ['ACCX', 'ACCY', 'ACCZ']:
            for emg in ['EMG1', 'EMG2']:
                features.append(np.corrcoef(window_data[acc], window_data[emg])[0,1])
        
        # Movement-EDA correlations
        for acc in ['ACCX', 'ACCY', 'ACCZ']:
            features.append(np.corrcoef(window_data[acc], window_data['EDA'])[0,1])
        
        return features
    
    def extract_arousal_features(self, window_data):
        """
        Extract arousal-specific features
        
        Args:
            window_data (pd.DataFrame): Window of data containing all signals
            
        Returns:
            list: List of arousal-specific features
        """
        features = []
        
        # HR-specific features
        if 'HR' in window_data.columns:
            hr_data = window_data['HR'].values
            features.extend([
                np.mean(hr_data),  # Mean heart rate
                np.std(hr_data),   # Heart rate variability
                np.max(hr_data) - np.min(hr_data),  # Heart rate range
                np.mean(np.abs(np.diff(hr_data))),  # Mean absolute heart rate change
                np.percentile(hr_data, 75) - np.percentile(hr_data, 25)  # Heart rate IQR
            ])
        
        # RESP-specific features
        if 'RESP' in window_data.columns:
            resp_data = window_data['RESP'].values
            features.extend([
                np.mean(resp_data),  # Mean respiration rate
                np.std(resp_data),   # Respiration rate variability
                np.max(resp_data) - np.min(resp_data),  # Respiration range
                np.mean(np.abs(np.diff(resp_data))),  # Mean absolute respiration change
                np.percentile(resp_data, 75) - np.percentile(resp_data, 25)  # Respiration IQR
            ])
        
        # Cross-signal features with HR and RESP
        if 'HR' in window_data.columns and 'RESP' in window_data.columns:
            features.extend([
                np.corrcoef(window_data['HR'], window_data['RESP'])[0,1],  # HR-RESP correlation
                np.corrcoef(window_data['HR'], window_data['EDA'])[0,1],   # HR-EDA correlation
                np.corrcoef(window_data['RESP'], window_data['EDA'])[0,1]  # RESP-EDA correlation
            ])
        
        # Rate of change features
        for signal in self.arousal_signals:
            if signal in window_data.columns:
                rate_of_change = np.gradient(window_data[signal].values)
                features.extend([
                    np.mean(rate_of_change),  # Mean rate of change
                    np.std(rate_of_change),   # Rate of change variability
                    np.max(np.abs(rate_of_change))  # Maximum rate of change
                ])
        
        return features
    
    def extract_features_from_window(self, window_data):
        """
        Extract all features from a window of data
        
        Args:
            window_data (pd.DataFrame): Window of data
            
        Returns:
            list: List of all extracted features
        """
        features = []
        
        # Extract features for each signal
        for signal in self.signals:
            signal_data = window_data[signal].values
            
            # Basic statistical features
            features.extend(self.extract_statistical_features(signal_data))
            
            # Signal-specific features
            if signal == 'EDA':
                features.extend(self.extract_eda_features(signal_data))
            elif signal.startswith('EMG'):
                features.extend(self.extract_emg_features(signal_data))
            elif signal.startswith('ACC'):
                features.extend(self.extract_acc_features(signal_data))
            
            # Frequency domain features
            features.extend(self.extract_frequency_features(signal_data))
        
        # Cross-signal features
        features.extend(self.extract_cross_features(window_data))
        
        return features
    
    def process_emotion_data(self, emotion_code):
        """
        Process all data for a specific emotion
        
        Args:
            emotion_code (str): Code for the emotion
        """
        # Load processed data
        data_path = self.input_dir / f'{emotion_code}_cleaned.csv'
        if not data_path.exists():
            logger.error(f"Processed data not found for {emotion_code}")
            return
        
        data = pd.read_csv(data_path)
        features = []
        labels = []
        subject_ids = []
        
        logger.info(f"Extracting features for {emotion_code}...")
        
        # Process each subject separately
        for subject_id in tqdm(data['subject_id'].unique()):
            subject_data = data[data['subject_id'] == subject_id]
            
            # Slide through the data with overlap
            for start in range(0, len(subject_data) - self.window_size, self.window_size // 2):
                window = subject_data.iloc[start:start + self.window_size]
                
                # Extract features
                window_features = self.extract_features_from_window(window)
                features.append(window_features)
                labels.append(window['emotion'].iloc[0])
                subject_ids.append(subject_id)
        
        # Convert to DataFrame
        feature_names = []
        
        # Statistical features
        for signal in self.signals:
            feature_names.extend([
                f"{signal}_mean", f"{signal}_std", f"{signal}_skew",
                f"{signal}_kurtosis", f"{signal}_max", f"{signal}_min",
                f"{signal}_iqr"
            ])
        
        # Signal-specific features
        for signal in self.signals:
            if signal == 'EDA':
                feature_names.extend([
                    f"{signal}_mean_diff", f"{signal}_std_diff",
                    f"{signal}_max_diff", f"{signal}_total_var",
                    f"{signal}_mean_abs", f"{signal}_range"
                ])
            elif signal.startswith('EMG'):
                feature_names.extend([
                    f"{signal}_mean_abs", f"{signal}_total_var",
                    f"{signal}_max_abs", f"{signal}_std_abs",
                    f"{signal}_p90_abs"
                ])
            elif signal.startswith('ACC'):
                feature_names.extend([
                    f"{signal}_movement", f"{signal}_mean_abs",
                    f"{signal}_max_abs", f"{signal}_std_abs",
                    f"{signal}_p90_abs"
                ])
        
        # Frequency features
        for signal in self.signals:
            feature_names.extend([
                f"{signal}_freq_mean", f"{signal}_freq_std",
                f"{signal}_freq_max", f"{signal}_freq_power",
                f"{signal}_freq_mean_pos"
            ])
        
        # Cross-signal features
        feature_names.extend([
            "EDA_EMG1_corr", "EDA_EMG2_corr",
            "ACCX_EMG1_corr", "ACCX_EMG2_corr",
            "ACCY_EMG1_corr", "ACCY_EMG2_corr",
            "ACCZ_EMG1_corr", "ACCZ_EMG2_corr",
            "ACCX_EDA_corr", "ACCY_EDA_corr", "ACCZ_EDA_corr"
        ])
        
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['emotion'] = labels
        features_df['subject_id'] = subject_ids
        
        # Save features
        features_df.to_csv(
            self.output_dir / f'{emotion_code}_features.csv',
            index=False
        )
        
        logger.info(f"Completed feature extraction for {emotion_code}")

    def process_arousal_data(self):
        """
        Process arousal data and extract features
        """
        # Load processed arousal data
        data_path = self.input_dir / 'arousal_data_cleaned.csv'
        if not data_path.exists():
            logger.error("Processed arousal data not found")
            return
        
        data = pd.read_csv(data_path)
        features = []
        subject_ids = []
        
        logger.info("Extracting arousal features...")
        
        # Process each subject separately
        for subject_id in tqdm(data['subject_id'].unique()):
            subject_data = data[data['subject_id'] == subject_id]
            
            # Slide through the data with overlap
            for start in range(0, len(subject_data) - self.window_size, self.window_size // 2):
                window = subject_data.iloc[start:start + self.window_size]
                
                # Extract features
                window_features = self.extract_features_from_window(window)
                window_features.extend(self.extract_arousal_features(window))
                features.append(window_features)
                subject_ids.append(subject_id)
        
        # Convert to DataFrame
        feature_names = []
        
        # Standard features
        for signal in self.signals:
            feature_names.extend([
                f"{signal}_mean", f"{signal}_std", f"{signal}_skew",
                f"{signal}_kurtosis", f"{signal}_max", f"{signal}_min",
                f"{signal}_iqr"
            ])
        
        # Signal-specific features
        for signal in self.signals:
            if signal == 'EDA':
                feature_names.extend([
                    f"{signal}_mean_diff", f"{signal}_std_diff",
                    f"{signal}_max_diff", f"{signal}_total_var",
                    f"{signal}_mean_abs", f"{signal}_range"
                ])
            elif signal.startswith('EMG'):
                feature_names.extend([
                    f"{signal}_mean_abs", f"{signal}_total_var",
                    f"{signal}_max_abs", f"{signal}_std_abs",
                    f"{signal}_p90_abs"
                ])
            elif signal.startswith('ACC'):
                feature_names.extend([
                    f"{signal}_movement", f"{signal}_mean_abs",
                    f"{signal}_max_abs", f"{signal}_std_abs",
                    f"{signal}_p90_abs"
                ])
        
        # Frequency features
        for signal in self.signals:
            feature_names.extend([
                f"{signal}_freq_mean", f"{signal}_freq_std",
                f"{signal}_freq_max", f"{signal}_freq_power",
                f"{signal}_freq_mean_pos"
            ])
        
        # Cross-signal features
        feature_names.extend([
            "EDA_EMG1_corr", "EDA_EMG2_corr",
            "ACCX_EMG1_corr", "ACCX_EMG2_corr",
            "ACCY_EMG1_corr", "ACCY_EMG2_corr",
            "ACCZ_EMG1_corr", "ACCZ_EMG2_corr",
            "ACCX_EDA_corr", "ACCY_EDA_corr", "ACCZ_EDA_corr"
        ])
        
        # Arousal-specific features
        feature_names.extend([
            "HR_mean", "HR_std", "HR_range", "HR_mean_change", "HR_iqr",
            "RESP_mean", "RESP_std", "RESP_range", "RESP_mean_change", "RESP_iqr",
            "HR_RESP_corr", "HR_EDA_corr", "RESP_EDA_corr"
        ])
        
        # Rate of change features
        for signal in self.arousal_signals:
            feature_names.extend([
                f"{signal}_mean_rate", f"{signal}_std_rate", f"{signal}_max_rate"
            ])
        
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['subject_id'] = subject_ids
        
        # Save features
        features_df.to_csv(
            self.output_dir / 'arousal_features.csv',
            index=False
        )
        
        logger.info("Completed arousal feature extraction")

def main():
    # Initialize feature extractor
    extractor = FeatureExtractor(
        input_dir="data/processed",
        output_dir="data/features"
    )
    
    # Process both emotions
    extractor.process_emotion_data("04_TENSION")
    extractor.process_emotion_data("05_EXCITE")
    
    # Process arousal data
    extractor.process_arousal_data()
    
    logger.info("Feature extraction completed!")

if __name__ == "__main__":
    main() 