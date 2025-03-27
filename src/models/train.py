import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the emotion classifier
        
        Args:
            input_dir (str): Directory containing feature data
            output_dir (str): Directory to save trained models and results
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load and combine feature data for both emotions and arousal
        
        Returns:
            tuple: (X, y, subject_ids) - Features, labels, and subject IDs
        """
        logger.info("Loading feature data...")
        
        # Load tension data
        tension_data = pd.read_csv(self.input_dir / '04_TENSION_features.csv')
        
        # Load excitement data
        excite_data = pd.read_csv(self.input_dir / '05_EXCITE_features.csv')
        
        # Load arousal data
        arousal_data = pd.read_csv(self.input_dir / 'arousal_features.csv')
        
        # Combine data
        data = pd.concat([tension_data, excite_data], axis=0)
        
        # Merge with arousal features
        data = data.merge(
            arousal_data,
            on='subject_id',
            how='left',
            suffixes=('', '_arousal')
        )
        
        # Separate features, labels, and subject IDs
        X = data.drop(['emotion', 'subject_id'], axis=1)
        y = data['emotion']
        subject_ids = data['subject_id']
        
        return X, y, subject_ids
    
    def create_lstm_model(self, input_shape):
        """
        Create an LSTM model for sequence classification
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.4),
            LSTM(64),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self):
        """
        Train multiple models and evaluate their performance
        """
        # Load data
        X, y, subject_ids = self.load_data()
        
        # Convert labels to binary
        y = (y == '05_EXCITE').astype(int)
        
        # Split data ensuring subject independence
        unique_subjects = subject_ids.unique()
        train_subjects = np.random.choice(
            unique_subjects,
            size=int(0.8 * len(unique_subjects)),
            replace=False
        )
        
        # Create train/test masks
        train_mask = subject_ids.isin(train_subjects)
        test_mask = ~train_mask
        
        # Split features and labels
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.output_dir / 'models' / 'scaler.joblib')
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            results[name] = {
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': roc_curve(y_test, y_pred_proba)
            }
            
            # Save model
            joblib.dump(model, self.output_dir / 'models' / f'{name.lower().replace(" ", "_")}.joblib')
        
        # Train LSTM model
        logger.info("\nTraining LSTM model...")
        
        # Reshape data for LSTM
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        # Create and train LSTM model
        lstm_model = self.create_lstm_model((1, X_train_scaled.shape[1]))
        lstm_model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate LSTM model
        y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
        y_pred_proba_lstm = lstm_model.predict(X_test_lstm).flatten()
        
        results['LSTM'] = {
            'classification_report': classification_report(y_test, y_pred_lstm),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lstm),
            'roc_auc': roc_curve(y_test, y_pred_proba_lstm)
        }
        
        # Save LSTM model
        lstm_model.save(self.output_dir / 'models' / 'lstm_model')
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """
        Save model results and create visualizations
        
        Args:
            results (dict): Dictionary containing results for each model
        """
        # Save classification reports
        for name, result in results.items():
            with open(self.output_dir / 'results' / f'{name.lower().replace(" ", "_")}_report.txt', 'w') as f:
                f.write(result['classification_report'])
        
        # Plot confusion matrices
        plt.figure(figsize=(15, 10))
        for i, (name, result) in enumerate(results.items(), 1):
            plt.subplot(2, 2, i)
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'results' / 'confusion_matrices.png')
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = result['roc_auc']
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend()
        plt.savefig(self.output_dir / 'results' / 'roc_curves.png')
        plt.close()

def main():
    # Initialize classifier
    classifier = EmotionClassifier(
        input_dir="data/features",
        output_dir="models"
    )
    
    # Train models and get results
    results = classifier.train_models()
    
    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(result['classification_report'])
    
    logger.info("Model training and evaluation completed!")

if __name__ == "__main__":
    main() 