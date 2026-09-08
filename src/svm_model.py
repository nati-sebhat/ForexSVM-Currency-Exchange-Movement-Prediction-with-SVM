"""
svm_model.py - Production-ready SVM model for currency movement prediction
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class CurrencySVM:
    """
    SVM model for currency exchange movement prediction
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the model with a path that resolves relative to the repo root,
        so the project works from any working directory.
        """
        self.repo_root = Path(__file__).resolve().parents[1]

        if data_path is None:
            data_path = self.repo_root / 'data' / 'processed' / 'processed_data.csv'
        else:
            path = Path(data_path)
            if not path.is_absolute():
                path = self.repo_root / path
            data_path = path.resolve()

        self.data_path = str(data_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.features = ['lag_1', 'lag_2', 'lag_3', 'return_1d', 'ma_5', 'ma_10']

    def _resolve_repo_path(self, path_value):
        """Resolve a project-relative path to a repo-root-based absolute path."""
        path = Path(path_value)
        if not path.is_absolute():
            path = self.repo_root / path
        return str(path.resolve())
        
    def load_data(self):
        """
        Load the processed data from Team 1
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Data loaded: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: Data file not found at {self.data_path}")
            return None
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare features and target, split into train/test
        """
        if self.data is None:
            self.load_data()
            
        X = self.data[self.features]
        y = self.data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"✅ Data prepared:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self):
        """
        Scale features using StandardScaler
        """
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("✅ Features scaled")
        return self.X_train_scaled, self.X_test_scaled
    
    def train(self, kernel='rbf', C=10, gamma='scale', random_state=42):
        """
        Train SVM model
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✅ Model trained")
        return self.model
    
    def hyperparameter_tuning(self, param_grid=None, cv=5):
        """
        Perform grid search for hyperparameter tuning
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        grid_search = GridSearchCV(
            SVC(random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.model = grid_search.best_estimator_
        
        print(f"✅ Hyperparameter tuning complete!")
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def evaluate(self):
        """
        Evaluate model performance
        """
        y_pred = self.model.predict(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print("\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_confusion_matrix(self, save_path='results/plots/confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        save_path = self._resolve_repo_path(save_path)
        y_pred = self.model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down/Same', 'Up'],
                   yticklabels=['Down/Same', 'Up'],
                   ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(self, save_path='results/plots/roc_curve.png'):
        """
        Plot ROC curve
        """
        save_path = self._resolve_repo_path(save_path)
        y_proba = self.model.decision_function(self.X_test_scaled)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curve saved: {save_path}")
    
    def save_model(self, model_path='models/best_svm_model.pkl'):
        """
        Save the trained model and scaler
        """
        model_path = self._resolve_repo_path(model_path)
        scaler_path = self._resolve_repo_path('models/scaler.pkl')
        feature_path = self._resolve_repo_path('models/feature_names.txt')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        with open(feature_path, 'w') as f:
            for feature in self.features:
                f.write(f"{feature}\n")
        
        print(f"✅ Model saved: {model_path}")
    
    def predict(self, new_data):
        """
        Predict on new data
        """
        if self.model is None:
            print("❌ Model not trained. Please train first.")
            return None
        
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        confidence = self.model.decision_function(new_data_scaled)
        
        return predictions, confidence
    
    def run_pipeline(self):
        """
        Run the complete pipeline
        """
        print("🚀 Running SVM Pipeline...\n")
        
        # Load data
        self.load_data()
        print()
        
        # Prepare data
        self.prepare_data()
        print()
        
        # Scale features
        self.scale_features()
        print()
        
        # Train model
        self.train(kernel='rbf', C=10, gamma=0.01)
        print()
        
        # Evaluate
        metrics = self.evaluate()
        print()
        
        # Generate plots
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        print()
        
        # Save model
        self.save_model()
        print()
        
        return metrics

# Main execution
if __name__ == "__main__":
    # Initialize and run
    model = CurrencySVM()
    metrics = model.run_pipeline()
    
    print("\n🎉 Pipeline complete!")