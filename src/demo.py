"""
demo.py - Interactive demo for currency movement prediction
"""

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / 'models'


def load_model():
    """Load the trained model and scaler from the project root."""
    model_path = MODELS_DIR / 'best_svm_model.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    feature_path = MODELS_DIR / 'feature_names.txt'

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {feature_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(feature_path, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    return model, scaler, features

def predict_currency_movement(model, scaler, features):
    """
    Interactive prediction function
    """
    print("\n" + "="*50)
    print("🔮 CURRENCY MOVEMENT PREDICTOR")
    print("="*50)
    
    print("\n📝 Enter the following values:")
    print("-" * 40)
    
    # Get user input
    lag_1 = float(input("   Yesterday's price (lag_1): "))
    lag_2 = float(input("   2 days ago price (lag_2): "))
    lag_3 = float(input("   3 days ago price (lag_3): "))
    return_1d = float(input("   Daily return (return_1d): "))
    ma_5 = float(input("   5-day moving average (ma_5): "))
    ma_10 = float(input("   10-day moving average (ma_10): "))
    
    # Create input array
    input_data = np.array([[lag_1, lag_2, lag_3, return_1d, ma_5, ma_10]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.decision_function(input_scaled)[0]
    
    # Show results
    print("\n" + "="*50)
    print("📊 PREDICTION RESULT:")
    print("-" * 40)
    
    if prediction == 1:
        print("   📈 PREDICTION: Price will GO UP")
    else:
        print("   📉 PREDICTION: Price will GO DOWN or stay SAME")
    
    print(f"   Confidence Score: {abs(confidence):.3f}")
    print("="*50)
    
    return prediction, confidence

def batch_predict(model, scaler, features):
    """
    Batch prediction from CSV file
    """
    print("\n📊 BATCH PREDICTION")
    print("-" * 40)
    
    # Load sample data
    sample_data = pd.DataFrame({
        'lag_1': [112.65, 115.20, 118.50, 110.30, 120.10],
        'lag_2': [111.76, 114.80, 117.90, 109.80, 119.50],
        'lag_3': [112.74, 115.10, 118.10, 110.20, 120.00],
        'return_1d': [-0.0008, 0.0035, 0.0051, -0.0045, 0.0083],
        'ma_5': [112.37, 114.90, 118.20, 110.50, 119.80],
        'ma_10': [111.65, 114.20, 117.80, 109.90, 119.20]
    })
    
    # Scale and predict
    input_scaled = scaler.transform(sample_data.values)
    predictions = model.predict(input_scaled)
    confidence = model.decision_function(input_scaled)
    
    # Show results
    print("\n📊 Batch Prediction Results:")
    print("-" * 60)
    
    for i, (pred, conf) in enumerate(zip(predictions, confidence), 1):
        direction = "📈 UP" if pred == 1 else "📉 DOWN"
        print(f"Sample {i}: {direction} | Confidence: {abs(conf):.3f}")

def main():
    """Main interactive menu"""
    print("🚀 Loading model...")
    model, scaler, features = load_model()
    print("✅ Model loaded successfully!")
    
    while True:
        print("\n" + "="*50)
        print("📊 CURRENCY PREDICTOR MENU")
        print("="*50)
        print("1. Predict Single Currency Movement")
        print("2. Batch Predict (Sample Data)")
        print("3. Show Model Information")
        print("4. Exit")
        print("="*50)
        
        choice = input("\nChoose an option (1-4): ")
        
        if choice == '1':
            predict_currency_movement(model, scaler, features)
        elif choice == '2':
            batch_predict(model, scaler, features)
        elif choice == '3':
            print("\n📋 MODEL INFORMATION:")
            print(f"   Model Type: {type(model).__name__}")
            print(f"   Kernel: {model.kernel}")
            print(f"   C Parameter: {model.C}")
            print(f"   Gamma: {model.gamma}")
            print(f"   Features: {', '.join(features)}")
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()