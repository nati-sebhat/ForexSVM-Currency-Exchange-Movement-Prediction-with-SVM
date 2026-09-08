# ForexSVM: Currency Exchange Movement Prediction with SVM

This project is a student machine learning project focused on predicting whether a currency exchange rate will move upward or downward based on historical market indicators.

## Project Overview

The work is divided into two major parts:

1. Data preparation and processing by Team 1
2. SVM model development and evaluation by Team 2

The dataset is prepared from currency and economy-related historical data and transformed into a supervised binary classification problem:

- Class 0: Down or same movement
- Class 1: Up movement

The project uses a Support Vector Machine (SVM) to classify future directional movement using engineered technical features.

## Objective

To build a model that learns from historical exchange-rate features and predicts the next movement direction using machine learning.

## Dataset

The project uses processed financial data stored in:

- `data/processed/processed_data.csv`

The processed dataset includes engineered features such as:

- `lag_1`
- `lag_2`
- `lag_3`
- `return_1d`
- `ma_5`
- `ma_10`
- `target`

## Methodology

The project follows a standard ML pipeline:

1. Load and inspect the processed dataset
2. Check target balance and feature distribution
3. Split the data into train and test sets using stratification
4. Standardize features using `StandardScaler`
5. Train an SVM classifier
6. Tune hyperparameters with `GridSearchCV`
7. Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC
8. Visualize the confusion matrix and ROC curve
9. Save the trained model, scaler, and feature list
10. Create a simple interactive prediction demo

## Model and Evaluation

The implemented model uses an SVM with a radial basis function (RBF) kernel and hyperparameter tuning.

### Current project results

From the saved evaluation metrics:

- Best parameters: `C=100`, `gamma='auto'`, `kernel='rbf'`
- Accuracy: `0.4798`
- Precision: `0.4934`
- Recall: `0.2725`
- F1-score: `0.3511`
- ROC-AUC: `0.4791`

These metrics show that the project is a useful baseline and learning exercise, but not yet a strong production-grade financial prediction model.

## Project Structure

- `README.md` – project overview and usage guide
- `requirements.txt` – required Python libraries
- `data/` – raw and processed datasets
- `notebooks/team2_svm_model.ipynb` – full training and evaluation notebook
- `src/svm_model.py` – reusable SVM pipeline class
- `src/demo.py` – command-line interactive prediction demo
- `models/` – saved trained model, scaler, and feature names
- `results/metrics/` – saved model metrics report
- `results/plots/` – generated plots such as confusion matrix and ROC curve

## How to Run

### Run the training pipeline

From the project root:

```bash
python src/svm_model.py
```

### Run the interactive demo

```bash
python src/demo.py
```

### Open the notebook

Open the notebook in Jupyter:

```bash
jupyter notebook notebooks/team2_svm_model.ipynb
```

## Outputs Generated

The project saves the following outputs:

- `models/best_svm_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.txt`
- `results/metrics/model_metrics.txt`
- `results/plots/confusion_matrix.png`
- `results/plots/roc_curve.png`

## Notes

This project is a strong student-level machine learning project that demonstrates:

- data preparation
- feature engineering
- SVM model development
- hyperparameter tuning
- model evaluation
- visualization
- reproducible artifact saving

It is best viewed as a baseline model for learning and experimentation rather than a fully reliable financial forecasting system.

## Future Improvements

The next steps to improve the project could include:

- stronger time-series validation using time-based splits or walk-forward validation
- comparison with baseline models such as Logistic Regression, Random Forest, or XGBoost
- additional financial features like volatility, momentum, RSI-like indicators, and macroeconomic inputs
- better handling of class imbalance and feature selection
- deployment as a small dashboard or web app
- more complete documentation and final project report

## Conclusion

This project successfully demonstrates how a machine learning pipeline can be built around currency movement prediction using SVMs. It is a good foundation for further study and later enhancement into a more advanced financial forecasting system.
