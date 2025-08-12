# Logistic Regression Binary Classification Project

## Objective
Implement a **binary classification** model using Logistic Regression.

## Dataset
- Place your CSV dataset inside the `data/` folder.
- By default, if no dataset is provided, the script uses the **Breast Cancer Wisconsin dataset** from scikit-learn.

## Features
- Train/Test split
- Standardization of features
- Logistic Regression training
- Evaluation: Confusion Matrix, Precision, Recall, ROC-AUC
- Threshold tuning
- Sigmoid function visualization

## Folder Structure
```
logistic_regression_project/
│
├── data/                 
├── outputs/              
├── src/                  
│   └── logistic_regression.py
├── README.md
└── requirements.txt
```

## How to Run
1. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  
venv\Scripts\activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run script (default sklearn dataset):
```bash
python src/logistic_regression.py
```

4. Run script with your CSV dataset:
```bash
python src/logistic_regression.py --csv data/your_dataset.csv --target target_column_name
```

## Outputs
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/metrics.json`
