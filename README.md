
# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave a service provider. The project includes data analysis, model comparison, and an interactive Streamlit dashboard.

## Project Overview

Customer churn is an important business problem in the telecom industry. Identifying customers who may leave helps businesses understand customer behavior and plan retention strategies.

This project uses historical telecom customer data to estimate churn probability and display insights through an interactive dashboard.

## Features

- Customer churn data analysis
- Business overview with key metrics
- Interactive charts for churn distribution, contract types, monthly charges, and tenure
- Customer-level churn prediction
- Estimated churn probability
- Comparison of Logistic Regression and Random Forest
- Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer demographics, account information, services, contract details, charges, and churn status.

- Total customers after data cleaning: 7,032
- Target column: `Churn`
- `Yes` represents churn; `No` represents customers who stayed.

Dataset source: [IBM Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the downloaded CSV file inside the `data` folder with this filename:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Workflow

1. Load and clean the dataset.
2. Convert `TotalCharges` to a numeric column and handle missing values.
3. Separate the target variable from the input features.
4. Split the data into training and testing sets.
5. Preprocess numeric and categorical features.
6. Train Logistic Regression and Random Forest models.
7. Compare model performance.
8. Save the selected model.
9. Use the model in the Streamlit dashboard.

## Model Evaluation

Both models were evaluated on the same held-out test dataset.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 72.57% | 75.76% |
| Churn Precision | 49.01% | 53.22% |
| Churn Recall | 79.68% | 72.99% |
| Churn F1-score | 0.6069 | 0.6156 |
| ROC-AUC | 0.8351 | 0.8241 |

Logistic Regression was selected based on ROC-AUC. It achieved a ROC-AUC of 0.8351 and churn recall of 79.68%.

The final model is selected using ROC-AUC, with F1-score used to break ties. The best model may vary if the data or training procedure changes.

## Dashboard

The Streamlit dashboard includes:

- Business overview metrics
- Customer churn analysis charts
- Customer details input form
- Churn probability and prediction result

## Project Structure

```text
Customer Churn Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.joblib
└── screenshots/
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── logistic_regression_confusion_matrix.png
    └── random_forest_confusion_matrix.png
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Customer Churn Prediction"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Make sure the dataset is placed in the `data` folder, then run:

```bash
python train_model.py
```

This trains and evaluates both models and saves the selected model in the `models` folder.

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Important Notes

- The dataset is not included in this repository by default. Download it from the source above and place it in the required folder.
- If the saved model file is not included in the repository, run `python train_model.py` before launching the dashboard.
- Predictions are estimates, not guarantees of future customer behavior.
- Model performance depends on the dataset and evaluation setup.

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Model explainability
- Improved prediction threshold selection
- Deployment to a cloud hosting platform
- Additional customer retention insights

## Author

**Ritesh Ahirwar**

Aspiring Data Analyst | Machine Learning Enthusiast
