
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

import joblib


# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATA_PATH = os.path.join(
    "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = "models"
OUTPUT_DIR = "screenshots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["TotalCharges"]).copy()

print("Dataset shape:", df.shape)

# Remove customer ID because it is not a useful
# customer behavior predictor.
df = df.drop(columns=["customerID"], errors="ignore")

# Convert target into 0 and 1.
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify numeric and categorical columns.
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category", "string"]
).columns.tolist()

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# 5. DEFINE BOTH MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

results = []
trained_models = {}

for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print(f"\n{model_name} Results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    # Save a confusion matrix for each model.
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# 7. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(
    results_df.to_string(
        index=False,
        formatters={
            column: (lambda value: f"{value:.4f}")
            for column in [
                "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC"
            ]
        }
    )
)

comparison_path = os.path.join(
    OUTPUT_DIR, "model_comparison.csv"
)

results_df.to_csv(comparison_path, index=False)


# ==========================================
# 8. SAVE COMPARISON CHART
# ==========================================

metrics = [
    "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
]

chart_df = results_df.set_index("Model")[metrics]

ax = chart_df.T.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0
)

ax.set_title("Logistic Regression vs Random Forest")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")
ax.legend(title="Model")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR, "model_comparison.png"
)

plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()


# ==========================================
# 9. SELECT AND SAVE BEST MODEL
# ==========================================

# ROC-AUC is the primary comparison metric.
# F1 Score is used to break ties.
best_result = max(
    results,
    key=lambda row: (row["ROC-AUC"], row["F1 Score"])
)

best_model_name = best_result["Model"]
best_model = trained_models[best_model_name]

model_path = os.path.join(
    MODEL_DIR, "churn_model.joblib"
)

joblib.dump(best_model, model_path)

print("\n========== FINAL RESULT ==========")
print("Selected model:", best_model_name)
print(f"Selected model ROC-AUC: {best_result['ROC-AUC']:.4f}")
print("Saved model:", os.path.abspath(model_path))
print("Comparison CSV:", os.path.abspath(comparison_path))
print("Comparison chart:", os.path.abspath(chart_path))
print("Confusion matrices saved in:", os.path.abspath(OUTPUT_DIR))

print("\nStep 20 completed successfully!")
