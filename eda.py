
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

sns.set_theme(style="whitegrid")

# 1. Overall Churn Rate
plt.figure(figsize=(7, 5))
churn_counts = df["Churn"].value_counts()
plt.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Overall Customer Churn Distribution")
plt.tight_layout()
plt.show()

# 2. Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 3. Monthly Charges by Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# 4. Tenure by Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()

print("Dataset shape:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nChurn percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))
print("\nCharts and analysis completed!")
