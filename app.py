
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Machine Learning | Customer Analytics | Churn Insights")

# ---------------- LOAD MODEL AND DATA ----------------
if not MODEL_PATH.exists():
    st.error("Model file nahi mila. Pehle train_model.py run karo.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset file nahi mili. Data folder aur CSV filename check karo.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

sns.set_theme(style="whitegrid")

# ---------------- BUSINESS OVERVIEW ----------------
st.subheader("📈 Business Overview")

total_customers = len(df)
churn_count = (df["Churn"] == "Yes").sum()
churn_rate = churn_count / total_customers * 100
avg_monthly = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Customers Churned", f"{churn_count:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

st.divider()

# ---------------- CHARTS ----------------
st.header("🔎 Customer Churn Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Churn Distribution",
    "Contract Analysis",
    "Monthly Charges",
    "Customer Tenure"
])

# Chart 1: Churn Distribution
with tab1:
    st.subheader("Overall Customer Churn")

    churn_data = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.pie(
        churn_data.values,
        labels=["Stayed", "Churned"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9}
    )

    ax.set_title("Customer Churn Distribution", fontsize=11)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# Chart 2: Contract Analysis
with tab2:
    st.subheader("Churn by Contract Type")

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
        ax=ax
    )

    ax.set_title("Churn by Contract", fontsize=11)
    ax.set_xlabel("Contract Type", fontsize=9)
    ax.set_ylabel("Customers", fontsize=9)
    ax.tick_params(axis="x", labelrotation=10, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, title_fontsize=8)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# Chart 3: Monthly Charges
with tab3:
    st.subheader("Monthly Charges and Churn")

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        ax=ax
    )

    ax.set_title("Monthly Charges by Churn", fontsize=11)
    ax.set_xlabel("Churn", fontsize=9)
    ax.set_ylabel("Monthly Charges ($)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# Chart 4: Customer Tenure
with tab4:
    st.subheader("Customer Tenure and Churn")

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        ax=ax
    )

    ax.set_title("Tenure by Churn", fontsize=11)
    ax.set_xlabel("Churn", fontsize=9)
    ax.set_ylabel("Tenure (Months)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

st.divider()

# ---------------- CUSTOMER PREDICTION ----------------
st.header("🔮 Predict Customer Churn")
st.write("Customer details enter karke prediction check karo.")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=2000.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )

# ---------------- PREDICTION RESULT ----------------
if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]

    churn_probability = probabilities[
        list(model.classes_).index(1)
    ]

    st.divider()
    st.subheader("📌 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.1%}"
    )

    result_col2.metric(
        "Prediction",
        "Likely to Churn" if prediction == 1
        else "Likely to Stay"
    )

    st.progress(float(churn_probability))

    if prediction == 1:
        st.warning(
            "Customer churn risk ho sakta hai. "
            "Customer retention support consider karein."
        )
    else:
        st.success(
            "Model ke according customer ke stay karne "
            "ki possibility zyada hai."
        )

    st.caption(
        "Prediction ek estimate hai, guarantee nahi. "
        "Results model aur training data ki quality par depend karte hain."
    )

st.divider()
st.caption("Customer Churn Prediction | Python • Pandas • Scikit-learn • Streamlit")
