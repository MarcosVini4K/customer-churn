import pandas as pd

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

BINARY_COLS = [
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "churn",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica só regras fixas de limpeza: não remove linhas nem preenche ausentes."""
    df = df.rename(columns=COLUMN_MAP).copy()

    df = df.replace({"No internet service": "No", "No phone service": "No"})

    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df.loc[df["tenure"] == 0, "total_charges"] = 0.0

    assert df[BINARY_COLS].notna().all().all(), "valor inesperado nas binárias"
    return df
