from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["gender", "internet_service", "contract", "payment_method"]
NUMERICAL_FEATURES = ["tenure", "monthly_charges", "total_charges"]
BINARY_FEATURES = [
    "senior_citizen",
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
]


def create_preprocessor(scale_numeric=True):
    """
    Cria o pré-processador das variáveis preditoras.

    Parâmetros
    ----------
    scale_numeric : bool
        Define se as variáveis numéricas contínuas serão padronizadas.


    Retorna
    -------
    ColumnTransformer
        Codificação categórica, imputação (mediana) e padronização opcional
        das numéricas. As binárias devem já estar codificadas como 0/1.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", Pipeline(numeric_steps), NUMERICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor
