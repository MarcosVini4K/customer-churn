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


def create_preprocessor(
    scale_numeric=True, categorical_features=None, binary_features=None
):
    """
    Cria o pré-processador das variáveis preditoras.

    Parâmetros
    ----------
    scale_numeric : bool
        Define se as variáveis numéricas contínuas serão padronizadas.
    categorical_features : list, opcional
        Variáveis categóricas para one-hot. Padrão: CATEGORICAL_FEATURES.
    binary_features : list, opcional
        Variáveis já codificadas como 0/1. Padrão: BINARY_FEATURES.

    Retorna
    -------
    ColumnTransformer
        Codificação categórica, imputação (mediana) e padronização opcional
        das numéricas. As binárias devem já estar codificadas como 0/1.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if binary_features is None:
        binary_features = BINARY_FEATURES

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
                categorical_features,
            ),
            ("numerical", Pipeline(numeric_steps), NUMERICAL_FEATURES),
            ("binary", "passthrough", binary_features),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor
