# Previsão de Churn de Clientes (Telco)

Modelo de classificação para identificar clientes de uma operadora de telecomunicações com maior risco de cancelar o serviço, a partir do dataset Telco Customer Churn https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data.

## Resumo

- **Problema:** prever quais clientes vão cancelar (churn), para permitir ações de retenção.
- **Dados:** 7.043 clientes, com cerca de 27% de churn (base desbalanceada).
- **Modelo final:** regressão logística com `class_weight="balanced"`, sem a variável `gender`, `C=1`.
- **Resultado no teste:** ROC-AUC 0,842, recall de 0,79 e precisão de 0,50 para a classe churn.
- **Principais fatores de risco:** contrato mês a mês, pouco tempo de casa, internet de fibra óptica e pagamento por cheque eletrônico.

## Metodologia

1. **Overview e limpeza:** padronização dos nomes das colunas, conversão de `total_charges` para número e mapeamento de Yes/No para 1/0.
2. **EDA:** análise da relação entre as variáveis e o churn. O churn é maior entre clientes com menos tempo de permanência, especialmente nos primeiros meses, e varia bastante com o tipo de internet: 42% com fibra óptica, contra 19% com DSL e 7% sem internet.
3. **Pré-processamento:** a função `clean_data` aplica apenas regras fixas. Imputação, one-hot encoding e padronização ficam dentro do `Pipeline` do scikit-learn e são ajustados somente no treino de cada fold, para evitar vazamento de dados.
4. **Validação:** divisão treino/teste estratificada (80/20) e validação cruzada estratificada de 5 folds no treino. O conjunto de teste foi usado uma única vez, no final. Semente fixa (`random_state=42`).
5. **Métricas:** recall, precisão, F1 e ROC-AUC. A acurácia não foi usada para escolher o modelo: um modelo que sempre prevê "ficou" já atinge 73,5%.

## Resultados

Validação cruzada (treino):

| Modelo | Recall | Precisão | F1 | ROC-AUC |
|---|---|---|---|---|
| Regressão logística | 0,543 | 0,654 | 0,593 | 0,846 |
| Regressão logística (balanceada) | 0,803 | 0,518 | 0,629 | 0,846 |
| Árvore de decisão | 0,540 | 0,618 | 0,574 | 0,829 |
| Árvore de decisão (balanceada) | 0,791 | 0,498 | 0,611 | 0,828 |
| Random Forest | 0,491 | 0,637 | 0,554 | 0,826 |
| Random Forest (balanceada) | 0,654 | 0,572 | 0,610 | 0,828 |
| XGBoost | 0,519 | 0,621 | 0,565 | 0,823 |
| XGBoost (balanceado) | 0,658 | 0,551 | 0,600 | 0,823 |
| Baseline (Dummy) | 0,000 | 0,000 | 0,000 | 0,500 |

Teste (modelo final):

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,842 |
| Recall (churn) | 0,79 |
| Precisão (churn) | 0,50 |
| F1 (churn) | 0,61 |

Os resultados do teste ficaram próximos aos da validação cruzada, o que indica boa generalização.

## Decisões

- **Regressão logística como modelo final:** empatou ou superou os modelos baseados em árvores e é interpretável.
- **`class_weight="balanced"`:** aumenta o recall (de 0,54 para 0,80) ao custo de precisão. Faz sentido quando perder um cliente que sairia custa mais do que abordar um que ficaria.
- **Sem `gender`:** o modelo foi testado com e sem a variável, e o desempenho foi o mesmo (ROC-AUC de 0,846).
- **`C=1` (padrão):** o ajuste de `C` entre 0,01 e 10 não trouxe ganho além do ruído da validação cruzada.

## Principais achados

Odds ratios da regressão logística (as variáveis numéricas foram padronizadas, então o efeito é por desvio padrão):

- **Contrato:** mês a mês aumenta as odds de churn (OR = 1,74), e o contrato de dois anos as reduz (OR = 0,41).
- **Tempo de casa:** cada desvio padrão a mais de `tenure` reduz as odds em cerca de 68% (OR = 0,32).
- **Internet:** fibra óptica tem OR = 2,38, contra 0,86 para DSL e 0,29 para quem não contrata internet.
- **Serviços:** suporte técnico e segurança online se associam a menos churn.
- **Pagamento:** cheque eletrônico se associa a mais churn, e pagamento automático, a menos.
- O risco de churn é maior nos primeiros meses de casa , então ações de acompanhamento logo após a contratação tendem a render mais do que ações para clientes antigos.

## Limitações

- Os coeficientes mostram associações, não causas.
- `tenure`, `monthly_charges` e `total_charges` são correlacionadas, então seus coeficientes não devem ser interpretados isoladamente.
- O limiar de decisão é o padrão (0,5), e o `class_weight` desloca as probabilidades previstas. O modelo serve para ordenar e sinalizar clientes, não para estimar probabilidades reais.
- Dados de um único período, sem validação em dados novos.

## Próximos passos

- Ajustar o limiar de decisão conforme o custo de cada tipo de erro.
- Testar modelos de árvore com ajuste de hiperparâmetros e novas variáveis.
- Comparar a associação da EDA (V de Cramér) com a importância no modelo usando permutation importance e SHAP.

## Como executar

```bash
pip install -r requirements.txt
```

Coloque a base original em `data/raw/` e execute os notebooks em ordem.

Para usar o modelo salvo em dados novos (a base precisa estar no formato original):

```python
import joblib
import pandas as pd
from src.cleaning import clean_data

modelo = joblib.load("models/logreg_churn.joblib")
df_novo = clean_data(pd.read_csv("data/raw/novos_clientes.csv"))
X_novo = df_novo.drop(columns=["customer_id", "churn"])  # remova "churn" só se a coluna existir
modelo.predict_proba(X_novo)[:, 1]
```

O arquivo `.joblib` depende da versão do scikit-learn usada no treino (1.9.1). Use a versão fixada no `requirements.txt`.

## Tecnologias

Python, pandas, scikit-learn, XGBoost, matplotlib,seaborn, Jupyter.