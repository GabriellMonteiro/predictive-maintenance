from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, hamming_loss
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Função para dividir os dados
# =========================
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# =========================
# Função para treinar múltiplos modelos
# =========================
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

def train_multiple_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }
    
    trained_models = {}
    trained_cols = {}  # vai guardar quais colunas foram efetivamente treinadas

    for name, base_model in models.items():
        # Seleciona colunas que têm mais de uma classe
        valid_cols = [col for col in y_train.columns if len(y_train[col].unique()) > 1]
        if not valid_cols:
            print(f"Nenhuma coluna válida para treinar {name}, pulando...")
            continue

        print(f"Treinando {name} nas colunas: {valid_cols}")
        multi_model = MultiOutputClassifier(base_model)
        multi_model.fit(X_train, y_train[valid_cols])
        
        trained_models[name] = multi_model
        trained_cols[name] = y_train.columns.tolist()

    return trained_models, trained_cols


# =========================
# Função para avaliar modelos
# =========================
def evaluate_models(models, X_val, y_val, falhas):
    metrics = {"Model": [], "F1 Score": [], "Hamming Loss": []}

    # Converter y_val para DataFrame se não for
    if not isinstance(y_val, pd.DataFrame):
        y_val = pd.DataFrame(y_val, columns=falhas)

    for name, model in models.items():
        # Previsão
        y_pred = model.predict(X_val)

        # Garantir que seja DataFrame e substituir NaN por 0
        y_pred = pd.DataFrame(y_pred, columns=y_val.columns, index=y_val.index)
        y_pred = y_pred.fillna(0)

        # Métricas
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        hamming = hamming_loss(y_val, y_pred)

        metrics["Model"].append(name)
        metrics["F1 Score"].append(f1)
        metrics["Hamming Loss"].append(hamming)

        print(f"\n=== Avaliação {name} ===")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")

    # Plot comparação
    metrics_df = pd.DataFrame(metrics)
    metrics_df.plot(x="Model", y=["F1 Score", "Hamming Loss"], kind="bar", figsize=(8,5), title="Comparação entre modelos")
    plt.ylabel("Score")
    plt.show()
