from data_loader import load_data
from preprocessing import preprocess_data
from model import split_data, train_multiple_models
from predict import predict_test
from visualization import plot_maintenance_risk, compare_models, plot_feature_importance, plot_eda
from metrics.metrics_client import plot_class_metrics, plot_accuracy, plot_confusion_matrix, plot_roc_curves
from api.fetch_data import send_csv
import pandas as pd

def main():

    train_path = "../data/bootcamp_train.csv"
    test_path = "../data/bootcamp_test.csv"
    
    falhas = [
        "FDF (Falha Desgaste Ferramenta)", 
        "FDC (Falha Dissipacao Calor)", 
        "FP (Falha Potencia)", 
        "FTE (Falha Tensao Excessiva)", 
        "FA (Falha Aleatoria)"
    ]

    # ==============================
    # Carregar dados
    # ==============================
    train, test = load_data(train_path, test_path)
    print(train.head())

    # ==============================
    # Exploração dos dados (EDA)
    # ==============================
    plot_eda(train, falhas)

    # ==============================
    # Pré-processamento
    # ==============================
    X_scaled, y, X_test_scaled, feature_names = preprocess_data(train, test, falhas)
    print(pd.DataFrame(X_scaled, columns=feature_names).head(20))

    # ==============================
    # Divisão treino/validação
    # ==============================
    X_train, X_val, y_train, y_val = split_data(X_scaled, y)

    # ==============================
    # Treinar múltiplos modelos
    # ==============================
    trained_models, trained_cols = train_multiple_models(X_train, y_train)

    # ==============================
    # Avaliar modelos
    # ==============================
    rf_cols = trained_cols["Random Forest"]
    # Agora passa todos os argumentos

    # ==============================
    # Previsões para teste
    # ==============================
    final_preds = predict_test(trained_models["Random Forest"], X_test_scaled, test["id"], falhas, rf_cols)
    final_preds.to_csv("predicoes.csv", index=False)
    print(final_preds.head())

    # ==============================
    # Importância das features
    # ==============================
    plot_feature_importance(trained_models["Random Forest"], feature_names)
    plot_feature_importance(trained_models["XGBoost"], feature_names)

    # ==============================
    # Avaliar e comparar modelos
    # ==============================
    compare_models(trained_models, X_val, y_val, trained_cols)

    # Plot de risco de manutenção preditiva
    plot_maintenance_risk(final_preds)

    # ==============================
    # CONEXÃO COM A API
    # ==============================
    data = send_csv(csv_path="predicoes.csv", username="gabrielbrava@outlook.com.br", password="197428")

    # Plotar gráficos das métricas
    plot_class_metrics(data, save=True)
    plot_accuracy(data, save=True)
    plot_confusion_matrix(data, class_idx=3, save=True)
    plot_roc_curves(data, save=True)

if __name__ == "__main__":
    main()
