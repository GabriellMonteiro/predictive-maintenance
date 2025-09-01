from sklearn.metrics import f1_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_eda(df, falhas, save_dir="plots"):
    """
    Exploração dos dados com gráficos em PNG:
    - Distribuição das classes de falha
    - Estatísticas descritivas
    - Histogramas, Boxplots
    - Correlação
    - Falhas por tipo de máquina
    - Boxplots de features vs falhas
    - Pairplot das features numéricas
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ----------------------
    # Distribuição das classes
    # ----------------------
    print("Distribuição das classes:")
    print(df[falhas].sum())

    plt.figure(figsize=(8,5))
    df[falhas] = df[falhas].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    df[falhas].sum().sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title("Distribuição de cada tipo de falha")
    plt.ylabel("Quantidade")
    plt.savefig(f"{save_dir}/dist_falhas.png")
    plt.close()

    # ----------------------
    # Estatísticas descritivas
    # ----------------------
    print("\nEstatísticas descritivas das features:")
    print(df.describe())

    # ----------------------
    # Colunas numéricas
    # ----------------------
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in falhas + ["id", "id_produto"]]

    # ----------------------
    # Histogramas
    # ----------------------
    df[numeric_cols].hist(figsize=(12,10))
    plt.suptitle("Histogramas das Features")
    plt.savefig(f"{save_dir}/histogramas.png")
    plt.close()

    # ----------------------
    # Boxplots
    # ----------------------
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Boxplots das Features")
    plt.savefig(f"{save_dir}/boxplots.png")
    plt.close()

    # ----------------------
    # Mapa de Correlação
    # ----------------------
    plt.figure(figsize=(12,10))
    corr = df[numeric_cols + falhas].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de Correlação entre Features e Falhas")
    plt.savefig(f"{save_dir}/correlacao.png")
    plt.close()

    # ----------------------
    # Falhas por tipo de máquina
    # ----------------------
    plt.figure(figsize=(10,6))
    df_melted = df.melt(id_vars="tipo", value_vars=falhas, var_name="falha", value_name="ocorreu")
    sns.countplot(data=df_melted[df_melted["ocorreu"]==1], x="falha", hue="tipo")
    plt.title("Falhas por tipo de máquina")
    plt.xticks(rotation=45)
    plt.savefig(f"{save_dir}/falhas_por_tipo.png")
    plt.close()

    # ----------------------
    # Boxplots de features vs falhas
    # ----------------------
    for col in numeric_cols:
        for falha in falhas:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[falha], y=df[col])
            plt.title(f"{col} vs {falha}")
            plt.savefig(f"{save_dir}/box_{col}_{falha.replace(' ','_')}.png")
            plt.close()

    # ----------------------
    # Pairplot das features numéricas
    # ----------------------
    pairplot_sample = df[numeric_cols].sample(min(500, len(df)))
    sns.pairplot(pairplot_sample)
    plt.savefig(f"{save_dir}/pairplot_features.png")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_dir="plots"):
    """
    Plota a importância das features para modelos que possuem feature_importances_
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Verifica se é MultiOutputClassifier
    if hasattr(model, "estimators_"):  
        base_model = model.estimators_[0]  
    else:
        base_model = model

    if not hasattr(base_model, "feature_importances_"):
        print(f"Modelo {model.__class__.__name__} não possui feature_importances_")
        return

    importances = base_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp.sort_values(by="importance", ascending=False, inplace=True)

    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=feat_imp.head(top_n))
    plt.title(f"Top {top_n} Features - {model.__class__.__name__}")
    plt.savefig(f"{save_dir}/feature_importance_{model.__class__.__name__}.png")
    plt.close()

def plot_model_comparison(results_df, save_dir="plots"):
    """
    Plota a comparação de métricas entre diferentes modelos.
    results_df: DataFrame com colunas ["Modelo", "F1_macro", "Hamming Loss"]
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # ----------------------
    # F1 Macro
    # ----------------------
    plt.figure(figsize=(8,5))
    sns.barplot(x="Modelo", y="F1_macro", data=results_df, palette="viridis")
    plt.title("Comparação do F1 Macro entre modelos")
    plt.ylim(0, 1)  # Ajuste conforme necessário
    for index, row in results_df.iterrows():
        plt.text(index, row.F1_macro + 0.01, f"{row.F1_macro:.3f}", ha='center')
    plt.savefig(f"{save_dir}/comparacao_f1_macro.png")
    plt.close()
    
    # ----------------------
    # Hamming Loss
    # ----------------------
    plt.figure(figsize=(8,5))
    sns.barplot(x="Modelo", y="Hamming Loss", data=results_df, palette="magma")
    plt.title("Comparação do Hamming Loss entre modelos")
    for index, row in results_df.iterrows():
        plt.text(index, row["Hamming Loss"] + 0.001, f"{row['Hamming Loss']:.4f}", ha='center')
    plt.savefig(f"{save_dir}/comparacao_hamming_loss.png")
    plt.close()

def compare_models(trained_models, X_val, y_val, trained_cols, save_dir="plots"):
    """
    Avalia múltiplos modelos usando dados de validação e gera gráficos comparativos.
    - trained_models: dict de modelos treinados
    - X_val: features de validação
    - y_val: labels de validação
    - trained_cols: dict com as colunas usadas em cada modelo
    - save_dir: diretório onde os gráficos serão salvos
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []

    for name, model in trained_models.items():
        cols = trained_cols.get(name, y_val.columns.tolist())  # fallback para todas as colunas
        y_pred = model.predict(X_val)

        # garante 2D
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        # cria DataFrame para alinhar com y_val
        y_pred_df = pd.DataFrame(0, index=np.arange(X_val.shape[0]), columns=y_val.columns)
        for i, col in enumerate(cols[:y_pred.shape[1]]):
            y_pred_df[col] = y_pred[:, i]

        # Calcula métricas
        f1 = f1_score(y_val, y_pred_df, average="macro", zero_division=0)
        hamming = hamming_loss(y_val, y_pred_df)

        results.append({
            "Modelo": name,
            "F1_macro": f1,
            "Hamming Loss": hamming
        })

    results_df = pd.DataFrame(results)
    print("\nComparação de modelos:\n", results_df)

    # Gráfico de F1_macro
    plt.figure(figsize=(8,5))
    sns.barplot(x="F1_macro", y="Modelo", data=results_df, hue="Modelo", dodge=False)
    plt.title("Comparação de F1_macro entre modelos")
    plt.xlabel("F1_macro")
    plt.ylabel("Modelo")
    plt.xlim(0, 1)
    plt.legend([],[], frameon=False)  # remove legenda extra
    plt.savefig(f"{save_dir}/comparacao_f1_macro.png")
    plt.close()

    # Gráfico de Hamming Loss
    plt.figure(figsize=(8,5))
    sns.barplot(x="Hamming Loss", y="Modelo", data=results_df, hue="Modelo", dodge=False)
    plt.title("Comparação de Hamming Loss entre modelos")
    plt.xlabel("Hamming Loss")
    plt.ylabel("Modelo")
    plt.legend([],[], frameon=False)
    plt.savefig(f"{save_dir}/comparacao_hamming_loss.png")
    plt.close()


def plot_maintenance_risk(predictions_df, save_dir="plots"):
    """
    Gera gráficos de risco de falha por tipo de falha.
    predictions_df: DataFrame retornado por predict_test (com colunas de falhas)
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calcula a taxa média de falha prevista por tipo de falha
    risk = predictions_df.iloc[:, 1:].mean().sort_values(ascending=False)  # ignora coluna 'id'

    plt.figure(figsize=(10,6))
    sns.barplot(x=risk.values, y=risk.index, hue=risk.index, dodge=False, legend=False, palette="Reds_r")
    plt.title("Risco Médio de Falha por Tipo de Falha")
    plt.xlabel("Probabilidade Média de Falha")
    plt.ylabel("Tipo de Falha")
    plt.xlim(0,1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/risco_falhas.png")
    plt.close()