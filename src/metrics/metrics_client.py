import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================
# Função helper para salvar gráfico
# =========================
def save_plot(filename, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {path}")

# =========================
# Função para plotar métricas por classe
# =========================
def plot_class_metrics(data, save=False):
    metrics = ["precision", "recall", "f1_score"]
    values = [data[m] for m in metrics]
    x = np.arange(len(values[0]))

    plt.figure(figsize=(10,6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*0.25, values[i], width=0.25, label=metric)

    plt.xticks(x + 0.25, [f"Classe {i}" for i in range(len(values[0]))])
    plt.ylabel("Score")
    plt.title("Métricas por classe")
    plt.legend()
    
    if save:
        save_plot("class_metrics.png")
    else:
        plt.show()

# =========================
# Função para plotar acurácia
# =========================
def plot_accuracy(data, save=False):
    plt.figure(figsize=(8,5))
    sns.barplot(x=[f"Classe {i}" for i in range(len(data["accuracy"]))],
                y=data["accuracy"])
    plt.title("Acurácia por classe")

    if save:
        save_plot("accuracy.png")
    else:
        plt.show()

# =========================
# Função para plotar matriz de confusão
# =========================
def plot_confusion_matrix(data, class_idx, save=False):
    cm = np.array(data["confusion_matrix"][class_idx])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusão - Classe {class_idx}")
    plt.xlabel("Predito")
    plt.ylabel("Real")

    if save:
        save_plot(f"confusion_matrix_class_{class_idx}.png")
    else:
        plt.show()

# =========================
# Função para plotar curvas ROC
# =========================
def plot_roc_curves(data, save=False):
    plt.figure(figsize=(8,6))
    for i, roc in enumerate(data["roc"]):
        fpr, tpr = roc
        auc = data["roc_auc"][i]
        plt.plot(fpr, tpr, label=f"Classe {i} (AUC={auc:.3f})")
    
    plt.plot([0,1],[0,1],"k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC por classe")
    plt.legend()
    
    if save:
        save_plot("roc_curves.png")
    else:
        plt.show()

