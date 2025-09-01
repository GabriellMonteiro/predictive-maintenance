# 📊 Manutenção Preditiva

Este projeto apresenta um modelo de **manutenção preditiva** utilizando aprendizado de máquina.  
O objetivo é prever falhas em equipamentos industriais antes que elas ocorram, possibilitando **redução de custos** e **aumento da eficiência operacional**.

---

## 🚀 Tecnologias Utilizadas
- Python 3.10+
- Pandas
- Matplotlib
- Scikit-learn

---

## 🔍 Fluxo do Projeto
1. **Coleta de Dados** – Leitura e pré-processamento do dataset.
2. **Análise Exploratória** – Avaliação de valores nulos e distribuição das falhas.
3. **Treinamento** – Uso de um modelo de classificação (Random Forest).
4. **Avaliação** – Métricas de desempenho do modelo.
5. **Visualização** – Geração de gráficos para interpretação dos resultados.

---

## 📊 Exemplo de Gráfico

Abaixo temos um gráfico ilustrando as principais falhas detectadas:

![Gráfico de Manutenção Preditiva](https://quickchart.io/chart?c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22FDF%20(Falha%20Desgaste%20Ferramenta)%22%2C%22FDC%20(Falha%20Dissipa%C3%A7%C3%A3o%20Calor)%22%2C%22FP%20(Falha%20Pot%C3%AAncia)%22%2C%22FTE%20(Falha%20Tens%C3%A3o%20Excessiva)%22%2C%22FA%20(Falha%20Atuador)%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Quantidade%20de%20Falhas%22%2C%22data%22%3A%5B6880%2C6896%2C3450%2C2980%2C2100%5D%7D%5D%7D%7D)

---

## 📈 Resultados Esperados
- Detecção antecipada de falhas.
- Aumento da vida útil dos equipamentos.
- Redução de paradas não planejadas.
- Melhor alocação de recursos para manutenção.

---

## ▶️ Como Executar
```bash
# Criar ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt

# Rodar o projeto
python main.py
