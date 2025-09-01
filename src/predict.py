# predict.py
from typing import Dict, List
import numpy as np
import pandas as pd

def _probas_from_model(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Retorna matriz (n_samples, n_classes) com prob de classe positiva para cada falha.
    - OneVsRestClassifier: predict_proba -> (n_samples, n_classes)
    - RandomForest multioutput: list of (n_samples, 2)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # RandomForest multioutput retorna lista de arrays
        if isinstance(proba, list) and len(proba) == n_classes:
            return np.column_stack([p[:, 1] for p in proba])
        # OneVsRest retorna np.ndarray
        if isinstance(proba, np.ndarray) and proba.shape[1] == n_classes:
            return proba
        # Alguns wrappers retornam lista np arrays por classe
        if isinstance(proba, list):
            return np.column_stack(proba)
    # Fallback: usar decisão e sigmoid quando não houver proba (raro aqui)
    # mas deixamos robusto:
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        dec = model.decision_function(X)
        return expit(dec)  # transforma em (0,1)
    raise ValueError("O modelo não suporta predict_proba/decision_function")

def predict_test(model, X_test, ids, all_cols, trained_cols):
    y_pred = model.predict(X_test)
    
    # garante que y_pred seja 2D
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    # cria DataFrame com todas as colunas zeradas
    y_pred_df = pd.DataFrame(0, index=np.arange(X_test.shape[0]), columns=all_cols)

    # preenche apenas as colunas que o modelo produziu
    for i, col in enumerate(trained_cols[:y_pred.shape[1]]):
        y_pred_df[col] = y_pred[:, i]

    y_pred_df.insert(0, "id", ids)
    return y_pred_df



