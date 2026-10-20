"""Utilidades de modelado: split temporal, CV por mes, preprocesamiento y umbral."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_temporal(panel: pd.DataFrame, ultimo_mes_train: str):
    """Train = fotos hasta `ultimo_mes_train` inclusive; test = fotos posteriores."""
    train = panel[panel["mes_foto"] <= ultimo_mes_train].reset_index(drop=True)
    test = panel[panel["mes_foto"] > ultimo_mes_train].reset_index(drop=True)
    return train, test


def cv_ventana_expansiva(train: pd.DataFrame, meses_validacion):
    """Folds tipo 'walk-forward': entreno con meses < m y valido en el mes m."""
    return [(np.where(train["mes_foto"] < m)[0], np.where(train["mes_foto"] == m)[0])
            for m in meses_validacion]


def crear_preprocesador(num_cols, cat_cols, escalar=True):
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(pasos_num), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def metricas(y_true, proba, umbral=0.5) -> dict:
    pred = (proba >= umbral).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def umbral_para_recall(y_true, proba, recall_objetivo=0.80) -> float:
    """Mayor umbral que todavía alcanza el recall objetivo (= máxima precisión posible)."""
    for u in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if recall_score(y_true, (proba >= u).astype(int)) >= recall_objetivo:
            return float(u)
    return 0.5
