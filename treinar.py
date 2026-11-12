import json
import os
from datetime import datetime

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dados import ler_csv, limpar   # reaproveita a leitura e a limpeza já validadas

CAMINHO_CSV = "data/Car details v3.csv"
CAMINHO_MODELO = "models/modelo.joblib"
CAMINHO_METRICAS = "models/metricas.json"
RANDOM_STATE = 42

# X (variáveis usadas para prever) e y (o que queremos prever)
FEATURES_CATEGORICAS = ["fuel", "transmission", "seller_type", "owner"]
FEATURES_NUMERICAS = ["year", "km_driven"]
ALVO = "selling_price"


def criar_pipeline(modelo):
    # Pré-processamento + modelo em um só objeto.
    # O OneHotEncoder só é ajustado quando chamamos fit() com os dados de TREINO.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             FEATURES_CATEGORICAS),
        ],
        remainder="passthrough",   # year e km_driven passam sem alteração
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", modelo)])


def treinar(caminho_csv=CAMINHO_CSV):
    # 1. Leitura e limpeza (as mesmas do dados.py, ANTES de dividir treino e teste)
    df = ler_csv(caminho_csv)
    df_limpo, registro, tamanhos = limpar(df)
    print(f"Base: {tamanhos['antes']} linhas antes da limpeza, {tamanhos['depois']} depois.")

    X = df_limpo[FEATURES_CATEGORICAS + FEATURES_NUMERICAS]
    y = df_limpo[ALVO]

    # 2. Divisão treino/teste: 80% / 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_train)} amostras (80%) | Teste: {len(X_test)} amostras (20%)")
    print(f"random_state: {RANDOM_STATE}\n")

    # 3. Comparação dos dois modelos (mesma divisão, mesmas métricas)
    modelos = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
    }
    preco_medio = y_test.mean()
    resultados = {}

    for nome, modelo in modelos.items():
        pipeline = criar_pipeline(modelo)
        pipeline.fit(X_train, y_train)        # ajusta encoder e modelo SÓ no treino
        y_pred = pipeline.predict(X_test)     # prevê no teste (dados nunca vistos)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        resultados[nome] = {
            "pipeline": pipeline,
            "mae": float(mae),
            "r2": float(r2),
            "erro_percentual": float(mae / preco_medio * 100),
        }
        print(f"Modelo: {nome}")
        print(f"  MAE: Rs {mae:,.0f} ({mae / preco_medio * 100:.1f}% do preço médio de teste)")
        print(f"  R2:  {r2:.4f}\n")

    # 4. Escolha final: menor MAE
    melhor_nome = min(resultados, key=lambda k: resultados[k]["mae"])
    print(f"==> Escolha final: {melhor_nome} (menor MAE)")

    # 5. Salva o Pipeline completo e um resumo do treino
    os.makedirs("models", exist_ok=True)
    joblib.dump(resultados[melhor_nome]["pipeline"], CAMINHO_MODELO)

    resumo = {
        "base_de_treino": caminho_csv,
        "treinado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "versao_scikit_learn": sklearn.__version__,
        "linhas_antes_da_limpeza": tamanhos["antes"],
        "linhas_depois_da_limpeza": tamanhos["depois"],
        "linhas_treino": len(X_train),
        "linhas_teste": len(X_test),
        "random_state": RANDOM_STATE,
        "modelo_escolhido": melhor_nome,
        "resultados": {n: {k: v for k, v in r.items() if k != "pipeline"}
                       for n, r in resultados.items()},
    }
    with open(CAMINHO_METRICAS, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"Modelo salvo em '{CAMINHO_MODELO}' e resumo em '{CAMINHO_METRICAS}'.")
    return resumo


if __name__ == "__main__":
    treinar()