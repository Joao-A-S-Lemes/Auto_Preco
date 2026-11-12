import json
import os
from datetime import date

import joblib
import pandas as pd

# Arquivos gerados por treinar.py (pasta models/, ao lado deste arquivo)
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(PASTA_BASE, "models", "modelo.joblib")
CAMINHO_METRICAS = os.path.join(PASTA_BASE, "models", "metricas.json")

# Mesmas colunas, na mesma ordem, usadas no treino (X de treinar.py)
COLUNAS_X = ["fuel", "transmission", "seller_type", "owner", "year", "km_driven"]

# Opções do formulário: valor enviado ao modelo -> texto mostrado ao usuário.
# Os valores à esquerda precisam ser exatamente os mesmos do CSV de treino.
OPCOES = {
    "fuel": {
        "Petrol": "Gasolina",
        "Diesel": "Diesel",
        "CNG": "GNV (CNG)",
        "LPG": "GLP (LPG)",
    },
    "transmission": {
        "Manual": "Manual",
        "Automatic": "Automático",
    },
    "seller_type": {
        "Individual": "Particular",
        "Dealer": "Concessionária",
        "Trustmark Dealer": "Concessionária certificada",
    },
    "owner": {
        "First Owner": "Primeiro dono",
        "Second Owner": "Segundo dono",
        "Third Owner": "Terceiro dono",
        "Fourth & Above Owner": "Quarto dono ou mais",
        "Test Drive Car": "Carro de test drive",
    },
}

# Limites aceitos no formulário (iguais aos usados na limpeza de dados.py)
ANO_MINIMO = 1980
ANO_MAXIMO = date.today().year
KM_MAXIMO = 500_000

_modelo = None   # o modelo é carregado uma vez e reaproveitado


def carregar_modelo():
    # Carrega o pipeline salvo (pré-processamento + regressor) na primeira chamada
    global _modelo
    if _modelo is None:
        if not os.path.exists(CAMINHO_MODELO):
            raise FileNotFoundError(
                "Modelo não encontrado. Rode 'python treinar.py' para gerar models/modelo.joblib."
            )
        _modelo = joblib.load(CAMINHO_MODELO)
    return _modelo


def nome_do_modelo():
    # Nome da classe do regressor escolhido (ex.: RandomForestRegressor)
    return type(carregar_modelo().named_steps["regressor"]).__name__


def carregar_metricas():
    # Resumo do treino salvo por treinar.py (models/metricas.json).
    # Devolve None se o arquivo não existir ou estiver ilegível: a previsão funciona sem ele.
    try:
        with open(CAMINHO_METRICAS, encoding="utf-8") as f:
            resumo = json.load(f)
        escolhido = resumo["resultados"][resumo["modelo_escolhido"]]
        comparacao = [
            {
                "nome": nome,
                "mae": valores["mae"],
                "r2": valores["r2"],
                "erro_percentual": valores["erro_percentual"],
            }
            for nome, valores in resumo["resultados"].items()
        ]
        return {
            "modelo": resumo["modelo_escolhido"],
            "mae": escolhido["mae"],
            "r2": escolhido["r2"],
            "erro_percentual": escolhido["erro_percentual"],
            "linhas_treino": resumo["linhas_treino"],
            "linhas_teste": resumo["linhas_teste"],
            "comparacao": comparacao,
            "versao_scikit_learn": resumo.get("versao_scikit_learn"),
        }
    except (OSError, ValueError, KeyError):
        return None


def validar_entrada(form):
    # Recebe os campos do formulário (dict-like) e devolve um dicionário limpo.
    # Se algo estiver errado, levanta ValueError com mensagem em português.
    try:
        ano = int(str(form.get("year", "")).strip())
    except ValueError:
        raise ValueError("Informe o ano de fabricação como um número inteiro (ex.: 2015).")
    if not ANO_MINIMO <= ano <= ANO_MAXIMO:
        raise ValueError(f"O ano deve estar entre {ANO_MINIMO} e {ANO_MAXIMO}.")

    try:
        km = float(str(form.get("km_driven", "")).strip().replace(",", "."))
    except ValueError:
        raise ValueError("Informe a quilometragem como um número (ex.: 60000).")
    if not 0 <= km <= KM_MAXIMO:
        raise ValueError(f"A quilometragem deve estar entre 0 e {KM_MAXIMO:,}.".replace(",", "."))

    entrada = {"year": ano, "km_driven": km}

    for campo, opcoes in OPCOES.items():
        valor = form.get(campo, "")
        if valor not in opcoes:
            raise ValueError("Escolha uma opção válida em todos os campos de seleção.")
        entrada[campo] = valor

    return entrada


def prever(entrada):
    # Monta UMA linha com as mesmas colunas do treino e aplica predict().
    # O pré-processamento (OneHotEncoder) já está dentro do pipeline salvo.
    linha = pd.DataFrame([entrada], columns=COLUNAS_X)
    preco = float(carregar_modelo().predict(linha)[0])
    return max(preco, 0.0)   # regressão linear pode devolver valor negativo; preço mínimo é 0
