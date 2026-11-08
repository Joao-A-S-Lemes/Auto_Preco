import pandas as pd
from pathlib import Path

COLUNAS_OBRIGATORIAS = ["year", "selling_price", "km_driven", "fuel",
                        "seller_type", "transmission", "owner",
                        "mileage", "engine", "max_power"]

# Caminho do arquivo
ARQUIVO = Path("data/Car details v3.csv")

# Lê o CSV e devolve um DataFrame; lança ValueError com mensagem clara se falhar
def ler_csv(arquivo=ARQUIVO):
    try:
        df = pd.read_csv(arquivo)
    except FileNotFoundError:
        raise ValueError("Arquivo não encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo CSV está vazio.")
    except pd.errors.ParserError:
        raise ValueError("O arquivo CSV está corrompido ou mal formatado.")
    except UnicodeDecodeError:
        raise ValueError("Não foi possível ler a codificação do arquivo (use UTF-8).")

    if df.empty:
        raise ValueError("O arquivo não contém registros.")
    return df

# Verifica se alguma coluna obrigatória está faltando
def validar_colunas(df):
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(faltando)}")

# Converte "23.4 kmpl", "1248 CC" e "74 bhp" em números (consumo, cilindrada, potência)
def extrair_numero(serie):
    return pd.to_numeric(serie.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")

# Limpeza: converte texto em número, remove duplicatas e linhas com preço/km inválidos
def limpar(df):
    log = {"linhas_antes": len(df)}
    df = df.copy()
    for col in ["mileage", "engine", "max_power"]:
        df[col] = extrair_numero(df[col])

    n_duplicadas = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["duplicatas_removidas"] = n_duplicadas

    antes = len(df)
    df = df[(df["selling_price"] > 0) & (df["km_driven"] >= 0)]
    log["removidas_preco_ou_km_invalido"] = antes - len(df)

    df = df.reset_index(drop=True)
    log["linhas_depois"] = len(df)
    return df, log

if __name__ == "__main__":
    try:
        dados = ler_csv()
        validar_colunas(dados)
        limpo, log = limpar(dados)
        print(f"Sucesso! {dados.shape[0]} linhas e {dados.shape[1]} colunas.")
        print(log)
    except ValueError as e:
        print(f"Erro: {e}")