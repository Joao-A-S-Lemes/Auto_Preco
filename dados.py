import pandas as pd
from pathlib import Path

COLUNAS_OBRIGATORIAS = ["year", "selling_price", "km_driven", "fuel",
                        "seller_type", "transmission", "owner",
                        "mileage", "engine", "max_power"]

Arquivo = Path("data/Car details v3.csv")

def ler_csv(arquivo=Arquivo):
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

if __name__ == "__main__":
    dados = ler_csv()
    print(f"Sucesso! {dados.shape[0]} linhas e {dados.shape[1]} colunas.")

