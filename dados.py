import pandas as pd

# Colunas obrigatórias dentro do csv enviado
COLUNAS_OBRIGATORIAS = [
    "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner",
]

# Dessas as que precisam ser números
COLUNAS_NUMERICAS = [
    "year", "selling_price", "km_driven"
]

# Lê o CSV e valida o formato.
# 'arquivo' pode ser um caminho (str) ou o arquivo enviado no Flask.
# Se algo estiver errado, levanta ValueError com uma mensagem em português.
def ler_csv(arquivo):
    try:
        df = pd.read_csv(arquivo)
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo está vazio.")
    except UnicodeDecodeError:
        raise ValueError("Não foi possível ler o arquivo. Salve-o como CSV em UTF-8.")
    except pd.errors.ParserError:
        raise ValueError("O arquivo não parece ser um csv válido.")

    if df.empty:
        raise ValueError("O arquivo tem o cabeçalho, mas nenhuma linha de dados.")

    # Colunas obrigatórias (se o separador for ';' em vez de ',', cai aqui também)
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(faltando) +
            ". Confira os nomes e se o separador do arquivo é vírgula."
        )

    # Valores incompatíveis: texto onde deveria haver número
    for col in COLUNAS_NUMERICAS:
        convertida = pd.to_numeric(df[col], errors="coerce")
        invalidos = convertida.isna() & df[col].notna()
        if invalidos.any():
            raise ValueError(
                f"A coluna '{col}' tem {invalidos.sum()} valor(es) que não são números."
            )
        df[col] = convertida

    return df