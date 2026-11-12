import pandas as pd
from datetime import date
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # modo sem janela: obrigatório para usar matplotlib dentro do Flask
import matplotlib.pyplot as plt

# Colunas obrigatórias dentro do csv enviado
COLUNAS_OBRIGATORIAS = [
    "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner",
]

# Dessas as que precisam ser números
COLUNAS_NUMERICAS = [
    "year", "selling_price", "km_driven"
]

# Categorias aceitas no dataset de referência. Valores fora dessas listas
# são rejeitados para evitar análises com categorias inesperadas ou erros de digitação.
CATEGORIAS_ACEITAS = {
    "fuel": {"Petrol", "Diesel", "CNG", "LPG"},
    "seller_type": {"Individual", "Dealer", "Trustmark Dealer"},
    "transmission": {"Manual", "Automatic"},
    "owner": {
        "First Owner", "Second Owner", "Third Owner",
        "Fourth & Above Owner", "Test Drive Car",
    },
}

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
        # NaN será tratado pela etapa de limpeza; infinito não é um valor válido.
        infinitos = ~np.isfinite(convertida.dropna().astype(float))
        if infinitos.any():
            raise ValueError(f"A coluna '{col}' contém valor(es) infinitos ou inválidos.")
        df[col] = convertida

    # O ano representa um ano-calendário, portanto precisa ser inteiro.
    anos_validos = df["year"].dropna()
    if ((anos_validos % 1) != 0).any():
        raise ValueError("A coluna 'year' deve conter anos inteiros (ex.: 2018).")

    # Validação de categorias: não aceitar silenciosamente grafias ou classes desconhecidas.
    for coluna, permitidas in CATEGORIAS_ACEITAS.items():
        presentes = set(df[coluna].dropna().astype(str).str.strip().unique())
        desconhecidas = sorted(presentes - permitidas)
        if desconhecidas:
            exemplos = ", ".join(desconhecidas[:5])
            raise ValueError(
                f"A coluna '{coluna}' contém categoria(s) não reconhecida(s): {exemplos}. "
                f"Confira as opções aceitas no README."
            )
        # Remove espaços acidentais nas categorias conhecidas.
        df[coluna] = df[coluna].apply(lambda valor: valor.strip() if isinstance(valor, str) else valor)

    return df

# Limites usados na limpeza
ANO_MINIMO = 1950
ANO_MAXIMO = date.today().year
KM_MAXIMO = 500_000   # acima disso, muito provavelmente é erro de digitação

def _mascara_impossiveis(df):
    # True nas linhas com valores impossíveis
    return (
        (df["selling_price"] <= 0)
        | (df["km_driven"] < 0)
        | (df["year"] < ANO_MINIMO)
        | (df["year"] > ANO_MAXIMO)
    )

def diagnosticar(df):
    # Resumo da base ANTES da limpeza
    return {
        "linhas": len(df),
        "colunas": df.shape[1],
        "tipos": df.dtypes.astype(str).to_dict(),
        "ausentes": {c: int(n) for c, n in df.isna().sum().items()},
        "duplicadas": int(df.duplicated().sum()),
        "impossiveis": int(_mascara_impossiveis(df).sum()),
        "km_acima_do_limite": int((df["km_driven"] > KM_MAXIMO).sum()),
    }

def limpar(df):
    # Limpa uma CÓPIA do DataFrame (o CSV original nunca é alterado)
    # e devolve: base limpa, registro das transformações e tamanhos antes/depois
    df = df.copy()
    antes = len(df)
    registro = []

    def registrar(etapa, tamanho_anterior, justificativa):
        registro.append({
            "etapa": etapa,
            "removidas": tamanho_anterior - len(df),
            "justificativa": justificativa,
        })

    n = len(df)
    df = df.drop_duplicates()
    registrar("Remover duplicatas", n,
              "Linhas idênticas são o mesmo anúncio repetido.")

    n = len(df)
    df = df.dropna(subset=COLUNAS_OBRIGATORIAS)
    registrar("Remover linhas com valores ausentes", n,
              "Sem todas as variáveis de X e o y a linha não pode ser usada.")

    n = len(df)
    df = df[~_mascara_impossiveis(df)]
    registrar("Remover valores impossíveis", n,
              "Preço <= 0, km negativo ou ano fora do intervalo possível.")

    n = len(df)
    df = df[df["km_driven"] <= KM_MAXIMO]
    registrar("Remover km_driven extremo", n,
              "Acima de 500.000 km é muito provavelmente erro de digitação.")

    df = df.reset_index(drop=True)
    return df, registro, {"antes": antes, "depois": len(df)}

def _formatar_rs(valor):
    # 639765 -> "639.765" (formato brasileiro, só para exibir)
    return f"{valor:,.0f}".replace(",", ".")


def _agrupar(df, coluna):
    # Preço por categoria: quantidade, média e mediana, da maior média para a menor
    return (
        df.groupby(coluna)["selling_price"]
        .agg(quantidade="count", media="mean", mediana="median")
        .round(0).astype(int)
        .sort_values("media", ascending=False)
        .reset_index()
    )


def achados(df):
    # Recebe a base LIMPA e devolve uma lista com 3 achados:
    # cada um tem título, tabela (lista de dicionários) e uma frase de resumo
    lista = []

    # Achado 1: combustível
    tab = _agrupar(df, "fuel")
    topo, base = tab.iloc[0], tab.iloc[-1]
    lista.append({
        "titulo": "Preço médio por combustível",
        "tabela": tab.to_dict("records"),
        "texto": (f"O combustível com maior preço médio é {topo['fuel']} "
                  f"(₹ {_formatar_rs(topo['media'])}) e o menor é {base['fuel']} "
                  f"(₹ {_formatar_rs(base['media'])})."),
    })

    # Achado 2: transmissão
    tab = _agrupar(df, "transmission")
    topo, base = tab.iloc[0], tab.iloc[-1]
    razao = topo["media"] / base["media"]
    lista.append({
        "titulo": "Preço médio por transmissão",
        "tabela": tab.to_dict("records"),
        "texto": (f"Os carros {topo['transmission']} custam em média {razao:.1f}x o preço dos "
                  f"{base['transmission']} (₹ {_formatar_rs(topo['media'])} contra "
                  f"₹ {_formatar_rs(base['media'])})."),
    })

    # Achado 3: relação com ano e km (correlação de Pearson, vai de -1 a 1)
    corr_ano = float(df["year"].corr(df["selling_price"]))
    corr_km = float(df["km_driven"].corr(df["selling_price"]))
    lista.append({
        "titulo": "Relação do preço com ano e quilometragem",
        "tabela": [
            {"variavel": "year", "correlacao": round(corr_ano, 2)},
            {"variavel": "km_driven", "correlacao": round(corr_km, 2)},
        ],
        "texto": (f"A correlação entre ano e preço é {corr_ano:.2f} e entre "
                  f"quilometragem e preço é {corr_km:.2f}."),
    })

    return lista

def _salvar(fig, pasta, nome):
    # Salva a figura como PNG na pasta (ex.: static) e libera a memória
    os.makedirs(pasta, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, nome), dpi=100)
    plt.close(fig)
    return nome


def grafico_distribuicao(df, pasta):
    # Gráfico 1 (distribuição): histograma do preço de venda
    preco = df["selling_price"] / 1_000_000   # em milhões de rupias, para os eixos ficarem legíveis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(preco, bins=40, color="#4C78A8", edgecolor="white")
    ax.axvline(preco.median(), color="#E45756", linestyle="--",
               label=f"Mediana: ₹ {preco.median():.2f} mi")
    ax.set_title("Distribuição do preço de venda")
    ax.set_xlabel("Preço de venda (₹ milhões)")
    ax.set_ylabel("Quantidade de carros")
    ax.legend()
    arquivo = _salvar(fig, pasta, "distribuicao_preco.png")

    abaixo = (df["selling_price"] < 1_000_000).mean() * 100
    texto = (f"{abaixo:.0f}% dos carros custam menos de ₹ 1 milhão. "
             f"A mediana é ₹ {_formatar_rs(df['selling_price'].median())} e a média "
             f"₹ {_formatar_rs(df['selling_price'].mean())}; o preço máximo chega a "
             f"₹ {_formatar_rs(df['selling_price'].max())}.")
    return {"arquivo": arquivo, "titulo": "Distribuição do preço de venda", "texto": texto}


def grafico_combustivel(df, pasta):
    # Gráfico 2 (comparação): preço médio por combustível
    tab = (df.groupby("fuel")["selling_price"].agg(["mean", "count"])
             .sort_values("mean", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(tab.index, tab["mean"] / 1_000_000, color="#4C78A8")
    ax.bar_label(barras, labels=[f"n={n}" for n in tab["count"]])   # quantidade de carros de cada grupo
    ax.set_title("Preço médio por tipo de combustível")
    ax.set_xlabel("Combustível")
    ax.set_ylabel("Preço médio (₹ milhões)")
    arquivo = _salvar(fig, pasta, "preco_combustivel.png")

    topo, base = tab.index[0], tab.index[-1]
    texto = (f"{topo} tem o maior preço médio (₹ {_formatar_rs(tab['mean'].iloc[0])}) e "
             f"{base} o menor (₹ {_formatar_rs(tab['mean'].iloc[-1])}). "
             f"Os números 'n' mostram quantos carros há em cada grupo.")
    return {"arquivo": arquivo, "titulo": "Preço médio por combustível", "texto": texto}


def grafico_ano_preco(df, pasta):
    # Gráfico 3 (relação): dispersão entre ano de fabricação e preço
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["year"], df["selling_price"] / 1_000_000, s=10, alpha=0.3, color="#4C78A8")
    ax.set_title("Ano de fabricação x preço de venda")
    ax.set_xlabel("Ano de fabricação")
    ax.set_ylabel("Preço de venda (₹ milhões)")
    arquivo = _salvar(fig, pasta, "ano_preco.png")

    corr = float(df["year"].corr(df["selling_price"]))
    texto = (f"Cada ponto é um carro. A correlação entre ano e preço é {corr:.2f}: "
             f"carros mais novos tendem a custar mais, mas com muita variação.")
    return {"arquivo": arquivo, "titulo": "Ano de fabricação x preço de venda", "texto": texto}


def gerar_graficos(df, pasta):
    # Gera os 3 gráficos e devolve a lista para a página de análise
    return [
        grafico_distribuicao(df, pasta),
        grafico_combustivel(df, pasta),
        grafico_ano_preco(df, pasta),
    ]
