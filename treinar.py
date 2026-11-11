import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Carregamento dos dados (ajuste o caminho se necessário para 'data/Car details v3.csv')
df = pd.read_csv("data/Car details v3.csv")

# Exemplo básico de limpeza (remover nulos na coluna alvo 'selling_price', se houver)
df = df.dropna(subset=["selling_price"])

# Definir features (X) e target (y)
# Certifique-se de que essas colunas existem no seu dataset 'Car details v3.csv'
features_categoricas = ["fuel", "transmission", "seller_type", "owner"]
features_numericas = [
    "year",
    "km_driven",
]  # Adicione outras numéricas se houver no seu dataset

X = df[features_categoricas + features_numericas]
y = df["selling_price"]

# 2.5 Treino e teste (Proporção 80/20 e random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Proporção de treino: {len(X_train)} amostras (80%)")
print(f"Proporção de teste: {len(X_test)} amostras (20%)")
print(f"Random State utilizado: 42\n")

# Configuração do pré-processador com ColumnTransformer
# O OneHotEncoder transforma as colunas categóricas. 'handle_unknown="ignore"' evita erros com categorias novas no teste.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            features_categoricas,
        )
    ],
    remainder="passthrough",  # Mantém as colunas numéricas intactas
)

# 2.6 Comparar dois modelos usando Pipeline
modelos = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        random_state=42
    ),  # fixando seed para reprodutibilidade
}

resultados = {}

print("--- Avaliação dos Modelos ---")
preco_medio = y_test.mean()

for nome, modelo in modelos.items():
    # Criando a Pipeline que une a transformação e o modelo
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", modelo)]
    )

    # Ajuste do encoder e do modelo estritamente nos dados de treino
    pipeline.fit(X_train, y_train)

    # Predições no conjunto de teste
    y_pred = pipeline.predict(X_test)

    # Métricas solicitadas: MAE e R²
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Porcentagem do erro em relação ao preço médio
    porcentagem_erro = (mae / preco_medio) * 100

    resultados[nome] = {"pipeline": pipeline, "mae": mae, "r2": r2}

    print(f"\nModelo: {nome}")
    print(
        f"- MAE: ₹ {mae:,.2f} (O modelo erra em média cerca de ₹ {mae:,.0f} rupias, o que representa {porcentagem_erro:.2f}% do preço médio dos carros)."
    )
    print(f"- R² (Coeficiente de Determinação): {r2:.4f}")

# Escolha final com base no menor MAE (Exemplo de lógica de salvamento)
melhor_nome = min(resultados, key=lambda k: resultados[k]["mae"])
melhor_pipeline = resultados[melhor_nome]["pipeline"]

print(
    f"\n==> Escolha Final: {melhor_nome} foi o selecionado por apresentar o menor MAE."
)

# Salvando o modelo escolhido para uso no Flask (app.py / modelo.py)
joblib.dump(melhor_pipeline, "modelo_escolhido.pkl")
print("Modelo salvo com sucesso como 'modelo_escolhido.pkl'!")