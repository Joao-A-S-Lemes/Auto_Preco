# Auto Preço

Aplicação web em Flask que analisa anúncios de carros usados e estima o preço de venda com Machine Learning.

Trabalho final de Tópicos Especiais em Ciência da Computação (Tema 06: preço de veículos usados — regressão).

## Pergunta do projeto

Dadas as características de um carro usado (ano, quilometragem, combustível, transmissão, tipo de vendedor e número de donos), qual é o preço de venda esperado?

## Integrantes e responsabilidades

| Integrante | Responsabilidade |
|---|---|
| Leonardo da Cruz | Desenvolvimento da aplicação web com Flask e implementação, treinamento e avaliação dos modelos de Machine Learning. |
| João Antonio | PREENCHER |
| Tiago Galhoto | PREENCHER |

## Instalação e execução

Requer Python 3.10 ou superior.

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Treinar os modelos (opcional, caso queira gerar novamente os arquivos)
python treinar.py

# Executar a aplicação
python app.py
```

A aplicação estará disponível em http://127.0.0.1:5000.

A pasta `models/` (modelo treinado e métricas) já está no repositório. O modelo foi salvo com uma versão específica do scikit-learn. Caso ocorra algum erro ao carregá-lo, execute `python treinar.py` novamente para gerar os arquivos na sua máquina.

## Páginas

- `/`: upload de CSV ou utilização do CSV de exemplo.
- `/analise`: amostra, diagnóstico, tratamento dos dados, três achados e três gráficos, todos calculados a partir do arquivo enviado.
- `/previsao`: formulário que retorna o preço estimado pelo modelo.

## Esquema do CSV

Colunas obrigatórias (outras colunas são ignoradas), separador vírgula e codificação UTF-8.

| Coluna | Tipo | Descrição |
|---|---|---|
| `year` | Inteiro | Ano de fabricação do veículo (1950 até o ano atual na etapa de limpeza). |
| `selling_price` | Número positivo | Preço anunciado/vendido, em rupias indianas (₹); variável-alvo y. |
| `km_driven` | Número não negativo | Quilometragem rodada, em quilômetros; a limpeza limita a 500.000 km. |
| `fuel` | Categoria | `Petrol`, `Diesel`, `CNG` ou `LPG`. |
| `seller_type` | Categoria | `Individual`, `Dealer` ou `Trustmark Dealer`. |
| `transmission` | Categoria | `Manual` ou `Automatic`. |
| `owner` | Categoria | `First Owner`, `Second Owner`, `Third Owner`, `Fourth & Above Owner` ou `Test Drive Car`. |

A validação rejeita arquivos vazios, contendo apenas cabeçalho, com colunas obrigatórias ausentes (inclusive por separador incorreto), texto em colunas numéricas, anos não inteiros, valores infinitos e categorias desconhecidas.

As categorias devem respeitar a grafia apresentada na tabela. Valores ausentes são aceitos na leitura e removidos durante a limpeza, com suas respectivas contagens registradas no diagnóstico.

## Dados

- **Base utilizada:** arquivo `Car details v3.csv`, identificado como dataset de veículos usados do CarDekho, incluído na pasta `data/`.
- **Moeda:** rupias indianas (₹). Os valores não devem ser interpretados como reais.
- **Limpeza:** realizada em uma cópia dos dados, sem alterar o CSV original, por meio de `dados.py`.
- São removidas duplicatas, linhas com valores ausentes, valores considerados impossíveis (preço menor ou igual a zero, quilometragem negativa e ano fora do intervalo permitido) e registros com quilometragem acima de 500.000 km.
- A justificativa e a quantidade de registros removidos em cada etapa são exibidas na página de análise.

## X e y

- **X (variáveis independentes):** `fuel`, `transmission`, `seller_type`, `owner`, `year` e `km_driven`. São informações disponíveis no momento de anunciar o veículo.
- **y (variável-alvo):** `selling_price`, correspondente ao preço de venda do veículo.

## Modelos e avaliação

O treinamento e a avaliação são realizados pelo script `treinar.py`.

- **Divisão dos dados:** `train_test_split`, com 80% dos dados para treinamento e 20% para teste, utilizando `random_state=42`.
- **Pré-processamento:** aplicação de `OneHotEncoder` nas colunas categóricas, dentro de um `Pipeline`, ajustado somente com os dados de treinamento.
- **Modelos comparados:** Linear Regression e Random Forest Regressor.
- **Métricas:** MAE (erro absoluto médio, em ₹) e R².
- O modelo com menor MAE é salvo em `models/modelo.joblib`.
- O resumo do treinamento, contendo métricas, tamanhos dos conjuntos e versão do scikit-learn, é salvo em `models/metricas.json`.
- A limpeza de `dados.py` é aplicada antes da divisão entre treino e teste, utilizando o mesmo procedimento da página de análise.

### Resultados obtidos

Resultados registrados em `models/metricas.json`, referentes ao treinamento de 12/11/2026, com 6.923 linhas após a limpeza, divisão 80%/20% e `random_state=42`.

| Modelo | MAE (₹) | R² | MAE / preço médio do teste |
|---|---:|---:|---:|
| Linear Regression | 214.958 | 0,399 | 41,9% |
| Random Forest Regressor | 185.252 | 0,491 | 36,1% |

**Modelo escolhido: Random Forest Regressor.**

O Random Forest apresentou MAE aproximadamente ₹29.706 menor que a Regressão Linear e R² aproximadamente 0,092 maior, considerando a mesma amostra de teste.

O MAE de ₹185.252 significa que, em média, a diferença absoluta entre os preços previstos e os preços observados foi de aproximadamente 185 mil rupias no conjunto de teste. Esse erro corresponde a 36,1% do preço médio desse conjunto, indicando que as previsões ainda apresentam incerteza considerável.

O R² de 0,491 indica que o modelo explica aproximadamente 49,1% da variação observada na variável-alvo, conforme essa avaliação. Esse resultado não significa que cada previsão tenha 49,1% de precisão.

## Limitações

- O modelo é treinado com a base de exemplo. O CSV enviado atualiza as análises e os gráficos, mas não realiza um novo treinamento.
- O modelo utiliza poucas variáveis: marca, modelo e potência do veículo não são considerados, fazendo com que as previsões sejam aproximadas e possam apresentar erros maiores em veículos muito diferentes.
- Os preços estão em rupias indianas e representam o mercado indiano da base histórica. Portanto, não servem como avaliação de veículos no Brasil nem como cotação atual.
- O desempenho foi medido em uma única divisão aleatória entre treino e teste. Não foi realizada validação cruzada nem busca sistemática de hiperparâmetros.
- A limpeza remove registros com dados ausentes nas colunas obrigatórias e registros considerados impossíveis ou extremos, o que pode introduzir viés na amostra analisada.
- O CSV enviado pela interface precisa seguir o esquema e as categorias documentadas. O upload atualiza a análise, mas não altera o modelo já treinado.

## Estrutura do projeto

```text
app.py              Rotas e integração da aplicação
dados.py            Leitura, validação, limpeza, achados e gráficos
modelo.py           Carregamento do modelo, validação do formulário e previsão
treinar.py          Treinamento e avaliação dos modelos
templates/          Páginas HTML
static/             CSS e gráficos gerados
static/graficos/    Gráficos recriados pela aplicação
models/             Modelo treinado e métricas
data/               CSV de exemplo
requirements.txt    Dependências do projeto
```

## Fontes e ferramentas

- **Dataset:** Vehicle Dataset from CarDekho — arquivo `Car details v3.csv`.
- **Fonte de origem:** https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- **Licença:** Database Contents License (DbCL) v1.0, em conjunto com a Open Database License (ODbL) 1.0.
- **Documento da licença:** https://opendatacommons.org/licenses/dbcl/1-0/
- **Bibliotecas utilizadas:** Flask, Pandas, NumPy, Matplotlib, scikit-learn e Joblib.
- **Apoio de IA:** Foram utilizadas IAs como Claude e ChatGPT para retirada de dúvidas, resolução de erros, sugestões de melhora e revisões realizadas no código.