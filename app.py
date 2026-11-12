import os
import time

from flask import Flask, redirect, render_template, request, url_for

import dados
import modelo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # uploads de até 10 MB

PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_GRAFICOS = os.path.join(PASTA_BASE, "static", "graficos")
CSV_EXEMPLO = os.path.join(PASTA_BASE, "data", "Car details v3.csv")
MIN_LINHAS = 30   # abaixo disso as análises não fazem sentido


@app.template_filter("rs")
def formatar_rs(valor):
    # 639765.4 -> "639.765" (formato brasileiro, só para exibir)
    return f"{valor:,.0f}".replace(",", ".")


@app.route("/")
def index():
    return render_template(
        "index.html",
        colunas=dados.COLUNAS_OBRIGATORIAS,
        numericas=dados.COLUNAS_NUMERICAS,
    )


@app.route("/analise", methods=["GET", "POST"])
def analise():
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        if request.form.get("usar_exemplo"):
            origem = "CSV de exemplo (Car details v3)"
            df = dados.ler_csv(CSV_EXEMPLO)
        else:
            arquivo = request.files.get("arquivo")
            if arquivo is None or arquivo.filename == "":
                raise ValueError("Escolha um arquivo CSV antes de enviar.")
            if not arquivo.filename.lower().endswith(".csv"):
                raise ValueError("O arquivo precisa ter a extensão .csv.")
            origem = arquivo.filename
            df = dados.ler_csv(arquivo)

        diagnostico = dados.diagnosticar(df)             # antes da limpeza
        amostra_original = df.head(10)
        df_limpo, registro, tamanhos = dados.limpar(df)  # df original não é alterado

        if len(df_limpo) < MIN_LINHAS:
            raise ValueError(
                f"Depois da limpeza sobraram só {len(df_limpo)} linhas; "
                f"são necessárias pelo menos {MIN_LINHAS} para a análise."
            )

        lista_achados = dados.achados(df_limpo)
        graficos = dados.gerar_graficos(df_limpo, PASTA_GRAFICOS)
    except ValueError as erro:
        return render_template(
            "index.html",
            erro=str(erro),
            colunas=dados.COLUNAS_OBRIGATORIAS,
            numericas=dados.COLUNAS_NUMERICAS,
        ), 400
    except Exception:
        app.logger.exception("Falha ao processar o arquivo")
        return render_template(
            "index.html",
            erro="Não foi possível processar o arquivo. Confira se ele segue o formato esperado.",
            colunas=dados.COLUNAS_OBRIGATORIAS,
            numericas=dados.COLUNAS_NUMERICAS,
        ), 400

    return render_template(
        "analise.html",
        origem=origem,
        diagnostico=diagnostico,
        tamanhos=tamanhos,
        registro=registro,
        amostra=amostra_original.to_html(classes="tabela", index=False, border=0),
        achados=lista_achados,
        graficos=graficos,
        versao=int(time.time()),   # evita o navegador mostrar gráfico antigo em cache
    )


@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    contexto = {
        "opcoes": modelo.OPCOES,
        "ano_min": modelo.ANO_MINIMO,
        "ano_max": modelo.ANO_MAXIMO,
        "km_max": modelo.KM_MAXIMO,
        "valores": {},
        "resultado": None,
        "erro": None,
        "nome_modelo": None,
        "metricas": modelo.carregar_metricas(),   # None se models/metricas.json não existir
    }

    try:
        contexto["nome_modelo"] = modelo.nome_do_modelo()
    except FileNotFoundError as erro:
        contexto["erro"] = str(erro)
        return render_template("previsao.html", **contexto), 500

    if request.method == "POST":
        contexto["valores"] = request.form.to_dict()
        try:
            entrada = modelo.validar_entrada(request.form)
            contexto["resultado"] = modelo.prever(entrada)
        except ValueError as erro:
            contexto["erro"] = str(erro)
            return render_template("previsao.html", **contexto), 400

    return render_template("previsao.html", **contexto)


@app.errorhandler(413)
def arquivo_grande(_erro):
    return render_template(
        "index.html",
        erro="O arquivo é grande demais (limite de 10 MB).",
        colunas=dados.COLUNAS_OBRIGATORIAS,
        numericas=dados.COLUNAS_NUMERICAS,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
