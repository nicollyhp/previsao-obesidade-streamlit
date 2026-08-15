# predictor.py

import joblib
import pandas as pd
import unicodedata
from typing import Union, List

# -------------------------------
# Caminho do modelo
# -------------------------------
MODEL_PATH = "modelo_obesidade.pkl"

# -------------------------------
# Ajuste de nomes de colunas
# -------------------------------
RENAME_MAP = {
    "Consumo de Alimentos Altamente Calóricos": "Consumo de Alimento Altamente Calórico",
}

# -------------------------------
# Categorias esperadas pelo modelo
# -------------------------------
SCHEMA_CATEGORIAS = {
    "Gênero": ["Feminino", "Masculino"],
    "Histórico Familiar": ["Não", "Sim"],
    "Consumo de Alimento Altamente Calórico": ["Não", "Sim"],
    "Consumo de Alimento Entre Refeições": [
        "Não",
        "Às vezes",
        "Frequentemente",
        "Sempre"
    ],
    "Fumante": ["Não", "Sim"],
    "Monitoramento de Consumo de Calorias": ["Não", "Sim"],
    "Consumo de Álcool": [
        "Não",
        "Às vezes",
        "Frequentemente",
        "Sempre"
    ],
    "Meio de Transporte Utilizado": [
        "Automóvel",
        "Bicicleta",
        "Moto",
        "Transporte Público",
        "Caminhando"
    ]
}

# -------------------------------
# Colunas numéricas
# -------------------------------
SCHEMA_NUMERICAS = [
    "Idade",
    "Altura",
    "Peso",
    "Consumo de Vegetais em Refeições Principais",
    "Número de Refeições Principais",
    "Consumo de Água Diário",
    "Frequência de Atividade Física",
    "Tempo de Uso de Dispositivos Tecnológicos",
    "IMC",
]

# -------------------------------
# Utilitários
# -------------------------------
def carregar_modelo():
    return joblib.load(MODEL_PATH)

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )

def _norm(s: str) -> str:
    return _strip_accents(s).strip().lower()

# -------------------------------
# Cálculos auxiliares
# -------------------------------
def calcular_imc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "IMC" not in df.columns:
        df["IMC"] = df.apply(
            lambda r: float(r["Peso"]) / (float(r["Altura"]) ** 2)
            if r.get("Peso") and r.get("Altura")
            else 0.0,
            axis=1,
        )
    return df

def calcular_estilo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def avaliar(row):
        atividade = float(row.get("Frequência de Atividade Física", 0)) >= 3
        vegetais = float(row.get("Consumo de Vegetais em Refeições Principais", 0)) >= 3
        agua = float(row.get("Consumo de Água Diário", 0)) >= 2
        calorico = _norm(row.get("Consumo de Alimento Altamente Calórico", ""))
        pouco_calorico = calorico not in ["sim", "yes", "true", "1", "y"]

        return "Saudável" if (atividade and vegetais and agua and pouco_calorico) else "Não Saudável"

    if "Estilo de vida saudável" not in df.columns:
        df["Estilo de vida saudável"] = df.apply(avaliar, axis=1)

    return df

# -------------------------------
# Normalização categórica
# -------------------------------
def _pt_to_en_value(col: str, val: str) -> str:

    v = _norm(val)

    if col in [
        "Histórico Familiar",
        "Fumante",
        "Monitoramento de Consumo de Calorias",
        "Consumo de Alimento Altamente Calórico",
    ]:
        return "Sim" if v in ["sim", "yes", "true", "1", "y"] else "Não"

    if col == "Consumo de Alimento Entre Refeições":

        if v in ["sempre", "always"]:
            return "Sempre"

        if v in ["frequente", "frequently", "frequentemente"]:
            return "Frequentemente"

        if v in ["as vezes", "às vezes", "sometimes"]:
            return "Às vezes"

        return "Não"

    if col == "Consumo de Álcool":

        if v in ["sempre", "always"]:
            return "Sempre"

        if v in ["frequente", "frequently", "frequentemente"]:
            return "Frequentemente"

        if v in ["as vezes", "às vezes", "sometimes"]:
            return "Às vezes"

        return "Não"

    if col == "Meio de Transporte Utilizado":

        if v in ["carro", "automovel", "automobile"]:
            return "Automóvel"

        if v in ["bicicleta", "bike"]:
            return "Bicicleta"

        if v in ["moto", "motocicleta", "motorbike"]:
            return "Moto"

        if "transporte" in v:
            return "Transporte Público"

        if "pe" in v:
            return "Caminhando"

        return "Automóvel"

    if col == "Gênero":

        if v in ["feminino", "female"]:
            return "Feminino"

        if v in ["masculino", "male"]:
            return "Masculino"

        return "Feminino"

    return val

def _normalizar_categoricos(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col, cats in SCHEMA_CATEGORIAS.items():

        if col in df.columns:

            df[col] = df[col].apply(
                lambda x: _pt_to_en_value(col, x)
            )

            df[col] = pd.Categorical(
                df[col],
                categories=cats
            )

    return df

    for col, cats in SCHEMA_CATEGORIAS.items():

        if col in df.columns:

            df[col] = df[col].apply(
                lambda x: _pt_to_en_value(col, x)
            )

            df[col] = df[col].astype("category")

    return df

def _aplicar_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalizar_categoricos(df)
    for col in SCHEMA_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df

# -------------------------------
# Mapeamento da saída do modelo
# -------------------------------
def mapear_predicao(score: float) -> str:

    classe = round(float(score))

    mapa = {
        0: "Obesidade Tipo I",
        1: "Obesidade Tipo II",
        2: "Obesidade Tipo III",
        3: "Peso Insuficiente",
        4: "Peso Normal",
        5: "Sobrepeso Nível I",
        6: "Sobrepeso Nível II"
    }

    return mapa.get(classe, "Desconhecido")

# -------------------------------
# Função principal
# -------------------------------
def prever_obesidade(df_usuario: Union[pd.DataFrame, dict, List[dict]]):
    model = carregar_modelo()

    if isinstance(df_usuario, dict):
        df_usuario = pd.DataFrame([df_usuario])
    elif isinstance(df_usuario, list):
        df_usuario = pd.DataFrame(df_usuario)

    df_usuario = calcular_imc(df_usuario)
    df_usuario = calcular_estilo(df_usuario)

    df_proc = df_usuario.rename(columns=RENAME_MAP)
    df_proc = _aplicar_schema(df_proc)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        df_proc = df_proc[feature_names]

    preds = model.predict(df_proc)

    resultados = []
    for i, p in enumerate(preds):
        imc_atual = round(df_usuario["IMC"].iloc[i], 2)

        label_modelo = mapear_predicao(float(p))

        # CLÍNICA + ML
        if imc_atual < 25 and "Obesidade" in label_modelo:
            label_modelo = "Sobrepeso Nível II"

        resultados.append({
            "pred_label_raw": float(p),
            "pred_label_pt": label_modelo,
            "IMC": imc_atual,
            "Estilo de vida saudável": df_usuario["Estilo de vida saudável"].iloc[i],
        })

    return resultados[0] if len(resultados) == 1 else resultados