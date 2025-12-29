# predictor.py

import joblib
import pandas as pd
import unicodedata
from typing import Union, List

# Caminho do modelo
MODEL_PATH = "modelo_obesidade.pkl"

# Tradução dos rótulos do modelo para PT-BR
TRADUCAO_TARGET = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}

# Ajuste de nomes de colunas (plural → singular)
RENAME_MAP = {
    "Consumo de Alimentos Altamente Calóricos": "Consumo de Alimento Altamente Calórico",
}

# Categorias esperadas pelo modelo (em inglês)
SCHEMA_CATEGORIAS = {
    "Gênero": ["Female", "Male"],
    "Histórico Familiar": ["No", "Yes"],
    "Consumo de Alimento Altamente Calórico": ["No", "Yes"],
    "Consumo de Alimento Entre Refeições": ["No", "Sometimes", "Frequently", "Always"],
    "Fumante": ["No", "Yes"],
    "Monitoramento de Consumo de Calorias": ["No", "Yes"],
    "Consumo de Álcool": ["No", "Sometimes", "Frequently", "Always"],
    "Meio de Transporte Utilizado": ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
}

# Colunas numéricas esperadas
SCHEMA_NUMERICAS = [
    "Idade", "Altura", "Peso", "Consumo de Vegetais em Refeições Principais",
    "Número de Refeições Principais", "Consumo de Água Diário",
    "Frequência de Atividade Física", "Tempo de Uso de Dispositivos Tecnológicos", "IMC"
]

# -------------------------------
# Funções utilitárias
# -------------------------------

def carregar_modelo():
    """Carrega o modelo treinado."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")

def _strip_accents(s: str) -> str:
    """Remove acentos de uma string."""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def _norm(s: str) -> str:
    """Normaliza texto para minúsculo sem acentos."""
    return _strip_accents(s).strip().lower()

# -------------------------------
# Pré-processamento
# -------------------------------

def calcular_imc(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula IMC se não existir na entrada."""
    df = df.copy()
    if "IMC" not in df.columns:
        df["IMC"] = df.apply(
            lambda r: float(r["Peso"]) / (float(r["Altura"])**2) if r.get("Peso") and r.get("Altura") else 0.0,
            axis=1
        )
    return df

def calcular_estilo(df: pd.DataFrame) -> pd.DataFrame:
    """Classifica estilo de vida como Saudável ou Não Saudável."""
    df = df.copy()

    def avaliar(row):
        cond_atividade = float(row.get("Frequência de Atividade Física", 0)) >= 3
        cond_vegetais = float(row.get("Consumo de Vegetais em Refeições Principais", 0)) >= 3
        cond_agua = float(row.get("Consumo de Água Diário", 0)) >= 2.0

        val_calorico = _norm(row.get("Consumo de Alimento Altamente Calórico", ""))
        cond_calorico = val_calorico not in ["yes", "sim", "true", "1", "y"]

        return "Saudável" if (cond_atividade and cond_vegetais and cond_calorico and cond_agua) else "Não Saudável"

    if "Estilo de vida saudável" not in df.columns:
        df["Estilo de vida saudável"] = df.apply(avaliar, axis=1)
    return df

def _renomear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas conforme mapa definido."""
    return df.rename(columns=RENAME_MAP)

def _pt_to_en_value(col: str, val: str) -> str:
    """Converte valores PT → EN conforme esperado pelo modelo."""
    v = _norm(val)
    # Mapeamentos simples
    if col in ["Histórico Familiar", "Fumante", "Monitoramento de Consumo de Calorias", "Consumo de Alimento Altamente Calórico"]:
        return "Yes" if v in ["sim", "yes", "true", "1", "y"] else "No"
    if col == "Consumo de Alimento Entre Refeições":
        if v in ["sempre", "always"]: return "Always"
        if v in ["frequente", "frequently"]: return "Frequently"
        if v in ["as vezes", "às vezes", "sometimes"]: return "Sometimes"
        return "No"
    if col == "Consumo de Álcool":
        if v in ["sempre", "always"]: return "Always"
        if v in ["frequente", "frequently"]: return "Frequently"
        if v in ["as vezes", "às vezes", "sometimes"]: return "Sometimes"
        return "No"
    if col == "Meio de Transporte Utilizado":
        if v in ["carro", "automovel"]: return "Automobile"
        if v in ["bicicleta", "bike"]: return "Bike"
        if v in ["moto", "motocicleta"]: return "Motorbike"
        if "transporte" in v: return "Public_Transportation"
        if "pe" in v: return "Walking"
        return "Automobile"
    if col == "Gênero":
        return "Female" if "fem" in v else "Male"
    return val

def _normalizar_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza valores categóricos PT → EN e aplica categorias fixas."""
    df = df.copy()
    for col, cats in SCHEMA_CATEGORIAS.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _pt_to_en_value(col, x))
            df[col] = pd.Categorical(df[col], categories=cats)
    return df

def _aplicar_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica schema esperado pelo modelo (categorias e tipos numéricos)."""
    df = _normalizar_categoricos(df)
    for col in SCHEMA_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df

# -------------------------------
# Função principal
# -------------------------------

def prever_obesidade(df_usuario: Union[pd.DataFrame, dict, List[dict]]):
    """Recebe dados do usuário, processa e retorna predição."""
    model = carregar_modelo()

    # Converte entrada para DataFrame
    if isinstance(df_usuario, dict):
        df_usuario = pd.DataFrame([df_usuario])
    elif isinstance(df_usuario, list):
        df_usuario = pd.DataFrame(df_usuario)
    elif not isinstance(df_usuario, pd.DataFrame):
        raise ValueError("Entrada deve ser DataFrame, dict ou lista de dicts")

    # Calcula campos adicionais
    df_usuario = calcular_imc(df_usuario)
    df_usuario = calcular_estilo(df_usuario)

    # Renomeia e aplica schema
    df_proc = _renomear_colunas(df_usuario)
    df_proc = _aplicar_schema(df_proc)

    # Ordena colunas conforme modelo
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        df_proc = df_proc[feature_names]

    # Predição
    preds = model.predict(df_proc)
    preds_pt = [TRADUCAO_TARGET.get(str(p), str(p)) for p in preds]

    # Monta saída
    resultados = []
    for i in range(len(preds)):
        resultados.append({
            "pred_label_raw": str(preds[i]),
            "pred_label_pt": preds_pt[i],
            "IMC": round(df_usuario["IMC"].iloc[i], 2),
            "Estilo de vida saudável": df_usuario["Estilo de vida saudável"].iloc[i],
        })
    return resultados[0] if len(resultados) == 1 else resultados
