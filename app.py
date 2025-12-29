# app.py

import streamlit as st
import pandas as pd
from predictor import prever_obesidade

# -------------------------------
# Configuração da página
# -------------------------------
st.set_page_config(page_title="Previsão de Obesidade", layout="wide")
st.title("🔍 Previsão de Obesidade")
st.caption("Informe seus dados, descubra seu IMC e seu estilo de vida.")

st.markdown("---")

# Função para classificar IMC
def classificar_imc(imc: float) -> str:
    if imc < 18.5:
        return "Peso Insuficiente"
    elif imc < 25:
        return "Peso Normal"
    elif imc < 27:
        return "Sobrepeso Nível I"
    elif imc < 30:
        return "Sobrepeso Nível II"
    elif imc < 35:
        return "Obesidade Tipo I"
    elif imc < 40:
        return "Obesidade Tipo II"
    else:
        return "Obesidade Tipo III"

# -------------------------------
# Layout com colunas
# -------------------------------
st.subheader("📋 Preencha seus dados")

# Organizar campos em 3 colunas para reduzir rolagem
col1, col2, col3 = st.columns(3)

dados_usuario = {}

# Coluna 1 - Dados básicos
with col1:
    dados_usuario["Idade"] = st.number_input("Idade (anos)", min_value=1, max_value=120, value=30, step=1)
    dados_usuario["Altura"] = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    dados_usuario["Peso"] = st.number_input("Peso (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1)
    dados_usuario["Gênero"] = st.radio("Gênero", ["Feminino", "Masculino"])

# Coluna 2 - Hábitos alimentares
with col2:
    dados_usuario["Consumo de Vegetais em Refeições Principais"] = st.slider("Porções de vegetais/dia", 0.0, 10.0, 3.0, 0.5)
    dados_usuario["Número de Refeições Principais"] = st.slider("Refeições principais/dia", 1, 6, 3)
    dados_usuario["Consumo de Água Diário"] = st.slider("Consumo de água (litros/dia)", 0.0, 5.0, 2.0, 0.1)
    dados_usuario["Consumo de Alimento Altamente Calórico"] = st.selectbox("Alimentos calóricos?", ["Sim", "Não"])
    dados_usuario["Consumo de Alimento Entre Refeições"] = st.selectbox("Lanches entre refeições", ["Às vezes", "Frequente", "Sempre", "Não"])

# Coluna 3 - Estilo de vida
with col3:
    dados_usuario["Frequência de Atividade Física"] = st.slider("Atividade física (dias/semana)", 0, 7, 3)
    dados_usuario["Tempo de Uso de Dispositivos Tecnológicos"] = st.slider("Uso de dispositivos (horas/dia)", 0.0, 24.0, 4.0, 0.5)
    dados_usuario["Histórico Familiar"] = st.selectbox("Histórico Familiar de Obesidade", ["Sim", "Não"])
    dados_usuario["Fumante"] = st.selectbox("Fumante", ["Sim", "Não"])
    dados_usuario["Monitoramento de Consumo de Calorias"] = st.selectbox("Monitora calorias?", ["Sim", "Não"])
    dados_usuario["Consumo de Álcool"] = st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequente", "Sempre"])
    dados_usuario["Meio de Transporte Utilizado"] = st.selectbox("Meio de transporte", ["Carro", "Bicicleta", "A pé", "Transporte público", "Moto"])

st.markdown("---")

# -------------------------------
# Botão de previsão
# -------------------------------
st.subheader("🔮 Resultado da Previsão")
if st.button("Calcular Previsão"):
    try:
        df_usuario = pd.DataFrame([dados_usuario])
        resultado = prever_obesidade(df_usuario)

        # Exibir resultados com estilo
        st.success("✅ Previsão realizada com sucesso!")
        st.metric(label="IMC (calculado)", value=f"{resultado['IMC']}")
        st.write(f"**Estilo de vida:** {resultado['Estilo de vida saudável']}")

        # Classificação baseada no IMC
        imc = resultado['IMC']
        grau_imc = classificar_imc(imc)

        st.subheader("📊 Classificação pelo IMC")
        if grau_imc == "Peso Normal":
            st.success(f"✅ {grau_imc}")
        elif "Sobrepeso" in grau_imc:
            st.warning(f"⚠️ {grau_imc}")
        else:
            st.error(f"❌ {grau_imc}")

    except Exception as e:
        st.error(f"Falha na previsão: {e}")
