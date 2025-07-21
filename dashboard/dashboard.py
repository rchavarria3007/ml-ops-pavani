import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Preditivo de Attrition",
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# FUNÇÕES PARA CARREGAR OS ARTEFATOS (com cache para performance)
# ==============================================================================
@st.cache_resource
def load_model():
    return joblib.load('modelo_campeao_xgb.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('dataset_dashboard.csv')

@st.cache_data
def load_test_data():
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    return X_test, y_test

@st.cache_resource
def load_shap_values():
    return joblib.load('shap_values.pkl')

# ==================== NOVA FUNÇÃO ADICIONADA ====================
@st.cache_data
def get_fx_min_max(_shap_values):
    """Calcula os valores mínimo e máximo de f(x) no conjunto de teste."""
    # f(x) = base_value + soma dos shap_values para cada instância
    fx_values = _shap_values.base_values + _shap_values.values.sum(axis=1)
    return fx_values.min(), fx_values.max()
# ================================================================

# Carregar todos os artefatos
try:
    modelo = load_model()
    df = load_data()
    X_test, y_test = load_test_data()
    shap_values = load_shap_values()
    # ==================== CHAMADA DA NOVA FUNÇÃO ====================
    fx_min, fx_max = get_fx_min_max(shap_values)
    # ==============================================================
except FileNotFoundError:
    st.error("Erro: Arquivos de modelo, dados ou SHAP não encontrados.")
    st.info("Certifique-se de que todos os 7 arquivos (.pkl e .csv) estão na mesma pasta que o dashboard.py.")
    st.stop()

# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================
#st.sidebar.image("https://i.imgur.com/g030b0G.png", width=150)
st.sidebar.title("Navegação")
st.sidebar.markdown("Use o menu abaixo para explorar as diferentes análises.")

pagina_selecionada = st.sidebar.radio("Selecione uma Análise", 
    ["Visão Geral e KPIs", "Análise do Modelo Preditivo", "Análise de Cenários (Threshold)", "Análise de Risco Individual"]
)

st.sidebar.title("Filtros Gerais")
depto_selecionado = st.sidebar.selectbox(
    "Filtre por Departamento",
    options=['Todos'] + sorted(df['Department'].unique().tolist())
)

if depto_selecionado != 'Todos':
    df_filtrado = df[df['Department'] == depto_selecionado].copy()
else:
    df_filtrado = df.copy()

# ==============================================================================
# PÁGINA 1: VISÃO GERAL E KPIS
# ==============================================================================
if pagina_selecionada == "Visão Geral e KPIs":
    st.title("📊 Visão Geral do Attrition")
    st.markdown(f"Analisando o departamento: **{depto_selecionado}**")

    col1, col2, col3 = st.columns(3)
    total_attrition = df_filtrado[df_filtrado['Attrition'] == 'Yes'].shape[0]
    taxa_attrition = (total_attrition / df_filtrado.shape[0]) * 100 if df_filtrado.shape[0] > 0 else 0
    col1.metric("Total de Funcionários", df_filtrado.shape[0])
    col2.metric("Total de Casos de Attrition", total_attrition)
    col3.metric("Taxa de Attrition (%)", f"{taxa_attrition:.2f}%")
    
    st.header("Distribuição de Attrition")
    if not df_filtrado.empty:
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            fig_job_role = px.histogram(df_filtrado, x='JobRole', color='Attrition', barmode='group', title="Attrition por Cargo").update_xaxes(categoryorder="total descending")
            st.plotly_chart(fig_job_role, use_container_width=True)
        with col_graf2:
            fig_marital = px.pie(df_filtrado[df_filtrado['Attrition']=='Yes'], names='MaritalStatus', title="Perfil de Attrition por Estado Civil")
            st.plotly_chart(fig_marital, use_container_width=True)
            
    st.header("Dados Detalhados")
    st.dataframe(df_filtrado)

# ==============================================================================
# PÁGINA 2: ANÁLISE DO MODELO PREDITIVO
# ==============================================================================
elif pagina_selecionada == "Análise do Modelo Preditivo":
    st.title("🔍 Análise do Modelo Preditivo (XGBoost Otimizado)")

    st.header("Importância das Features (Feature Importance)")
    st.markdown("Quais variáveis o modelo considera mais importantes para prever o turnover?")
    feature_importance = pd.DataFrame({'feature': X_test.columns, 'importance': modelo.feature_importances_}).sort_values('importance', ascending=False)
    fig_importance = px.bar(feature_importance.head(15), x='importance', y='feature', orientation='h', title="Top 15 Features Mais Importantes")
    st.plotly_chart(fig_importance, use_container_width=True)

    st.header("Interpretabilidade com SHAP")
    st.markdown("Como cada variável impacta a decisão do modelo? O gráfico abaixo mostra o impacto médio de cada feature.")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    st.header("Matriz de Confusão")
    st.markdown("Como o modelo performou no conjunto de teste? A matriz mostra os acertos e erros.")
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou', 'Saiu'])
    fig, ax = plt.subplots(figsize=(6, 4))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    st.pyplot(fig)

# ==============================================================================
# PÁGINA 3: ANÁLISE DE CENÁRIOS (THRESHOLD)
# ==============================================================================
elif pagina_selecionada == "Análise de Cenários (Threshold)":
    st.title("⚙️ Simulador de Threshold de Decisão")
    st.markdown("O modelo gera uma probabilidade de saída para cada funcionário. O 'threshold' é o ponto de corte (padrão 0.5) para classificar alguém como 'alto risco'. Ajuste o threshold para ver o impacto no número de funcionários sinalizados e na performance do modelo.")
    
    y_probs = modelo.predict_proba(X_test)[:, 1]

    threshold = st.slider(
        "Selecione o Threshold de Risco",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    y_pred_custom = (y_probs >= threshold).astype(int)

    st.header(f"Resultados com Threshold de {threshold:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Funcionários Sinalizados", f"{sum(y_pred_custom)}")
    col2.metric("Recall", f"{recall_score(y_test, y_pred_custom):.2f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred_custom, zero_division=0):.2f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred_custom, zero_division=0):.2f}")

    st.markdown("---")
    st.markdown("**Recall:** Dos funcionários que **realmente saíram**, qual porcentagem o modelo conseguiu capturar?")
    st.markdown("**Precision:** Dos funcionários que o modelo **sinalizou como risco**, qual porcentagem realmente saiu?")

# ==============================================================================
# PÁGINA 4: ANÁLISE DE RISCO INDIVIDUAL
# ==============================================================================
elif pagina_selecionada == "Análise de Risco Individual":
    st.title("👤 Análise de Risco Individual")
    st.markdown("Selecione um funcionário (pelo seu ID) para ver a previsão do modelo e a explicação detalhada do porquê dessa previsão.")

    employee_numbers_disponiveis = df.loc[X_test.index, 'EmployeeNumber']

    id_selecionado = st.selectbox(
        "Selecione o EmployeeNumber do Funcionário",
        options=employee_numbers_disponiveis.values
    )

    if id_selecionado is not None:
        idx_selecionado = employee_numbers_disponiveis[employee_numbers_disponiveis == id_selecionado].index[0]

        st.header(f"Analisando Funcionário (ID: {id_selecionado})")

        probabilidade = modelo.predict_proba(X_test.loc[[idx_selecionado]])[0, 1]
        
        st.metric("Probabilidade de Saída", f"{probabilidade:.2%}")
        
        st.markdown("---")
        st.subheader("Explicação da Previsão (SHAP Waterfall Plot)")
        st.markdown("O gráfico abaixo mostra como cada característica do funcionário contribuiu para a pontuação final de risco.")
        
        shap_index = X_test.index.get_loc(idx_selecionado)
        
        shap_explanation_individual = shap.Explanation(
            values=shap_values.values[shap_index],
            base_values=shap_values.base_values[shap_index],
            data=X_test.loc[idx_selecionado],
            feature_names=X_test.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation_individual, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # ==================== LEGENDA ATUALIZADA ====================
        with st.expander("Como interpretar este gráfico?"):
            st.markdown(f"""
            Este gráfico de cascata (Waterfall Plot) mostra como a previsão de risco é construída passo a passo:

            - **E[f(x)] (Valor Base):** É o risco médio de saída para todos os funcionários. A barra cinza na parte inferior representa esse valor.
            
            - **Barras Vermelhas:** Representam as características deste funcionário que **aumentam** o risco de saída. Elas "empurram" o valor para cima.
            
            - **Barras Azuis:** Representam as características que **diminuem** o risco (fatores de proteção). Elas "empurram" o valor para baixo.
            
            - **f(x) (Valor Final):** É a pontuação final de risco para este funcionário. Para referência, no nosso conjunto de teste, as pontuações variaram de **{fx_min:.2f}** (risco mais baixo) a **{fx_max:.2f}** (risco mais alto).
            """)
        # ==========================================================