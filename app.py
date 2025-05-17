import streamlit as st
import yfinance as yf
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ----- PRIMER PASO: INTERFAZ -----
st.set_page_config(page_title="Análisis Proyecto de Vivienda", layout="wide")
st.title("FinSight")
st.subheader("Visualiza, proyecta y evalúa la viabilidad financiera de tu desarrollo inmobiliario.")

# ----- SEGUNDO PASO: FUNCIÓN DE CÁLCULO DE RATIOS -----
def calcular_indicadores_resumen(df): #se converte en formato largo 
    tabla = df.pivot_table(index=['PROYECTO', 'CIUDAD'], 
                           columns='METRICAS', 
                           values='VALOR', 
                           aggfunc='sum').reset_index()

    for col in tabla.columns:
        if col not in ['PROYECTO', 'CIUDAD']:
            tabla[col] = pd.to_numeric(tabla[col], errors='coerce') #se asegura de leer todo en formato numero 

    #calculo de indicadores financieros 
    tabla['GASTOS_INDIRECTOS'] = tabla[[
        'COMISIONES', 'CONSTRUCT_FEE', 'MNG_FEE', 'SALES_FEE', 'MONITOR_FEE',
        'DESING_FEE', 'ESTRUCTURACION', 'INDIRECTOS'
    ]].sum(axis=1)

    tabla['UT_BRUTA'] = tabla['VENTAS'] - tabla['COSTOS'] - tabla['LOTE']
    tabla['UT_OPERATIVA'] = tabla['UT_BRUTA'] - tabla['GASTOS_INDIRECTOS']
    tabla['EBITDA'] = tabla['UT_OPERATIVA'] - tabla['COST_DEBT']
    tabla['UT_NETA'] = tabla['EBITDA'] - tabla['TAX']
    tabla['MARGEN_NETO'] = tabla['UT_NETA'] / tabla['VENTAS']
    tabla['ROI'] = tabla['UT_NETA'] / (tabla['COSTOS'] + tabla['LOTE'] + tabla['GASTOS_INDIRECTOS'])

    resumen = tabla[[ #indicadores clave 
        'PROYECTO', 'UT_BRUTA', 'UT_OPERATIVA', 'EBITDA',
        'UT_NETA', 'MARGEN_NETO', 'ROI'
    ]]
    
    return resumen, tabla



# Subida de archivo .xlsx
uploaded_file = st.file_uploader("📂 Carga tu archivo .xlsx con la base del proyecto", type=["xlsx"])

# Si el archivo se carga, lo leemos y mostramos
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Archivo cargado con éxito.")
    
    st.subheader("Vista previa de los datos:")
    st.dataframe(df, use_container_width=True, height=500) #lista desplegable

    st.subheader("Indicadores clave del proyecto")
    resultados, tabla_completa = calcular_indicadores_resumen(df)

    # Lista desplegable para seleccionar el proyecto
    proyectos_disponibles = resultados['PROYECTO'].unique()
    proyecto_seleccionado = st.selectbox("🔽 Selecciona el proyecto que deseas analizar", proyectos_disponibles)

    # Filtrar solo la fila correspondiente al proyecto seleccionado
    fila = resultados[resultados['PROYECTO'] == proyecto_seleccionado].iloc[0]

    # Formatear valores
    ut_neta = f"${fila['UT_NETA']:,.0f}".replace(",", ".")
    ebitda = f"${fila['EBITDA']:,.0f}".replace(",", ".")
    ut_operativa = f"${fila['UT_OPERATIVA']:,.0f}".replace(",", ".")
    ut_bruta = f"${fila['UT_BRUTA']:,.0f}".replace(",", ".")
    margen = f"{fila['MARGEN_NETO']:.2%}"
    roi = f"{fila['ROI']:.2%}"

    # Mostrar tarjetas visuales
    st.markdown("### Indicadores Financieros Clave")

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilidad Neta", ut_neta)
    col2.metric("Margen Neto", margen)
    col3.metric("ROI", roi)

    col4, col5 = st.columns(2)
    col4.metric("Utilidad Operativa", ut_operativa)
    col5.metric("EBITDA", ebitda)

    st.markdown("### Simulación de Escenarios Financieros")

    # Extraer los datos del proyecto seleccionado desde la tabla completa
    base = tabla_completa[tabla_completa["PROYECTO"] == proyecto_seleccionado].iloc[0]

    # Inicializar valores en session_state si no existen
    for key in ["ventas_adj", "costos_adj", "indirectos_adj"]:
        if key not in st.session_state:
            st.session_state[key] = 0

    # Botón para restablecer sliders
    if st.button("🔄 Restablecer valores"):
        st.session_state.ventas_adj = 0
        st.session_state.costos_adj = 0
        st.session_state.indirectos_adj = 0
        st.rerun()

    # Sliders que usan el valor actual del session_state
    ventas_adj = st.slider("🔼 Aumento % en Ventas", -30, 50, key="ventas_adj")
    costos_adj = st.slider("🔽 Variación % en Costos", -50, 30, key="costos_adj")
    indirectos_adj = st.slider("🔽 Variación % en Gastos Indirectos", -50, 30, key="indirectos_adj")

    # Aplicar ajustes
    ventas_sim = base['VENTAS'] * (1 + ventas_adj / 100)
    costos_sim = base['COSTOS'] * (1 + costos_adj / 100)
    indirectos_sim = base['GASTOS_INDIRECTOS'] * (1 + indirectos_adj / 100)
    lote_sim = base['LOTE']
    cost_debt_sim = base['COST_DEBT']
    tax_sim = base['TAX']

    # Recalcular indicadores simulados
    net_profit_sim = ventas_sim - costos_sim - lote_sim - indirectos_sim - cost_debt_sim - tax_sim
    roi_sim = net_profit_sim / (costos_sim + lote_sim + indirectos_sim)
    margen_sim = net_profit_sim / ventas_sim

    # Mostrar tarjetas simuladas
    st.markdown("### Resultado del Escenario Simulado")

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilidad Neta Simulada", f"${net_profit_sim:,.0f}".replace(",", "."))
    col2.metric("ROI Simulado", f"{roi_sim:.2%}")
    col3.metric("Margen Neto Simulado", f"{margen_sim:.2%}")


    st.markdown("## Predicción de Viabilidad con Random Forest")

    try:
        # Validar que existan las columnas necesarias
        columnas_requeridas = ["PROYECTO", "METRICAS", "VALOR"]
        if not all(col in df.columns for col in columnas_requeridas):
            st.error("❌ El archivo debe contener las columnas: PROYECTO, METRICAS, VALOR.")
            st.stop()

        # Convertir a formato ancho
        df_wide = df.pivot_table(index="PROYECTO", columns="METRICAS", values="VALOR", aggfunc="first").reset_index()

        # Seleccionar solo la fila del proyecto que el usuario escogió
        base = df_wide[df_wide["PROYECTO"] == proyecto_seleccionado].iloc[0]

        # === PREDECIR CON RANDOM FOREST ===

        # Variables usadas por el modelo
        columnas_modelo = [
            'COMISIONES', 'CONSTRUCT_FEE', 'COSTOS', 'COST_DEBT', 'DESING_FEE',
            'ESTRUCTURACION', 'INDIRECTOS', 'LOTE', 'M2_CONSTRUIDOS',
            'M2_VENDIBLES', 'M2_VENTA', 'MNG_FEE', 'MONITOR_FEE', 'SALES_FEE',
            'TASA', 'TAX', 'VALOR_CREDITO', 'VENTAS', 'INFLACION', 'DESEMPLEO',
            'PIB_SECTORIAL'
        ]

        # Construir input para el modelo
        X_proyecto = pd.DataFrame([{col: base[col] for col in columnas_modelo}])

        # Cargar modelo y escalador usado en el entrenamiento
        import joblib
        modelo_rf = joblib.load("modelo_rf.pkl")
        scaler = joblib.load("scaler.pkl")

        X_scaled = scaler.transform(X_proyecto)
        resultado = modelo_rf.predict(X_scaled).item() #resultado binario (objetivo)
        probabilidad = modelo_rf.predict_proba(X_scaled)[:, 1].item() #resultado de probabilidad de viabilidad 

        # === IMPORTANCIA DE VARIABLES ===

        # Obtener las importancias de cada variable
        importancias = modelo_rf.feature_importances_
        variables = columnas_modelo

        # Crear un DataFrame ordenado
        df_importancia = pd.DataFrame({
            'Variable': variables,
            'Importancia': importancias
        }).sort_values(by='Importancia', ascending=True)

        # Mostrar como gráfico horizontal de barras
        st.markdown("### Importancia de Variables en la Predicción")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_importancia['Variable'], df_importancia['Importancia'])
        ax.set_xlabel("Importancia")
        ax.set_ylabel("Variable")
        ax.set_title("Importancia relativa de cada variable en el modelo")
        st.pyplot(fig)

        # Mostrar resultado
        if resultado == 1:
            st.success(f"✅ El modelo predice que el proyecto es **VIABLE** (probabilidad: {probabilidad:.2%})")
        else:
            st.warning(f"❌ El modelo predice que el proyecto **NO es viable** (probabilidad: {probabilidad:.2%})")

        st.progress(probabilidad)

    except Exception as e:
        st.error(f"❌ Error al convertir y predecir el proyecto: {e}")
    
    st.markdown("## Simulación Monte Carlo de Ventas, Costos y Utilidad (2025–2029)")

    # Sliders de variables macroeconómicas
    col1, col2, col3 = st.columns(3)
    media_inflacion = col1.slider("Media Inflación (%)", 0.0, 0.15, 0.05, 0.005)
    media_tasa = col2.slider("Media Tasa de Interés (%)", 0.0, 0.20, 0.08, 0.005)
    media_pib = col3.slider("Media PIB (%)", -0.05, 0.10, 0.03, 0.005)

    # Parámetros fijos (puedes hacerlos sliders también)
    std_inflacion = 0.01
    std_tasa = 0.015
    std_pib = 0.01

    # Valores base
    años = list(range(2025, 2030))
    ventas_base = 100_000_000_000
    costos_base = 60_000_000_000
    n_simulaciones = 1000

    # Matrices para almacenar resultados
    ventas_simuladas = np.zeros((n_simulaciones, len(años)))
    costos_simuladas = np.zeros((n_simulaciones, len(años)))
    utilidad_simuladas = np.zeros((n_simulaciones, len(años)))

    # Simular
    for i in range(n_simulaciones):
        ventas_anio = ventas_base
        costos_anio = costos_base
        for t in range(len(años)):
            inflacion = np.random.normal(media_inflacion, std_inflacion)
            tasa_interes = np.random.normal(media_tasa, std_tasa)
            pib = np.random.normal(media_pib, std_pib)

            ventas_anio *= (1 + (-1 * tasa_interes) + (1.5 * pib))
            costos_anio *= (1 + 0.8 * inflacion)

            ventas_simuladas[i, t] = ventas_anio
            costos_simuladas[i, t] = costos_anio
            utilidad_simuladas[i, t] = ventas_anio - costos_anio

    # Promedios por año
    ventas_prom = ventas_simuladas.mean(axis=0)
    costos_prom = costos_simuladas.mean(axis=0)
    utilidad_prom = utilidad_simuladas.mean(axis=0)

    # Mostrar gráfico
    st.subheader("Proyecciones Promedio con Bandas de Escenario")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(años, ventas_prom / 1e9, label='Ventas promedio', color='green', marker='o')
    ax.plot(años, costos_prom / 1e9, label='Costos promedio', color='red', marker='o')
    ax.plot(años, utilidad_prom / 1e9, label='Utilidad promedio', color='blue', marker='o')
    ax.fill_between(años,
                    utilidad_simuladas.min(axis=0) / 1e9,
                    utilidad_simuladas.max(axis=0) / 1e9,
                    color='blue', alpha=0.1, label='Rango de utilidad')
    ax.set_title("Monte Carlo: Ventas, Costos y Utilidad (Miles de millones COP)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Miles de millones")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("## 🌍 Panorama Internacional en Tiempo Real")
    st.markdown("""
    Conectamos con fuentes globales para revisar en vivo algunas variables que pueden afectar decisiones financieras relacionadas con tus proyectos:
    """)

    # Rango de fechas
    end = datetime.today()
    start = end - timedelta(days=90)

    # Descargar datos
    usd_cop = yf.download("USDCOP=X", start=start, end=end)
    tnx = yf.download("^TNX", start=start, end=end)
    tip = yf.download("TIP", start=start, end=end)

    # Mostrar métricas si hay datos
    st.subheader("- USD/COP - Tasa de cambio")
    if not usd_cop.empty and "Close" in usd_cop:
        usd_actual = float(usd_cop['Close'].dropna().iloc[-1])
        st.metric("Valor actual del dólar", f"${usd_actual:,.2f} COP")
    else:
        st.warning("No se pudo obtener el valor actual del USD/COP.")

    st.subheader("- Tasa de interés a 10 años - EE. UU.")
    if not tnx.empty and "Close" in tnx:
        tnx_actual = float(tnx['Close'].dropna().iloc[-1])
        st.metric("Tasa 10Y (yield)", f"{tnx_actual:.2f} %")
    else:
        st.warning("No se pudo obtener la tasa de interés 10Y.")

    st.subheader("- ETF TIP (Inflación protegida)")
    if not tip.empty and "Close" in tip:
        tip_actual = float(tip['Close'].dropna().iloc[-1])
        st.metric("Precio actual TIP ETF", f"${tip_actual:.2f} USD")
    else:
        st.warning("No se pudo obtener el precio del ETF TIP.")

    # Gráfico si hay datos suficientes
    if not usd_cop.empty and not tnx.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(usd_cop.index, usd_cop['Close'], label='USD/COP', color='green')
        ax.set_ylabel("COP por USD", color='green')
        ax2 = ax.twinx()
        ax2.plot(tnx.index, tnx['Close'], label='Tasa 10Y', color='blue')
        ax2.set_ylabel("Tasa %", color='blue')
        plt.title("Evolución: USD/COP vs Tasa de interés 10Y USA (últimos 90 días)")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No se pudo generar el gráfico por falta de datos.")

else:
    st.info("Por favor, carga un archivo xls para comenzar.")


