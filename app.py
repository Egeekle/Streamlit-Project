# app.py
import streamlit as st

st.title("Analisis Tecnico de Acciones")
st.write("Este es un ejemplo de una aplicación web para análisis técnico de acciones.")

from scipy.optimize import minimize
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import plotly.express as px

def analyze_stock(ticker, start_date, end_date):
    # 1. Fetch Data
    try:
        # Descargar solo los precios de cierre
        data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
        if data.empty:
            st.error(f"No data found for {ticker} between {start_date} and {end_date}")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return data.dropna()

    # Asegúrate de que la columna 'Close' sea unidimensional
    data['Close'] = data['Close'].squeeze()

    # Verifica que haya suficientes datos
    if len(data) < 200:
        st.error(f"Not enough data to calculate all indicators for {ticker}.")
        return

    # 2. Calculate Indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['20d_MA'] = data['Close'].rolling(window=20).mean()
    data['20d_STD'] = data['Close'].rolling(window=20).std()
    data['Upper'] = data['20d_MA'] + (data['20d_STD'] * 2)
    data['Lower'] = data['20d_MA'] - (data['20d_STD'] * 2)
    data['Buy_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
    data['Sell_Signal'] = np.where(data['SMA_50'] < data['SMA_200'], 1, 0)
    
    # 3. Plotting with Plotly Express
    fig = px.line(data.reset_index(), x='Date', y=data['Close'].squeeze(), title=f"{ticker} Close Price")
    fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50')
    fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200')
    fig.add_scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50')
    fig.add_scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200')
    fig.add_scatter(x=data.index, y=data['Upper'], mode='lines', name='Upper BB')
    fig.add_scatter(x=data.index, y=data['Lower'], mode='lines', name='Lower BB')

    # Add Buy/Sell signals
    fig.add_scatter(x=data.index[data['Buy_Signal'] == 1], y=data
                    ['Close'].squeeze()[data['Buy_Signal'] == 1], mode='markers', name='Buy Signal', marker=dict(color='green', size=10))
    fig.add_scatter(x=data.index[data['Sell_Signal'] == 1], y=data['Close'].squeeze()[data['Sell_Signal'] == 1],
                    mode='markers', name='Sell Signal', marker=dict(color='red', size=10))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    fig.update_layout(legend_title="Precio de Cierre",
    width=2000,  # Ancho del gráfico
    height=600)  # Alto del gráfico)
    st.plotly_chart(fig)  # Display the Plotly chart

    # 4. Risk/Return analysis (placeholder)
    

assets = ["NU", "ORCL", "NEM",'AAPL']

# Crear lista desplegable en la barra lateral
ticker = st.sidebar.selectbox("Selecciona un activo:", assets)
start_date = st.sidebar.date_input("Start Date:", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date:", value=datetime.date(2024, 1, 1))


    
    
data = yf.download(ticker, start=start_date, end=end_date)


import streamlit as st
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
def analyze_stock(ticker, start_date, end_date):
    # 1. Fetch Data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker} between {start_date} and {end_date}")
            return
        if 'Close' not in data.columns:
            st.error(f"'Close' column not found in data for {ticker}")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Asegúrate de que la columna 'Close' sea unidimensional
    data['Close'] = data['Close'].squeeze()

    # Verifica que haya suficientes datos
    if len(data) < 200:
        st.error(f"Not enough data to calculate all indicators for {ticker}.")
        return

    # 2. Calculate Indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['20d_MA'] = data['Close'].rolling(window=20).mean()
    data['20d_STD'] = data['Close'].rolling(window=20).std()
    data['Upper'] = data['20d_MA'] + (data['20d_STD'] * 2)
    data['Lower'] = data['20d_MA'] - (data['20d_STD'] * 2)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    # 3. Add checkboxes for visibility of lines
    show_sma_50 = st.sidebar.checkbox("Show SMA 50", value=True)
    show_sma_200 = st.sidebar.checkbox("Show SMA 200", value=True)
    show_ema_50 = st.sidebar.checkbox("Show EMA 50", value=True)
    show_ema_200 = st.sidebar.checkbox("Show EMA 200", value=True)
    show_upper_bb = st.sidebar.checkbox("Show Upper Bollinger Band", value=True)
    show_lower_bb = st.sidebar.checkbox("Show Lower Bollinger Band", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    show_macd = st.sidebar.checkbox("Show MACD", value=True)
    # 4. Plotting with Plotly Express
    fig = px.line(data.reset_index(), x='Date', y=data['Close'].squeeze(), title=f"{ticker} Close Price")

    # Add lines based on checkbox selection
    if show_sma_50:
        fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50')
    if show_sma_200:
        fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200')
    if show_ema_50:
        fig.add_scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50')
    if show_ema_200:
        fig.add_scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200')
    if show_upper_bb:
        fig.add_scatter(x=data.index, y=data['Upper'], mode='lines', name='Upper BB')
    if show_lower_bb:
        fig.add_scatter(x=data.index, y=data['Lower'], mode='lines', name='Lower BB')
    if show_rsi:
        fig.add_scatter(x=data.index, y=data['RSI'].squeeze(), mode='lines', name='RSI', line=dict(color='blue'))
    
    if show_macd:
        fig.add_scatter(x=data.index, y=data['MACD'].squeeze(), mode='lines', name='MACD', line=dict(color='purple'))
        fig.add_scatter(x=data.index, y=data['MACD_Signal'].squeeze(), mode='lines', name='MACD Signal', line=dict(color='orange'))
    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Price", 
        width=800, 
        height=600
    )
    
    st.plotly_chart(fig)  # Display the Plotly chart

    # 5. Risk/Return analysis (placeholder)
    st.write("Risk/Return analysis is under construction.")


# Inputs
def analyze_returns(data, ticker):
    # Calculate Returns
    data['Arithmetic Return'] = data['Close'].pct_change()  # Rendimiento aritmético
    data['Logarithmic Return'] = np.log(data['Close'] / data['Close'].shift(1))  # Rendimiento logarítmico
    
    # Drop NaN values
    data = data.dropna()

    # Calculate Risk
    arithmetic_risk = data['Arithmetic Return'].std()
    logarithmic_risk = data['Logarithmic Return'].std()

    # Display risk
    st.write(f"**Risk Analysis for {ticker}**")
    st.write(f"- Arithmetic Return Risk: {arithmetic_risk:.4f}")
    st.write(f"- Logarithmic Return Risk: {logarithmic_risk:.4f}")

    # Plot Returns
    fig = go.Figure()

    # Add Arithmetic and Logarithmic Returns
    fig.add_trace(go.Scatter(x=data.index, y=data['Arithmetic Return'], mode='lines', name='Arithmetic Return', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Logarithmic Return'], mode='lines', name='Logarithmic Return', line=dict(color='orange')))

    # Layout
    fig.update_layout(
        title=f"{ticker} Returns and Risk",
        xaxis_title="Date",
        yaxis_title="Returns",
        legend_title="Metrics",
        width=800,
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig)

# Inputs



import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

def analyze_stock(ticker, start_date, end_date):
    # 1. Fetch Data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker} between {start_date} and {end_date}")
            return
        if 'Close' not in data.columns:
            st.error(f"'Close' column not found in data for {ticker}")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Asegúrate de que la columna 'Close' sea unidimensional
    data['Close'] = data['Close'].squeeze()

    # Verifica que haya suficientes datos
    if len(data) < 200:
        st.error(f"Not enough data to calculate all indicators for {ticker}.")
        return

    # 2. Calculate Indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['20d_MA'] = data['Close'].rolling(window=20).mean()
    data['20d_STD'] = data['Close'].rolling(window=20).std()
    data['Upper'] = data['20d_MA'] + (data['20d_STD'] * 2)
    data['Lower'] = data['20d_MA'] - (data['20d_STD'] * 2)
    data['Buy_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
    data['Sell_Signal'] = np.where(data['SMA_50'] < data['SMA_200'], 1, 0)

    # 3. Plotting with Indicators
    fig = px.line(data.reset_index(), x='Date', y=data['Close'].squeeze(), title=f"{ticker} Close Price")
    fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50')
    fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200')
    fig.add_scatter(x=data.index, y=data['Upper'], mode='lines', name='Upper BB')
    fig.add_scatter(x=data.index, y=data['Lower'], mode='lines', name='Lower BB')
    fig.add_scatter(x=data.index[data['Buy_Signal'] == 1], 
                    y=data['Close'][data['Buy_Signal'] == 1], 
                    mode='markers', name='Buy Signal', 
                    marker=dict(color='green', size=10))
    fig.add_scatter(x=data.index[data['Sell_Signal'] == 1], 
                    y=data['Close'][data['Sell_Signal'] == 1], 
                    mode='markers', name='Sell Signal', 
                    marker=dict(color='red', size=10))

    fig.update_layout(xaxis_title="Date", yaxis_title="Price", width=800, height=600)

    st.plotly_chart(fig)

    # 4. Add Returns and Risk Analysis
    analyze_returns(data, ticker)

def analyze_returns(data, ticker):
    # Calculate Returns
    data['Arithmetic Return'] = data['Close'].pct_change()  # Rendimiento aritmético
    data['Logarithmic Return'] = np.log(data['Close'] / data['Close'].shift(1))  # Rendimiento logarítmico
    
    # Drop NaN values
    data = data.dropna()

    # Calculate Risk
    arithmetic_r = data['Arithmetic Return'].mean()
    logarithmic_r = data['Logarithmic Return'].mean()
    arithmetic_risk = data['Arithmetic Return'].std()
    logarithmic_risk = data['Logarithmic Return'].std()
    # Display risk
    st.write(f"**Risk Analysis for {ticker}**")
    st.write(f"- Arithmetic Return : {arithmetic_r:.4f}")
    st.write(f"- Logarithmic Return : {logarithmic_r:.4f}")
    st.write(f"- Arithmetic Risk: {arithmetic_risk:.4f}")
    st.write(f"- Logarithmic Risk : {logarithmic_risk:.4f}")

    # Plot Returns
    fig = go.Figure()

    # Add Arithmetic and Logarithmic Returns
    fig.add_trace(go.Scatter(x=data.index, y=data['Arithmetic Return'], mode='lines', name='Arithmetic Return', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Logarithmic Return'], mode='lines', name='Logarithmic Return', line=dict(color='orange')))
    fig.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],  # Línea horizontal desde inicio a fin
        y=[arithmetic_risk, arithmetic_risk],
        mode='lines',
        name='Logarithmic Risk (+)',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[-logarithmic_risk, -logarithmic_risk],
        mode='lines',
        name='Logarithmic Risk (-)',
        line=dict(color='blue', dash='dash')
    ))
    # Layout
    fig.update_layout(
        title=f"{ticker} Returns and Risk",
        xaxis_title="Date",
        yaxis_title="Returns",
        legend_title="Metrics",
        width=800,
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig)


if st.sidebar.button("Analyze"):  # Button to trigger analysis
    analyze_stock(ticker, start_date, end_date)

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os
# Función para realizar la simulación Monte Carlo
def monte_carlo_simulation(initial_price, mu, sigma, days, simulations):
    dt = 1
    price_matrix = np.zeros((days, simulations))
    price_matrix[0] = initial_price
    for t in range(1, days):
        random_shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
        price_matrix[t] = price_matrix[t - 1] * np.exp(random_shocks)
    return price_matrix

# Función para analizar con Monte Carlo
def analyze_with_monte_carlo(ticker, years=5, simulations=1000):
    # Descargar datos históricos del activo
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    
    if data.empty:
        st.error(f"No se encontraron datos para {ticker}.")
        return
    
    # Calcular rendimientos diarios
    returns = data.pct_change().dropna()
    
    # Parámetros estadísticos
    last_price = float(data.iloc[-1])  # Convertir a escalar
    mu = returns.mean()
    sigma = returns.std()
    days = years * 252  # Aproximación a días hábiles

    # Simulación Monte Carlo
    simulated_prices = monte_carlo_simulation(last_price, mu, sigma, days, simulations)
    
    # Generar fechas de simulación
    simulation_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    
    # Convertir simulaciones a DataFrame
    simulation_df = pd.DataFrame(simulated_prices, index=simulation_dates)
    simulation_df.columns = [f"Simulacion{i + 1}" for i in range(simulations)]
    
    # Identificar la mejor simulación
    final_prices = simulation_df.iloc[-1]
    best_simulation_idx = final_prices.idxmax()  # Índice de la mejor simulación
    best_simulation = simulation_df[best_simulation_idx]  # Datos de la mejor simulación
    
    best_simulation.to_csv(f"{ticker}_montecarlo_simulations.csv")
    # Guardar los resultados si se desea
    output_folder = "data"
    output_file = f"{ticker}_montecarlo_simulations.csv"
    output_path = os.path.join(output_folder, output_file)
    # Visualizar simulaciones con Plotly
    fig = go.Figure()
    for i in range(10):  # Mostrar 10 simulaciones
        fig.add_trace(go.Scatter(
            x=simulation_dates,
            y=simulation_df.iloc[:, i],
            mode='lines',
            name=f"Simulacion {i + 1}",
            line=dict(width=1, dash='dash')
        ))

    # Agregar el precio actual al gráfico
    fig.add_trace(go.Scatter(
        x=[data.index[-1]],
        y=[last_price],
        mode='markers+text',
        name='Precio Actual',
        marker=dict(color='red', size=10),
        text=[f"{last_price:.2f}"],
        textposition='top center'
    ))

    # Layout del gráfico
    fig.update_layout(
        title=f"Simulaciones de Monte Carlo: {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio Simulado",
        template="plotly_white",
        showlegend=True,
        width=900,
        height=600
    )

    st.plotly_chart(fig)


# Integrar al proyecto Streamlit
st.sidebar.header("Simulación Monte Carlo")

monte_carlo_ticker = st.sidebar.text_input("Ingrese Ticker:", value="AAPL")
monte_carlo_years = st.sidebar.slider("Años Simulados:", 1, 10, 5)
monte_carlo_simulations = st.sidebar.slider("Número de Simulaciones:", 100, 5000, 1000)

if st.sidebar.button("Ejecutar Monte Carlo"):
    analyze_with_monte_carlo(monte_carlo_ticker, monte_carlo_years, monte_carlo_simulations)
    
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# 1. Función para cargar datos desde Yahoo Finance
def get_stock_data(tickers, start_date, end_date):
    """Descarga datos de precios ajustados y calcula rendimientos diarios."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        returns = data.pct_change().dropna()  # Calcular rendimientos diarios
        return returns
    except Exception as e:
        st.error(f"Error al descargar datos de Yahoo Finance: {e}")
        return None


# 2. Función para optimizar la cartera
def optimize_portfolio(returns, risk_free_rate=0.02, num_portfolios=10000):
    """Genera múltiples carteras y encuentra la frontera eficiente."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    results = np.zeros((3 + num_assets, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    results_df = pd.DataFrame(
        results.T, columns=["Return", "Risk", "Sharpe"] + [f"Asset_{i}" for i in range(num_assets)]
    )
    max_sharpe_idx = results_df["Sharpe"].idxmax()
    min_risk_idx = results_df["Risk"].idxmin()
    max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
    min_risk_portfolio = results_df.iloc[min_risk_idx]

    return results_df, max_sharpe_portfolio, min_risk_portfolio


# 3. Función para visualizar la frontera eficiente
def plot_efficient_frontier_with_cml(results_df, max_sharpe_portfolio, min_risk_portfolio, tickers, risk_free_rate=0.02):
    """Muestra la frontera eficiente y la línea de mercado de capitales (CML)."""
    # Cálculo de la pendiente de la CML
    max_sharpe_return = max_sharpe_portfolio["Return"]
    max_sharpe_risk = max_sharpe_portfolio["Risk"]
    cml_x = np.linspace(0, max_sharpe_risk, 100)  # Valores de riesgo (eje X)
    cml_y = risk_free_rate + (max_sharpe_return - risk_free_rate) * (cml_x / max_sharpe_risk)  # Fórmula de la CML

    fig = go.Figure()

    # Frontera eficiente
    fig.add_trace(
        go.Scatter(
            x=results_df["Risk"],
            y=results_df["Return"],
            mode="markers",
            marker=dict(color=results_df["Sharpe"], colorscale="Viridis", size=5),
            name="Portafolios",
        )
    )

    # Punto de Máximo Sharpe
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_portfolio["Risk"]],
            y=[max_sharpe_portfolio["Return"]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Máximo Sharpe",
        )
    )

    # Punto de Mínimo Riesgo
    fig.add_trace(
        go.Scatter(
            x=[min_risk_portfolio["Risk"]],
            y=[min_risk_portfolio["Return"]],
            mode="markers",
            marker=dict(color="blue", size=10),
            name="Mínimo Riesgo",
        )
    )

    # Línea de Mercado de Capitales (CML)
    fig.add_trace(
        go.Scatter(
            x=cml_x,
            y=cml_y,
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Línea de Mercado de Capitales (CML)",
        )
    )

    # Configuración del gráfico
    fig.update_layout(
        title="Frontera Eficiente y Línea de Mercado de Capitales (CML)",
        xaxis_title="Riesgo (Volatilidad)",
        yaxis_title="Retorno Esperado",
        template="plotly_white",
    )
    st.plotly_chart(fig)



# 4. Interfaz de usuario con Streamlit
st.title("Optimizador de Carteras")
st.sidebar.header("Cartera Eficiente")
# Ingreso de tickers
tickers = st.sidebar.text_area("Ingrese los tickers separados por comas:", value="ORCL,NU,NEM").split(",")
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime("2023-01-01"))

# Botón para ejecutar la optimización
# Entrada para la tasa libre de riesgo
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo (%)", value=2.0) / 100

# Optimización y visualización
if st.sidebar.button("Optimizar Cartera"):
    with st.spinner("Descargando datos y optimizando la cartera..."):
        # Descargar datos
        returns = get_stock_data(tickers, start_date, end_date)

        if returns is not None:
            # Optimizar cartera
            results_df, max_sharpe, min_risk = optimize_portfolio(returns, risk_free_rate=risk_free_rate)

            # Mostrar resultados
            st.subheader("Resultados de la Optimización")

            # Portafolio de Máximo Sharpe
            st.write("Portafolio de Máximo Sharpe:")
            st.table(
                pd.DataFrame({
                    "Ticker": tickers,
                    "Peso Óptimo (%)": (max_sharpe[3:] * 100).values
                }).set_index("Ticker")
            )

            # Resumen de Retorno y Riesgo
            st.write("Resumen del Punto Eficiente:")
            resumen_punto_eficiente = pd.DataFrame({
                "Métrica": ["Retorno", "Riesgo (Volatilidad)", "Sharpe Ratio"],
                "Valor": [
                    round(max_sharpe["Return"], 4),
                    round(max_sharpe["Risk"], 4),
                    round(max_sharpe["Sharpe"], 4)
                ]
            })
            st.table(resumen_punto_eficiente)

            # Graficar Frontera Eficiente con CML
            plot_efficient_frontier_with_cml(results_df, max_sharpe, min_risk, tickers, risk_free_rate)


