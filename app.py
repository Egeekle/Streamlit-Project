# app.py
import streamlit as st

st.title("Mi Dashboard en Streamlit")
st.write("¡Hola, este es mi primer dashboard en Streamlit!")
'''import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir los tickers
tickers = ['ORCL', 'NEM', 'NU']

# Descargar los precios históricos de cierre ajustados
data = yf.download(tickers, start="2019-01-01", end="2024-11-25")['Adj Close']

# Calcular los retornos diarios
returns = data.pct_change().dropna()

# Calcular estadísticas básicas
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Simulación de portafolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    # Generar pesos aleatorios
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Guardar los pesos
    weights_record.append(weights)
    
    # Calcular retorno esperado del portafolio
    portfolio_return = np.dot(weights, mean_returns)
    # Calcular riesgo del portafolio
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Calcular la relación Sharpe (supongamos tasa libre de riesgo = 0)
    sharpe_ratio = portfolio_return / portfolio_std_dev
    
    # Registrar resultados
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    results[2, i] = sharpe_ratio

# Convertir los resultados en un DataFrame
results_df = pd.DataFrame({
    'Return': results[0],
    'Risk': results[1],
    'Sharpe Ratio': results[2]
})

# Identificar el portafolio con la máxima relación Sharpe
max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]

# Identificar el portafolio con el mínimo riesgo
min_risk_idx = results_df['Risk'].idxmin()
min_risk_portfolio = results_df.iloc[min_risk_idx]

# Graficar la frontera eficiente
plt.figure(figsize=(10, 7))
plt.scatter(results_df['Risk'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio['Risk'], max_sharpe_portfolio['Return'], color='red', label='Máx Sharpe', edgecolors='black')
plt.scatter(min_risk_portfolio['Risk'], min_risk_portfolio['Return'], color='blue', label='Mín Riesgo', edgecolors='black')
plt.title('Frontera Eficiente')
plt.xlabel('Riesgo (Volatilidad)')
plt.ylabel('Retorno Esperado')
plt.legend()
plt.grid()
plt.show()
import plotly.express as px

# Crear un gráfico interactivo con Plotly
fig = px.scatter(
    results_df, x='Risk', y='Return', color='Sharpe Ratio',
    title='Frontera Eficiente',
    labels={'Risk': 'Riesgo (Volatilidad)', 'Return': 'Retorno Esperado'},
    hover_data={'Risk': ':.4f', 'Return': ':.4f', 'Sharpe Ratio': ':.4f'}
)

# Añadir los puntos de máximo Sharpe y mínimo riesgo
fig.add_scatter(
    x=[max_sharpe_portfolio['Risk']], y=[max_sharpe_portfolio['Return']],
    mode='markers', marker=dict(color='red', size=10, symbol='x'),
    name='Máx Sharpe'
)
fig.add_scatter(
    x=[min_risk_portfolio['Risk']], y=[min_risk_portfolio['Return']],
    mode='markers', marker=dict(color='blue', size=10, symbol='x'),
    name='Mín Riesgo'
)

fig.show()'''

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
    st.write("Risk/Return analysis is under construction.")

assets = ["NU", "ORCL", "NEM",'AAPL']

# Crear lista desplegable en la barra lateral
ticker = st.sidebar.selectbox("Selecciona un activo:", assets)
start_date = st.sidebar.date_input("Start Date:", value=datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date:", value=datetime.date(2023, 1, 1))


    
    
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
    data = yf.download(ticker, start="2019-11-25", end="2024-11-25")['Close']
    
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
    
def load_simulation_data(tickers):
    """Carga las simulaciones y calcula rendimientos esperados y volatilidades."""
    returns_data = {}
    for ticker in tickers:
        try:
            # Leer datos del archivo
            simulation_data = pd.read_csv(f"{ticker}_montecarlo_simulations.csv", index_col=0)
            simulation_data.index = pd.to_datetime(simulation_data.index)
            
            # Calcular rendimientos diarios
            returns = simulation_data.pct_change().dropna()
            returns_data[ticker] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'returns': returns
            }
        except FileNotFoundError:
            st.warning(f"Archivo de simulación para {ticker} no encontrado.")
        except Exception as e:
            st.error(f"Error al procesar {ticker}: {e}")
    return returns_data

def optimize_portfolio(returns_data):
    """Optimiza la cartera para encontrar la frontera eficiente."""
    # Extraer activos
    tickers = list(returns_data.keys())
    if not tickers:
        raise ValueError("No hay datos de retornos disponibles para optimizar la cartera.")
    
    # Matriz de rendimientos y covarianza
    returns_matrix = np.vstack([returns_data[ticker]['returns'].values.flatten() for ticker in tickers])
    cov_matrix = np.cov(returns_matrix)
    mean_returns = np.array([returns_data[ticker]['returns'].mean() for ticker in tickers])
    variances = np.diag(cov_matrix)
    cov_with_variances = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    cov_with_variances['Variance'] = variances  # Añadir la columna de varianzas
    # Función objetivo: minimizar riesgo para un retorno esperado
    def portfolio_volatility(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)

    # Definir pesos iniciales (uniformes)
    initial_weights = np.ones(len(tickers)) / len(tickers)
    bounds = tuple((0, 1) for _ in range(len(tickers)))  # Los pesos deben estar entre 0 y 1
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # La suma de los pesos debe ser 1
    )

    # Generar la frontera eficiente
    efficient_portfolios = []
    for target_return in np.linspace(mean_returns.min(), mean_returns.max(), 50):
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return}
        )
        try:
            result = minimize(portfolio_volatility, initial_weights, 
                              method='SLSQP', bounds=bounds, constraints=constraints)
        except Exception as e:
            print(f"Error en la optimización para retorno objetivo {target_return}: {e}")
            continue
        
        if not result.success:
            print(f"Optimización fallida para retorno objetivo {target_return}: {result.message}")
            continue
        
        efficient_portfolios.append({
            'Return': target_return,
            'Volatility': result.fun,
            'Weights': result.x
        })
    
    if not efficient_portfolios:
        raise ValueError("No se pudo generar una frontera eficiente.")
    
    return pd.DataFrame(efficient_portfolios), cov_with_variances



def plot_efficient_frontier(portfolio_df):
    """Visualiza la frontera eficiente."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df['Volatility'],
        y=portfolio_df['Return'],
        mode='lines+markers',
        name='Frontera Eficiente'
    ))
    fig.update_layout(
        title="Frontera Eficiente de la Cartera",
        xaxis_title="Volatilidad (Riesgo)",
        yaxis_title="Retorno Esperado",
        template="plotly_white"
    )
    st.plotly_chart(fig)
    
tickers = st.sidebar.text_area("Ingrese Tickers de activos simulados:", value="AAPL, MSFT, TSLA").split(",")
if st.sidebar.button("Optimizar Cartera"):
    returns_data = load_simulation_data(tickers)
    if returns_data:
        portfolio_df = optimize_portfolio(returns_data)
        plot_efficient_frontier(portfolio_df)
        st.write("Pesos Óptimos:", portfolio_df.iloc[-1]['Weights'])
   

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

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
    data = yf.download(ticker, start="2019-11-25", end="2024-11-25")['Close']
    
    if data.empty:
        st.error(f"No se encontraron datos para {ticker}.")
        return None, None
    
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
    
    # Combinar datos históricos con la mejor simulación
    combined_data = pd.concat([data, best_simulation])
    combined_returns = combined_data.pct_change().dropna()

    return combined_returns, combined_data

# Función para optimización de cartera eficiente
def optimize_portfolio(returns):
    num_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.02

    def portfolio_performance(weights):
        returns = np.sum(weights * mean_returns) * 252
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (returns - risk_free_rate) / risk
        return -sharpe_ratio  # Negativo para maximizar Sharpe

    # Restricciones
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(portfolio_performance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Integrar análisis completo
def analyze_portfolio(tickers, years=5, simulations=1000):
    all_returns = []
    for ticker in tickers:
        returns, _ = analyze_with_monte_carlo(ticker, years, simulations)
        if returns is not None:
            all_returns.append(returns)

    # Combinar los retornos de todos los activos
    combined_returns = pd.concat(all_returns, axis=1).dropna()
    combined_returns.columns = tickers

    # Optimización de portafolio
    optimal_weights = optimize_portfolio(combined_returns)
    st.write("Pesos del portafolio óptimo:")
    for ticker, weight in zip(tickers, optimal_weights):
        st.write(f"{ticker}: {weight:.2%}")

    # Frontera eficiente
    risks, returns, weights_list = efficient_frontier(combined_returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risks, y=returns, mode='lines', name='Frontera Eficiente'))
    st.plotly_chart(fig)

# Función para calcular la frontera eficiente
def efficient_frontier(returns, num_points=100):
    results = []
    num_assets = len(returns.columns)
    weights_list = []
    target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, num_points)

    for target in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                       {'type': 'eq', 'fun': lambda w: np.sum(w * returns.mean() * 252) - target})
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w))),
                          num_assets * [1. / num_assets],
                          bounds=bounds, constraints=constraints)
        if result.success:
            results.append((result.fun, target))
            weights_list.append(result.x)
    
    risks, returns = zip(*results)
    return risks, returns, weights_list

# Streamlit Interface
st.sidebar.header("Optimización de Portafolio")

tickers = st.sidebar.text_input("Ingrese Tickers (separados por comas):", value="AAPL,MSFT,GOOGL")
tickers_list = [ticker.strip() for ticker in tickers.split(",")]




if st.sidebar.button("Ejecutar Análisis de Portafolio"):
    analyze_portfolio(tickers_list, monte_carlo_years, monte_carlo_simulation)



