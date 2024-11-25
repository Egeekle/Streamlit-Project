# app.py
import streamlit as st

st.title("Mi Dashboard en Streamlit")
st.write("¡Hola, este es mi primer dashboard en Streamlit!")
import yfinance as yf
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

fig.show()