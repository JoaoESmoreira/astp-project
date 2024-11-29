import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from matplotlib import pyplot as plt

# Gerar dados de exemplo
np.random.seed(42)
n = 200  # Número de observações
time_index = pd.date_range('2022-01-01', periods=n, freq='D')

# Criar três séries temporais fictícias
ts1 = np.cumsum(np.random.randn(n)) + 10
ts2 = np.cumsum(np.random.randn(n)) + 20
ts3 = np.cumsum(np.random.randn(n)) + 30
exogenous = np.random.rand(n, 1)  # Variável exógena

data = pd.DataFrame({'ts1': ts1, 'ts2': ts2, 'ts3': ts3}, index=time_index)

# Visualizar os dados
data.plot(title="Séries Temporais")
plt.show()

# Exemplo 1: ARMAX
armax_model = ARIMA(data['ts1'], order=(1, 0, 1), exog=exogenous)
armax_fit = armax_model.fit()
print("ARMAX Summary:\n", armax_fit.summary())

# Exemplo 2: ARIMAX
arimax_model = ARIMA(data['ts1'], order=(1, 1, 1), exog=exogenous)
arimax_fit = arimax_model.fit()
print("ARIMAX Summary:\n", arimax_fit.summary())

# Exemplo 3: SARIMAX
sarimax_model = SARIMAX(data['ts1'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exogenous)
sarimax_fit = sarimax_model.fit()
print("SARIMAX Summary:\n", sarimax_fit.summary())

# Exemplo 4: VAR
var_model = VAR(data)
var_fit = var_model.fit(maxlags=15)
print("VAR Summary:\n", var_fit.summary())

# Exemplo 5: VARMAX
varmax_model = VARMAX(data, order=(1, 1), exog=exogenous)
varmax_fit = varmax_model.fit(disp=False)
print("VARMAX Summary:\n", varmax_fit.summary())

# Forecasts
forecast_steps = 10
print("ARMAX Forecast:", armax_fit.forecast(steps=forecast_steps, exog=exogenous[:forecast_steps]))
print("VAR Forecast:", var_fit.forecast(y=data.values[-var_fit.k_ar:], steps=forecast_steps))
