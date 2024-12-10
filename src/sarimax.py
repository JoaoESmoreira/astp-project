import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics import tsaplots as tsa
from statsmodels.tsa.statespace.sarimax import SARIMAX

def calculate_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mse, rmse, mae, mape




columns_of_interest = ['temperature_mean', 'temperature_max', 'temperature_min', 'apparent_temperature_mean']  
data = pd.read_csv("../data/open_meteo_tokyo_multivariative.csv")
data = data[columns_of_interest]

data.dropna(inplace=True)

data = data.dropna()
n = len(data)
groups = data.iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
data = groups.mean()
data = data[-12 * 10:]
train = data[:-12]
test = data[-12:]


d = 1   # there is a litle trend
D = 1   # there is a seasonality
s = 12  # seasonality of periodicity
p = 5   # number of samples out in PACS
q = 3   # number of samples out in ACS
P = 2   # number of samples out in PACS
Q = 1   # number of samples out in ACS

variables = ['temperature_mean', 'temperature_max', 'temperature_min', 'apparent_temperature_mean']
_, axes = plt.subplots(len(variables), 2, figsize=(18, 12))  # Segunda coluna para autocorrelação
for i, variable in enumerate(variables):
    print(variable)
    
    # Ajustar o modelo SARIMAX
    model = SARIMAX(
        endog=train[variable],
        exog=train[variables[:i] + variables[i+1:]],  # Variáveis exógenas
        order=(p, d, q),
        seasonal_order=(P, D, Q, s)
    )
    results = model.fit(disp=False)
    
    # Previsão com intervalos de confiança
    forecast_obj = results.get_forecast(
        steps=len(test),
        exog=test[variables[:i] + variables[i+1:]],  # Variáveis exógenas
    )
    forecast_mean = forecast_obj.predicted_mean  # Previsões
    forecast_conf_int = forecast_obj.conf_int()  # Intervalos de confiança

    # Gráfico de treino, teste, previsão e intervalo de confiança
    sns.lineplot(ax=axes[i, 0], x=train.index, y=train[variable], label='Train', color='blue')
    sns.lineplot(ax=axes[i, 0], x=test.index, y=test[variable], label='Test', color='orange')
    sns.lineplot(ax=axes[i, 0], x=test.index, y=forecast_mean, label='Forecast', color='green')
    
    # Adicionando o intervalo de confiança ao gráfico
    axes[i, 0].fill_between(
        test.index, 
        forecast_conf_int.iloc[:, 0],  # Limite inferior
        forecast_conf_int.iloc[:, 1],  # Limite superior
        color='green', alpha=0.2, label='Confidence Interval'
    )
    axes[i, 0].set_xlabel('Time')
    axes[i, 0].set_ylabel('Values')
    axes[i, 0].set_title(f'{variable} - Train vs Test vs Forecast')
    axes[i, 0].legend()

    # Análise de resíduos
    train_fitted_values = results.fittedvalues  # Valores ajustados pelo modelo
    train_residuals = train[variable] - train_fitted_values  # Resíduos
    plot_acf(train_residuals, lags=12, ax=axes[i, 1])  # Lag ajustável
    axes[i, 1].set_title(f'{variable} - Residuals Autocorrelation')

plt.tight_layout()
plt.show()


# _, axes = plt.subplots(len(variables), figsize=(12, 6))
# for i in range(len(variables)):
#     print(variables[i])
# 
#     model = SARIMAX(
#         endog=train[variables[i]],  
#         exog=train[variables[:i] + variables[i+1:]],  # Variáveis exógenas
#         order=(p, d, q),  
#         seasonal_order=(P, D, Q, s)
#     )
# 
#     results = model.fit(disp=False)
#     forecast = results.predict(
#         start=len(train), 
#         end=len(train) + len(test) - 1, 
#         exog=test[variables[:i] + variables[i+1:]],  # Variáveis exógenas
#     )
#     train_fitted_values = results.fittedvalues  # Valores ajustados pelo modelo
#     train_residuals = train[variables[i]] - train_fitted_values  # Resíduos
# 
#     plot_acf(train_residuals, lags=12, ax=axes[i])  # Lag ajustável
#     axes[i].set_title(f'{variables[i]} - Residuals Autocorrelation')
# 
# 
#     # sns.lineplot(ax=axes[i], x=train.index, y=train[variables[i]], label='Train', color='blue')
#     # sns.lineplot(ax=axes[i], x=test.index,  y=test[variables[i]],  label='Test', color='orange')
#     # sns.lineplot(ax=axes[i], x=test.index,  y=forecast, label='Forecast', color='green')
#     # axes[i].set_xlabel('time')
#     # axes[i].set_ylabel('Values')
#     # axes[i].set_title(variables[i])
# 
#     # error = calculate_metrics(test['temperature_mean'], forecast)
#     # print("#################        ######################")
#     # print("Linear Model - MSE:", error[0], "RMSE:", error[1], "MAE:", error[2], "MAPE:", error[3])
#     # print("#################        ######################")
# 
# 
# 
# plt.legend()
# plt.tight_layout()
# plt.show()
