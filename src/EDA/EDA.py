import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics import tsaplots
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Ajuste del modelo ARIMA
def ARIMA_model(data_train,data_test,order):
    order = order  # Puedes definir p, d, q basado en el análisis de los gráficos ACF y PACF
    start_index = len(data_train)
    end_index = start_index + len(data_test) - 1
    model = ARIMA(data_train, order=order)
    arima_model = model.fit()
    predictions = arima_model.predict(start=start_index, end=end_index, typ='levels')
    
    # Calcular el error de pronóstico
    error = mean_squared_error(data_test, predictions)**0.5
    return error,predictions

def resultados(train,test,predictions):
    fig,ax=plt.subplots()
    ax.plot(train, label='Train')
    ax.plot(test, label='Test')
    ax.plot(predictions, label='Predictions')
    ax.legend()
    return fig

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    describe_result=data.describe()
    
    eda_path = './files/modeling_output/figures/'
    reports_path='./files/modeling_output/reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)
    
    # Exporting the file
    with open(reports_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open(reports_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
    
    fig , ax  = plt.subplots()
    ax.plot(data)
    fig.savefig(eda_path+'fig.png')
    
    #Mes
    data['month']=data.index.month
    data_1=data.groupby('month')['num_orders'].sum()
    fig_1 , ax_1  = plt.subplots()
    ax_1.plot(data_1)
    fig_1.savefig(eda_path+'fig_1.png')
    
    #Hora
    data['hour']=data.index.hour
    data_2=data.groupby('hour')['num_orders'].mean()
    fig_2 , ax_2  = plt.subplots()
    ax_2.plot(data_2)
    fig_2.savefig(eda_path+'fig_2.png')
    
    #Tendencias y estacionalidad 
    
    data_seasonal=data['num_orders']
    data_seasonal.head()
    descomposed = seasonal_decompose(data_seasonal)

    fig_3,ax_3=plt.subplots(3,1,figsize=(8, 12))

    ax=ax_3[0]
    ax.plot(descomposed.trend)
    ax.set_title('Tendencia')
    ax=ax_3[1]
    ax.plot(descomposed.seasonal)
    ax.set_title('Estacionalidad')
    ax=ax_3[2]
    ax.plot(descomposed.resid)
    ax.set_title('Residuales')

    plt.tight_layout()
    fig_3.savefig(eda_path+'fig_3.png')
    
    # Análisis de la estacionalidad para los primeros 8 días de agosto.
    
    fig_4,ax_4=plt.subplots()
    ax_4.plot(descomposed.seasonal['2018-08-01':'2018-08-08'])
    ax_4.set_title('Análisis de la estacionalidad para los primeros 8 días de agosto')
    ax_4.set_xticklabels(ax_4.get_xticklabels(), rotation=90)
    fig_4.savefig(eda_path+'fig_4.png')
    
    # Diferencias de series temporales
    
    data_shifted=data.copy()
    data_shifted-=data_shifted.shift()
    data_shifted.drop(['rolling_mean'],axis=1,inplace=True)
    data_shifted['mean'] = data['num_orders'].rolling(20).mean()
    data_shifted['std'] = data['num_orders'].rolling(20).std()
    fig_5,ax_5=plt.subplots()
    ax_5.plot(data_shifted['num_orders'],color='blue')
    ax_5.plot(data_shifted['mean'],color='red')
    ax_5.plot(data_shifted['std'],color='black')
    ax_5.set_title('Diferencias de series temporales')
    fig_5.savefig(eda_path+'fig_5.png')
    
    # Diferencias de series temporales para los primeros 8 días de agosto
    
    data_shifted=data['2018-08-01':'2018-08-08'].copy()
    data_shifted-=data_shifted.shift()
    data_shifted.drop(['rolling_mean'],axis=1,inplace=True)
    data_shifted['mean'] = data['num_orders'].rolling(20).mean()
    data_shifted['std'] = data['num_orders'].rolling(20).std()
    fig_6,ax_6=plt.subplots()
    ax_6.plot(data_shifted['num_orders'],color='blue')
    ax_6.plot(data_shifted['mean'],color='red')
    ax_6.plot(data_shifted['std'],color='black')
    ax_6.set_title('Diferencias de series temporales para los primeros 8 días de agosto')
    ax_6.set_xticklabels(ax_6.get_xticklabels(), rotation=90)
    fig_6.savefig(eda_path+'fig_6.png')
    
    # Prueba Augmented Dikey-Fuller test
    # H0 = La serie temporal es no estacionaria
    # H1 = La serie temporal es estacionaria
    
    data.drop(['rolling_mean','month','hour'],axis=1,inplace=True)
    # Realizar la prueba ADF
    result = adfuller(data)

    # Extraer los resultados
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Imprimir los resultados
    with open(reports_path+'resultados_adf.txt', 'w') as file:
        file.write(f'Estadística ADF: {adf_statistic}\n')
        file.write(f'Valor p: {p_value}\n')
        file.write('Valores críticos:\n')
        for key, value in critical_values.items():
            file.write(f'   {key}: {value}\n')

        if p_value < 0.05:
            file.write('La serie temporal es estacionaria.\n')
        else:
            file.write('La serie temporal es no estacionaria.\n')
            

    # Prueba KPSS
    kpss_stat,p_value,lags,critical_values=kpss(data)
    
    with open(reports_path+'resultados_kpss.txt','w') as file:
        file.write(f'Estadística KPSS: {kpss_stat}')
        file.write(f'Valor p: {p_value}')
        file.write('Valores críticos:')
        for key, value in critical_values.items():
            file.write(f'   {key}: {value}')
        if p_value < 0.05:
            file.write('La serie temporal es no estacionaria.')
        else:
            file.write('La serie temporal es estacionaria.')
        
    # Metodologia Box - Jenkins 
       
    data_diff=data-data.shift(1)
    
    # Realizar la prueba ADF
    result = adfuller(data_diff.dropna())

    # Extraer los resultados
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Imprimir los resultados
    with open(reports_path+'resultados_bj_adfuller.txt','w') as file:
        file.write(f'Estadística ADF: {adf_statistic}')
        file.write(f'Valor p: {p_value}')
        file.write('Valores críticos:')
        for key, value in critical_values.items():
            file.write(f'   {key}: {value}')
        if p_value < 0.05:
            file.write('La serie temporal es estacionaria.')
        else:
            file.write('La serie temporal es no estacionaria.')
            
    kpss_stat,p_value,lags,critical_values=kpss(data_diff.dropna())
    with open(reports_path+'resultados_bj_kpss.txt','w') as file:
        file.write(f'Estadística KPSS: {kpss_stat}')
        file.write(f'Valor p: {p_value}')
        file.write('Valores críticos:')
        for key, value in critical_values.items():
            file.write(f'   {key}: {value}')
        if p_value < 0.05:
            file.write('La serie temporal es no estacionaria.')
        else:
            file.write('La serie temporal es estacionaria.')
            
            
    # Análisis de autocorrelación
    
    fig7=tsaplots.plot_acf(data,lags=10)
    fig7.savefig(eda_path+'autocorrelacion.png', dpi=300, bbox_inches='tight')

    # Autocorrelación parcial
    
    fig8=tsaplots.plot_pacf(data,lags=10)
    fig8.savefig(eda_path+'partial_autocorrelacion.png', dpi=300, bbox_inches='tight')
    
    # ARIMA 
    
    seed=12345
    train_arima,test_arima=train_test_split(data_diff,
                            test_size=0.10,shuffle=False,random_state=seed)
    
    # IMA(0,1,1)
    
    order=(0,1,1)
    with open(reports_path+'IMA011_result','w') as file:
        file.write(f'Resultado: {ARIMA_model(train_arima,test_arima,order)[0]}')
    fig_9=resultados(train_arima,test_arima,ARIMA_model(train_arima,test_arima,order)[1])
    fig_9.savefig(eda_path+'IMA011_RESULT')
    
    # I(0,1,0)
    
    order=(0,1,0)
    with open(reports_path+'I010_result','w') as file:
        file.write(f'Resultado: {ARIMA_model(train_arima,test_arima,order)[0]}')
    fig_10=resultados(train_arima,test_arima,ARIMA_model(train_arima,test_arima,order)[1])
    fig_10.savefig(eda_path+'I010_RESULT')
    
    ## ARMA(1,0,1)
    
    order=(1,0,1)
    with open(reports_path+'ARMA010_result','w') as file:
        file.write(f'Resultado: {ARIMA_model(train_arima,test_arima,order)[0]}')
    fig_11=resultados(train_arima,test_arima,ARIMA_model(train_arima,test_arima,order)[1])
    fig_11.savefig(eda_path+'ARMA010_RESULT')
    
    # Auto ARIMA
    
    # Aplicar auto_arima
    modelo_auto_arima = auto_arima(data_diff.dropna(), seasonal=True, m=12, trace=True)

    # Resumen del modelo
    with open(reports_path+'autoarima_summary.txt','w') as file:
        file.write(str(modelo_auto_arima.summary()))
        
    # Ajustar el modelo ARIMA
    order = (0, 0, 1)  # ARIMA(0,0,1)
    seasonal_order = (1, 0, 2, 12)  # Estacional ARIMA(1,0,2) con periodo de estacionalidad 12
    start_index = len(train_arima)
    end_index = start_index + len(test_arima) - 1
    model = ARIMA(train_arima, order=order, seasonal_order=seasonal_order, trend='c')  # 'c' indica un término de intercepto
    results = model.fit()

    predictions = results.predict(start=start_index, end=end_index, typ='levels')

    # Calcular el error de pronóstico
    error = mean_squared_error(test_arima, predictions)**0.5
    with open(reports_path+'autoarima_error.txt','w') as file:
        file.write(f'Error: {error}')
    
    fig_12=resultados(train_arima,test_arima,predictions)
    fig_12.savefig(eda_path+'autoarima.png')