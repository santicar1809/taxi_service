import pandas as pd
import os
import re
import numpy as np

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def nan_values(data):
    # Tratamiento de ausentes   
    if data.isna().sum()/data.shape[0] < 0.15:
        data.fillna(data.mean())
    elif data.isna().sum()/data.shape[0] < 0.05:
        data.dropna()
    return data

def duplicated_values(data):
    # Tratamiento de duplicados
    if data.duplicated().sum() > 0:
            data.drop_duplicates()
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID''' 
        
    # Pasamos primera columna como el indice para setear la serie de tiempo
    data['datetime']=pd.to_datetime(data['datetime'])
    data.set_index('datetime',inplace=True)
    data.sort_index(inplace=True)
    data.head(10)
    
    # Pasamos columnas a formato snake_case
    data = columns_transformer(data)
    
    # Ausentes
    
    data=data.apply(nan_values)
    
    # Duplicados
    
    data=data.apply(duplicated_values)
    
    path = './files/datasets/intermediate/'

    # Remuestreo
    
    data=data.resample('1H').sum()
    data['rolling_mean']=data.rolling(12).mean()

    if not os.path.exists(path):
        os.makedirs(path)

    data.to_csv(path+'preprocessed_data.csv', index=False)
    return data