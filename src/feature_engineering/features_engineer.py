import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
    
def scaler(columns):
    '''Function to normalize numeric values'''
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(columns)
    return scaled_cols

def correlation(data):
    '''This function assists in selecting the columns for modeling 
    by identifying the columns that have the highest positive and
    negative correlations with the `EndDate'.'''

    output_path = './files/reports/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    corr = (data).corr()
    
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.savefig(output_path+'corr_heatmap.png')
    
    corr_results = (corr['cancel']*100).sort_values(ascending=False)
    results = pd.DataFrame(corr_results)
    results.to_csv(output_path+'corr_results.csv')
    
    selected_columns = corr_results[abs(corr_results)>17].index #Modify at criteria
    return selected_columns

def feature_engineer(data):
    data['cancel']=pd.to_numeric(data['cancel'])
    data_model=data.drop(['customer_id','begin_date','end_date'],axis=1)
    
    one_hot=pd.get_dummies(data_model[['type','payment_method','internet_service']])
    data_model = pd.concat([data_model, one_hot], axis=1).drop(columns=['type','payment_method','internet_service'])
    
    bolean_list=['type_Month-to-month','type_One year','type_Two year','payment_method_Bank transfer (automatic)','payment_method_Credit card (automatic)','payment_method_Electronic check','payment_method_Mailed check','internet_service_DSL','internet_service_Fiber optic']
    data_model[bolean_list] = data_model[bolean_list].astype(int)
    
    ## Correlation Analysis
    selected_features=correlation(data_model)
    
    output_path = './files/datasets/output/'
    data_model.to_csv(output_path + 'data_model.csv',index=False)
    
    return data_model[selected_features]