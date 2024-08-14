from sklearn.model_selection import train_test_split

#Funcion para agregar características
def make_features(data,max_lag,rolling_mean_size):
    data['year']=data.index.year
    data['month']=data.index.month
    data['day']=data.index.day
    data['hour']=data.index.hour
    for lag in range(1,max_lag+1):
        data[f'lag_{lag}']=data['num_orders'].shift(lag)
    data['rolling_mean']=data['num_orders'].shift().rolling(rolling_mean_size).mean()
    return data

def feature_engineer(data):
    
    data_model=make_features(data,4,12)
    
    return data_model