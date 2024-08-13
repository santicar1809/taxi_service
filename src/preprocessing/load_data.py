import pandas as pd

def load_datasets():
    '''This function will upload the necessary datasets
    to perform the project.'''
    data = pd.read_csv('./files/datasets/input/taxi.csv') 
    return data