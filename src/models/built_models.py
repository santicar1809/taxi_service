import pandas as pd 
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,roc_auc_score
from src.models.hyper_parameters import all_models
from sklearn.impute import SimpleImputer
import joblib

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    results = []

    # Iterating the models
    models_name = ['lr','xg','lgbm','rf','cat','dt']
    for model,i in zip(models,models_name):
        best_estimator, best_score, val_score,predictions,target= model_structure(data, model[1], model[2])
        results.append([model[0],best_estimator,best_score, val_score])
        # Grafico de resultados
        eda_path = './files/modeling_output/figures/'
        pred=pd.DataFrame(predictions,index=target.index)
        
        fig,ax=plt.subplots()
        ax.plot(target,color='red')
        ax.plot(pred,color='black')
        ax.set_title('Resultados')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig.savefig(eda_path+f'resultados{i}.png')
        
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{i}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    
    tf_results = tens_flow(data) 
    #joblib.dump(tf_results[1],output_path +f'best_random_nn.joblib')
    tf_results[1].save(output_path+'best_random_nn.h5')
    # Concatening logistic models and neuronal network
    final_rev = pd.concat([results_df,tf_results[0]])
    final_rev.to_csv('./files/modeling_output/reports/model_report.csv',index=False)

    return final_rev[['model','validation_score']]


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    train,test=train_test_split(data,
                                test_size=0.10,shuffle=False,random_state=seed)
    train.dropna(inplace=True)
    features_train=train.drop(['num_orders'],axis=1)
    features_valid=test.drop(['num_orders'],axis=1)
    target_train=train['num_orders']
    target_valid=test['num_orders']    
    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    rmse_val = eval_model(best_estimator,features_valid,target_valid)[0]
    predictions=eval_model(best_estimator,features_valid,target_valid)[1]
    print(f'RMSE: {rmse_val}')
    
    results = best_estimator, best_score,rmse_val,predictions,target_valid 
    return results
    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    random_rmse=mean_squared_error(target_valid,random_prediction)**0.5
    print("RMSE:",random_rmse)
    return random_rmse,random_prediction
## Network Model Structure

def build_model(data):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # Dropout for regularization
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(1, activation='linear')
    ])
    return model

def tens_flow(data):
    
    seed=12345
    train,test=train_test_split(data,
                                test_size=0.10,shuffle=False,random_state=seed)
    train.dropna(inplace=True)
    features_train=train.drop(['num_orders'],axis=1)
    features_valid=test.drop(['num_orders'],axis=1)
    target_train=train['num_orders']
    target_valid=test['num_orders'] 
    
    # Compiling the model
    model = build_model(features_train.shape[1])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.MeanSquaredError()])
    
    model.summary()
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Training the model using GPU if available
    with tf.device('/GPU:0'):  
        history = model.fit(features_train, target_train, epochs=200, batch_size=32, 
                            validation_data=(features_valid, target_valid), callbacks=[early_stopping])

    # Evaluating the model
    y_pred = model.predict(features_valid)
    rmse_score = mean_squared_error(target_valid, y_pred)**0.5
    print(f"rmse Score: {rmse_score}")
    results = ['Keras',rmse_score]
    results_df = pd.DataFrame({'model':[results[0]],'validation_score':[results[1]]})

    return results_df,model