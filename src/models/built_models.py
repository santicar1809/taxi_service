import pandas as pd 
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
    models_name = ['lr','xg','lgbm','rf','cat']
    for model,i in zip(models,models_name):
        best_estimator, best_score, val_score = model_structure(data, model[1], model[2]) #data, pipeline, param_grid
        results.append([model[0],best_estimator,best_score, val_score])
        
        joblib.dump(best_estimator,output_path +f'best_random_{i}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    
    tf_results = tens_flow(data) 
    #joblib.dump(tf_results[1],output_path +f'best_random_nn.joblib')
    tf_results[1].save(output_path+'best_random_nn.h5')
    # Concatening logistic models and neuronal network
    final_rev = pd.concat([results_df,tf_results[0]])
    final_rev.to_csv('./files/modeling_output/model_report.csv',index=False)

    return final_rev[['model','validation_score']]


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    data_train,data_test=train_test_split(data,random_state=seed,test_size=0.1)
    
    features=data_train.drop(['cancel'],axis=1)
    target=data_train['cancel']
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,random_state=seed,test_size=0.2)
    
    features_test=data_test.drop(['cancel'],axis=1)
    target_test=data_test['cancel']
    
    imputer = SimpleImputer(strategy='mean')
    
    features_train_imp = imputer.fit_transform(features_train)
    features_valid_imp = imputer.transform(features_valid)
    features_test_imp = imputer.transform(features_test)
    
    features_train_imp=pd.DataFrame(features_train_imp,columns=features_train.columns,index=target_train.index)
    features_valid_imp=pd.DataFrame(features_valid_imp,columns=features_valid.columns,index=target_valid.index)
    features_test_imp=pd.DataFrame(features_test_imp,columns=features_test.columns,index=target_test.index)
    
    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=2)
    gs.fit(features_train_imp,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    score_val = eval_model(best_estimator,features_valid_imp,target_valid)
    print(f'AU-ROC: {score_val}')
    features_test_imp.to_csv('./test/files/features_test.csv',index=False)
    target_test.to_csv('./test/files/target_test.csv',index=False)
    results = best_estimator, best_score, score_val 
    return results
    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    prob = best.predict_proba(features_valid)[:, 1]
    random_accuracy=accuracy_score(target_valid,random_prediction)
    random_f1_score=f1_score(target_valid,random_prediction)
    random_roc_auc=roc_auc_score(target_valid,prob)
    print("Accuracy:",random_accuracy)
    print('f1: ',random_f1_score)
    print('ROC_AUC: ',random_roc_auc)
    print('Best_Model: ',best)
    return random_roc_auc
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
        Dense(1, activation='sigmoid')
    ])
    return model

def tens_flow(data):
    
    seed=12345
    data_train,data_test=train_test_split(data,random_state=seed,test_size=0.1)
    
    features=data_train.drop(['cancel'],axis=1)
    target=data_train['cancel']
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,random_state=seed,test_size=0.2)
    
    features_test=data_test.drop(['cancel'],axis=1)
    target_test=data_test['cancel']
    
    imputer = SimpleImputer(strategy='mean')
    
    features_train_imp = imputer.fit_transform(features_train)
    features_valid_imp = imputer.transform(features_valid)
    features_test_imp = imputer.transform(features_test)
    
    features_train_imp=pd.DataFrame(features_train_imp,columns=features_train.columns,index=target_train.index)
    features_valid_imp=pd.DataFrame(features_valid_imp,columns=features_valid.columns,index=target_valid.index)
    features_test_imp=pd.DataFrame(features_test_imp,columns=features_test.columns,index=target_test.index)
    
    # Compiling the model
    model = build_model(features_train_imp.shape[1])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
    
    model.summary()
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Training the model using GPU if available
    with tf.device('/GPU:0'):  
        history = model.fit(features_train_imp, target_train, epochs=200, batch_size=32, 
                            validation_data=(features_valid_imp, target_valid), callbacks=[early_stopping])

    # Evaluating the model
    y_pred = model.predict(features_valid_imp).ravel()
    auc_score = roc_auc_score(target_valid, y_pred)
    print(f"AU-ROC Score: {auc_score}")
    results = ['Keras',auc_score]
    results_df = pd.DataFrame({'model':[results[0]],'validation_score':[results[1]]})

    return results_df,model