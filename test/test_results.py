import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from tensorflow.keras.models import load_model
import joblib

def test_main():
    features_test=pd.read_csv('./test/files/features_test.csv')
    target_test=pd.read_csv('./test/files/target_test.csv')
    models_name=['lr','xg','lgbm','rf','cat']
    results=[]
    for model in models_name:    
        new_model = joblib.load(f'./files/modeling_output/model_fit/best_random_{model}.joblib')
        prob = new_model.predict_proba(features_test)[:, 1]
        roc_auc = roc_auc_score(target_test,prob)
        results.append([model,roc_auc])
        
    results_df = pd.DataFrame(results, columns=['model','test_score'])
    
    #model_nn = joblib.load(f'./files/modeling_output/model_fit/best_random_nn.joblib')
    model_nn = load_model('./files/modeling_output/model_fit/best_random_nn.h5')
    proba=model_nn.predict(features_test).ravel()
    roc_auc_nn=roc_auc_score(target_test,proba)
    results_n=['Keras',roc_auc_nn]
    results_n_df = pd.DataFrame({'model':[results_n[0]],'test_score':[results_n[1]]})
    final_rev = pd.concat([results_df,results_n_df])
    final_rev.to_csv('./files/modeling_output/reports/model_test_report.csv',index=False)
    print(results_n_df)
    return final_rev
    #return results_df
results_test = test_main()

print(results_test)