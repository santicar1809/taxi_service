import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from tensorflow.keras.models import load_model
import joblib

def test_main():
    next_hr_features={'year':[2018],'month':[9],'day':[1],'hour':[0],'lag_1':[205.0],'lag_2':[223.0],'lag_3':[159.0],'lag_4':[136.0],'rolling_mean':[170.750000]}
    next_hr_features=pd.DataFrame(next_hr_features)
    features_test=next_hr_features
    models_name=['lr','xg','lgbm','rf','cat','dt']
    list_pred=[]
    for model in models_name:    
        new_model = joblib.load(f'./files/modeling_output/model_fit/best_random_{model}.joblib')
        pred_next_hour = new_model.predict(features_test)
        list_pred.append(pred_next_hour)
    results_df = pd.DataFrame(list_pred, columns=['test_score'],index=models_name)
    print(results_df)
    #model_nn = joblib.load(f'./files/modeling_output/model_fit/best_random_nn.joblib')
    model_nn = load_model('./files/modeling_output/model_fit/best_random_nn.h5')
    predict=model_nn.predict(features_test)
    results_n_df = pd.DataFrame({'test_score':[predict]},index=['Keras'])
    final_rev = pd.concat([results_df,results_n_df])
    final_rev.to_csv('./files/modeling_output/reports/model_test_report.csv',index=True)

    return final_rev
    
results_test = test_main()

print(results_test)