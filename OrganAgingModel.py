import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
import pandas as pd

def XGboostTrain(X,Y):
    reg = xgb.XGBRegressor()
    param_grid = {'eta':[0.01,0.1,0.3],'n_estimators':[100,200,300,400,500]}
    grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
    gs = grid_search.fit(X,Y)
    
    params = gs.best_params_
    
    xgb_model = xgb.XGBRegressor(eta=params['eta'],n_estimators=params['n_estimators'])
    xgb_model.fit(X,Y)
    
    return xgb_model

def RFTrain(X,Y):
    reg = RandomForestRegressor()
    param_grid = {'n_estimators':[100,200,300,400,500]}
    grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
    gs=grid_search.fit(X,Y)
    
    params = gs.best_params_

    rf_model = RandomForestRegressor(n_estimators=params['n_estimators'])
    rf_model.fit(X,Y)

    return rf_model
   
def ENTrain(X,Y):
    reg = ElasticNet()
    param_grid = {'alpha':np.array(range(1,100,1))/100,'l1_ratio':np.array(range(1,100,1))/100}
    grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
    gs=grid_search.fit(X,Y)
    
    params = gs.best_params_
    en_model = ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'])
    
    en_model.fit(X,Y)
    
    return en_model
    
def evaluate_model(y_true, y_pred):
    r = pearsonr(y_true, y_pred)
	
    return r   
    
if __name__ == '__main__':
    
    #Load the full data for a specific organ, where each row represents a sample and each column represents an organ-enriched protein(including Sex and Age). 
    full_data = pd.read_csv('organ_data.csv',header=0,sep='\t',index_col=0)
    y = full_data['Age']
    X = full_data.drop('Age',axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Training model in training_set
    xgb_model = XGboostTrain(X_train,y_train)
    rf_model = RFTrain(X_train,y_train)
    en_model = ENTrain(X_train,y_train)
    
    #predict organ age of training sets
    y_pred_train_xgb =  xgb_model.predict(X_train)
    y_pred_train_rf =  rf_model.predict(X_train)
    y_pred_train_en =  en_model.predict(X_train)
    
    #predict organ age of test sets
    y_pred_test_xgb =  xgb_model.predict(X_test)
    y_pred_test_rf =  rf_model.predict(X_test)
    y_pred_test_en =  en_model.predict(X_test)

	# Evaluate XGBoost model performance on training and test sets
	r_train_xgb = evaluate_model(y_train, y_pred_train_xgb)
	r_test_xgb = evaluate_model(y_test, y_pred_test_xgb)
	
    # Evaluate Random Forest model performance on training and test sets
    r_train_rf = evaluate_model(y_train, y_pred_train_rf)
    r_test_rf = evaluate_model(y_test, y_pred_test_rf)

    # Evaluate Elastic Net model performance on training and test sets
    r_train_en = evaluate_model(y_train, y_pred_train_en)
    r_test_en = evaluate_model(y_test, y_pred_test_en)

	# Print out the evaluation results for all models
	print("XGBoost Model Evaluation:")
	print(f"Training R: {r_train_xgb:.4f}")
	print(f"Test R: {r_test_xgb:.4f}")
	print()

	print("Random Forest Model Evaluation:")
	print(f"Training R: {r_train_rf:.4f}")
	print(f"Test R: {r_test_rf:.4f}")
	print()

	print("Elastic Net Model Evaluation:")
	print(f"Training R: {r_train_en:.4f}")
	print(f"Test R: {r_test_en:.4f}")

    # Creating DataFrame for training results
    train_results = pd.DataFrame({
    'XGB_pred': y_pred_train_xgb,
    'RF_pred': y_pred_train_rf,
    'EN_pred': y_pred_train_en
    }, index=X_train.index)  # Using the same index as X_train to maintain consistency

    # Creating DataFrame for test results
    test_results = pd.DataFrame({
    'XGB_pred': y_pred_test_xgb,
    'RF_pred': y_pred_test_rf,
    'EN_pred': y_pred_test_en
    }, index=X_test.index)  # Using the same index as X_test to maintain consistency

    # Add a new column to indicate whether the data is from the training set or the test set
    train_results['Dataset'] = 'Training'
    test_results['Dataset'] = 'Test'

    # Concatenate the two DataFrames vertically (stack them)
    combined_results = pd.concat([train_results, test_results])
    combined_results.to_csv('organ_aging_prediction_result',header=True,index=True,sep='\t')
    
    
    
    
    
    

    
    
    
    
    
