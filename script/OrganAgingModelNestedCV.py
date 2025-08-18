import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import KFold
import argparse

def XGboostTrain(X,Y):
	reg = xgb.XGBRegressor(n_jobs=64)
	param_grid = {'eta':[0.01,0.1,0.3],'n_estimators':[100,200,300,400,500]}
	grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
	gs = grid_search.fit(X,Y)
	
	params = gs.best_params_
	
	xgb_model = xgb.XGBRegressor(eta=params['eta'],n_estimators=params['n_estimators'])
	xgb_model.fit(X,Y)
	
	return xgb_model,params

def RFTrain(X,Y):
	reg = RandomForestRegressor(n_jobs=64)
	param_grid = {'n_estimators':[100,200,300,400,500]}
	grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
	gs=grid_search.fit(X,Y)
	
	params = gs.best_params_

	rf_model = RandomForestRegressor(n_estimators=params['n_estimators'])
	rf_model.fit(X,Y)

	return rf_model,params
   
def ENTrain(X,Y):
	reg = ElasticNet(selection='random')
	param_grid = {'alpha':np.array(range(1,100,1))/100,'l1_ratio':np.array(range(1,100,1))/100}
	grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
	gs=grid_search.fit(X,Y)
	
	params = gs.best_params_
	en_model = ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'])
	
	en_model.fit(X,Y)
	
	return en_model,params
	
def evaluate_model(y_true, y_pred):
	r2 = r2_score(y_true, y_pred)
	
	return r2
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--protein_data',type=str,help='Path of protein data')
	parser.add_argument('--organ_enriched_genes',type=str,help='Path of organ enriched genes')
	parser.add_argument('--sex',type=str,help='Path of sex information')
	parser.add_argument('--age',type=str,help='Path of age information')
		
	args = parser.parse_args()
		
	full_data = pd.read_csv(args.protein_data,header=0,index_col=0,sep='\t')
	organ_enriched_genes = np.loadtxt(args.organ_enriched_genes,dtype=str)
	organ_data = full_data[organ_enriched_genes]

	sex = pd.read_csv(args.sex,header=0,sep='\t',index_col=0)
	organ_data = organ_data.join(sex)

	age = pd.read_csv(args.age,header=0,sep='\t',index_col=0)
	organ_data = organ_data.join(age)

	y = organ_data['age']
	X = organ_data.drop('age',axis=1)
		
	n_nested = 5
	kf = KFold(n_splits=5, shuffle=True, random_state=42)
		
	for k, (train_index, test_index) in enumerate(kf.split(X), 1):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		#Training model in training_set
		xgb_model,xgb_params = XGboostTrain(X_train,y_train)
		rf_model,rf_params = RFTrain(X_train,y_train)
		en_model,en_params = ENTrain(X_train,y_train)
			
		#predict organ age of training sets
		y_pred_train_xgb =	xgb_model.predict(X_train)
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
		print(f"Training R2 for fold {k} outer fold: {r_train_xgb:.4f}")
		print(f"Test R2 for fold {k} outer fold: {r_test_xgb:.4f}")
		print()

		print("Random Forest Model Evaluation:")
		print(f"Training R2 for fold {k} outer fold: {r_train_rf:.4f}")
		print(f"Test R2 for fold {k} outer fold: {r_test_rf:.4f}")
		print()

		print("Elastic Net Model Evaluation:")
		print(f"Training R2 for fold {k} outer fold: {r_train_en:.4f}")
		print(f"Test R2 for fold {k} outer fold: {r_test_en:.4f}")

		# Creating DataFrame for training results
		train_results = pd.DataFrame({
		'XGB_pred': y_pred_train_xgb,
		'RF_pred': y_pred_train_rf,
		'EN_pred': y_pred_train_en
		}, index=X_train.index)	 # Using the same index as X_train to maintain consistency

		# Creating DataFrame for test results
		test_results = pd.DataFrame({
		'XGB_pred': y_pred_test_xgb,
		'RF_pred': y_pred_test_rf,
		'EN_pred': y_pred_test_en
		}, index=X_test.index)	# Using the same index as X_test to maintain consistency

		# Add a new column to indicate whether the data is from the training set or the test set
		train_results['Dataset'] = 'Training'
		test_results['Dataset'] = 'Test'

		# Concatenate the two DataFrames vertically (stack them)
		combined_results = pd.concat([train_results, test_results])
		combined_results.to_csv('aging_prediction_result_'+str(k),header=True,index=True,sep='\t')
			
		protein_weights_xgb = xgb_model.feature_importances_
		protein_weights_rf = rf_model.feature_importances_
		protein_weights_en = en_model.coef_
			
		protein_names = X_train.columns
			
		protein_weights_df = pd.DataFrame({
		'Feature': protein_names, 
		'XGBoost': protein_weights_xgb, 
		'RandomForest': protein_weights_rf, 
		'ElasticNet': protein_weights_en
		})
		   
		protein_weights_df.to_csv('model_protein_weights_'+str(k),header=True,index=True,sep='\t')
		   
		with open("xgb_model_params_"+str(k), "w") as file:
			for key, value in xgb_params.items():
				file.write(f"{key}: {value}\n")
			
				   
		with open("rf_model_params_"+str(k), "w") as file:
			for key, value in rf_params.items():
				file.write(f"{key}: {value}\n")
				  
		with open("en_model_params_"+str(k), "w") as file:
			for key, value in en_params.items():
				file.write(f"{key}: {value}\n")
