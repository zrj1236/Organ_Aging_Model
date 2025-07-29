#import xgboost as xgb
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import linear_model
from statsmodels.nonparametric.smoothers_lowess import lowess

def XGboostTrain(X,Y):
    reg=xgb.XGBRegressor()
    param_grid = {'eta':[0.01,0.1,0.3],'n_estimators':[100,200,300,400,500]}
    grid_search = GridSearchCV(reg, param_grid, cv=5,scoring='r2')
    gs=grid_search.fit(X,Y)
    
    params = gs.best_params_
    
    xgb_model = xgb.XGBRegressor(eta=params['eta'],n_estimators=params['n_estimators'])
    xgb_model.fit(X,Y)
    
    return xgb_model

def RFTrain(X,Y):
    reg=RandomForestRegressor()
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


def compute_lowess_residuals(chronological_age, predicted_age, frac=0.3):
    """
    Compute residuals from LOWESS regression between actual and predicted ages.

    Parameters:
    - actual_age: array-like, actual chronological ages (usually integers)
    - predicted_age: array-like, predicted ages (usually continuous values)
    - frac: float, smoothing parameter for LOWESS (between 0 and 1)

    Returns:
    - residuals: array of residuals (predicted - lowess_fitted)
    - fitted: array of LOWESS fitted values
    """
    actual_age = np.array(chronological_age)
    predicted_age = np.array(predicted_age)
    
    # Perform LOWESS regression
    lowess_fit = lowess(endog=predicted_age, exog=chronological_age, frac=frac, return_sorted=False)
    
    # Calculate residuals
    residuals = predicted_age - lowess_fit
    
    return residuals, lowess_fit
    
if __name__ == '__main__':
    
    #Load the proteomics data, where each row represents a sample and each column represents a protein. 
    protein_data = pd.read_csv('protein_data',header=0,index_col=0,sep='\t')
    
    #read immune-enriched genes
    immu_genes = np.loadtxt('immune.enriched.genes1',dtype=str)
    
    #extract protein data of immune-enriched genes
    print(protein_data,immu_genes)
    protein_data = protein_data[np.intersect1d(protein_data.columns,immu_genes)]
    
    #read age and sex of samples
    sex = pd.read_csv('sex',header=0,sep='\t',index_col=0)
    chronological_age = pd.read_csv('ages',header=0,sep='\t',index_col=0)
    
    #add sex to proteomics data as a covriate
    protein_sex = protein_data.join(sex)
    
    #Ensure that the index of age and protein data are consistent
    chronological_age = chronological_age.squeeze().loc[protein_sex.index]
    
    #tuned the L1 regularization parameters via five-fold cross-validation using the GridSearchCV function from scikit-learn in training sets
    model = LassoTrain(protein_sex,chronological_age)
    
    #calculate predicted age of training and testing sets
    predicted_age = model.predict(protein_sex)
    
    delta_age,lowess_fit_age = compute_lowess_residuals(chronological_age, predicted_age)

    output = pd.DataFrame()
    output['chronological age'] = chronological_age
    output['predicted age'] = predicted_age
    output['lowess age'] = lowess_fit_age
    output['delta age'] = delta_age
    output['Zscored delta age'] = (delta_age-np.mean(delta_age))/np.std(delta_age)
    
    output.to_csv('predicted.result',header=True,index=True,sep='\t')
    
    
    
    
    
