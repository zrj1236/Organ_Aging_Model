import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

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
    
    #delta age was calculated as the residuals from a locally weighted scatterplot smoothing (lowess) regression of predicted age on chronological age
    delta_age,lowess_fit_age = compute_lowess_residuals(chronological_age, predicted_age)
    
    output = pd.DataFrame()
    output['chronological age'] = chronological_age
    output['predicted age'] = predicted_age
    output['lowess age'] = lowess_fit_age
    output['delta age'] = delta_age
    output['Zscored delta age'] = (delta_age-np.mean(delta_age))/np.std(delta_age)
    
    #saving result for follow-up analysis
    output.to_csv('organ.result',header=True,index=True,sep='\t')