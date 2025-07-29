Aging Model Training and Delta Age Calculation
This repository contains the code used for training aging prediction models using proteomic data from healthy individuals, and for calculating the corrected predicted age difference (PAD), also referred to as delta age.

Overview
The repository is divided into two main parts:

Aging Model Training: We trained machine learning models using proteomic data to predict age, using healthy individuals who had no ICD-10 disease records. The models were trained using organ-enriched proteins and sex as a covariate. The dataset was split into training and test datasets, with a five-fold cross-validation strategy for model optimization.

Delta Age Calculation: To address age bias in the predictions, a post-prediction correction is applied using a method based on locally weighted scatterplot smoothing (LOWESS). This method ensures that predicted age and chronological age are more aligned.

Project Setup
To get started, clone this repository to your local machine:


git clone https://github.com/your-username/aging-model.git
cd aging-model
Required Libraries
You will need the following Python libraries to run the code:

scikit-learn: For model training, parameter optimization, and cross-validation.

XGBoost: For XGBoost model training.

pandas: For data manipulation.

numpy: For numerical operations.

statsmodels: For LOWESS regression.

To install the required libraries, you can create a virtual environment and install the dependencies:


pip install -r requirements.txt
Where the requirements.txt file should contain:


scikit-learn
xgboost
pandas
numpy
statsmodels
Code Structure
model_training.py: Contains the code for model training, hyperparameter tuning using GridSearchCV, and the evaluation of models (Elastic Net, XGBoost, and Random Forest).

delta_age.py: Implements the method for calculating the delta age, which uses LOWESS regression to correct the age bias in predicted age.

data/: Contains any necessary data files (e.g., organ_data.csv).

results/: Saves model performance metrics, predicted ages, and other results.

README.md: This file.

Model Training
Dataset
The proteomics data is assumed to be in the form of a CSV file (organ_data.csv) where:

Rows represent individual samples.

Columns represent organ-enriched proteins, along with Age and Sex as additional features.

The dataset is divided into:

Training dataset: 80% of the samples are used for training the model.

Test dataset: 20% of the samples are reserved for testing, and this data is completely isolated from the training process.

Training Procedure
Hyperparameter tuning: We used five-fold cross-validation with the GridSearchCV function to optimize hyperparameters for each model:

Elastic Net: Tuned parameters include alpha (regularization strength) and l1_ratio (balance between L1 and L2 penalties).

XGBoost: Tuned parameters include learning_rate and n_estimators (number of estimators).

Random Forest: Tuned parameters include n_estimators (number of estimators).

Models Used:

Elastic Net: A linear regression model with L1 and L2 regularization.

XGBoost: A gradient boosting algorithm for structured/tabular data.

Random Forest: A bagging ensemble model using decision trees.

After cross-validation, the best models were selected based on their performance on the validation folds.

Example Usage
To train the models and optimize parameters, run the following code:

python model_training.py
This will train the models using the training data and optimize the hyperparameters. It will also evaluate the models on both the training and test datasets.

Delta Age Calculation
Addressing Age Bias
To avoid the common issue of predicted age bias, where the predicted age difference (PAD) tends to correlate negatively with chronological age, we apply the LOWESS regression technique to adjust for this bias.

The delta age is calculated as the residuals from the LOWESS regression model. This method allows for a more accurate and unbiased prediction of age differences.

Example Usage
To calculate the delta age for predicted values:

python delta_age.py
This will calculate the delta age (corrected PAD) by applying the LOWESS regression on the predicted and chronological ages.
