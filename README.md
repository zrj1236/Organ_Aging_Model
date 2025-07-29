# Aging Model Training and Delta Age Calculation
This repository contains the code used for training aging prediction models using proteomic data from healthy individuals, and for calculating the corrected predicted age difference (PAD), also referred to as **delta age**.

## Overview
1.**Aging Model Training**: We trained machine learning models using proteomic data to predict age, using healthy individuals who had no ICD-10 disease records. The models were trained using organ-enriched proteins and sex as a covariate. The dataset was split into training and test datasets, with a five-fold cross-validation strategy for model optimization.

2.**Delta Age Calculation**: To address age bias in the predictions, a post-prediction correction is applied using a method based on locally weighted scatterplot smoothing (LOWESS). This method ensures that predicted age and chronological age are more aligned.
