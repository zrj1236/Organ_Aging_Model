# Organ-Specific Aging Prediction Using Proteomic Data
This repository contains the code for training and evaluating models that predict organ-specific aging using proteomic data. The models are developed using nested cross-validation.

## Overview
The goal of this project is to predict organ-specific aging from proteomic data collected from healthy individuals. The models utilize organ-enriched proteins and include sex as a covariate. The methodology employs nested cross-validation for unbiased model performance assessment and hyperparameter optimization.

## Requirements
Python = 3.13.5<br>
xgboost = 3.0.3<br>
scikit-learn = 1.7.1<br>
numpy  = 2.3.2<br>
scipy  =  1.16.1<br>
pandas  = 2.3.1<br>

## Script and Data
**Script**：<br>
OrganAgingModelNestedCV.py：Developing model using nested cross-validation<br>
**Data**：<br>
protein_data: Simulated proteomics data containing 1,000 samples and 100 proteins<br>
ages: Age of the samples<br>
sex: Sex of the samples<br>
test_organ_enriched_genes: Simulated organ-specific proteins (n=10)<br>

## How to Install and Run the Project

### Installation
Make sure you have Python 3 installed. You can install the required dependencies using:
```bash
pip install -r requirements.txt

### Runing the model
To run the Organ Aging model, execute the following command:
```bash
python3 OrganAgingModelNestedCV.py \
  --protein_data PATH_OF_PROTEIN_DATA \
  --organ_enriched_genes ORGAN_ENRICHED_GENE \
  --sex SEX \
  --age AGE
