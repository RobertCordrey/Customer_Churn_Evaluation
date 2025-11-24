End-to-End Machine Learning Workflow in R for Predicting Customer Churn Using Synthetic Data

This repository contains an end-to-end machine learning workflow in R that demonstrates how to generate synthetic customer data, preprocess features, train classification models, evaluate performance, export model artifacts, and run inference on new observations. The project is self-contained and is designed for learning, portfolio demonstration, or adapting to real-world churn modeling.

Project Overview
This script walks through the complete lifecycle of a supervised machine learning project using R. It simulates realistic customer-level data, applies standard preprocessing steps, and trains two popular classification models, logistic regression and random forest. The workflow then compares model performance using metrics such as accuracy, precision, recall, F1, and AUC. The best-performing model is saved and used to score new customers.

Key Features
Synthetic customer dataset with demographic and behavioral variables
Train-test split with stratified sampling
Preprocessing pipeline including centering, scaling, and one-hot encoding
Logistic regression and random forest models with repeated cross-validation
Model evaluation using accuracy, sensitivity, specificity, precision, recall, F1, and AUC
ROC curve visualization and variable importance ranking
Export of predictions, model comparison table, and saved model
Example inference for scoring new customers
Workflow Summary
Load required libraries
Generate synthetic dataset
Split data into training and testing sets
Preprocess numeric and categorical features
Train logistic regression and random forest models
Evaluate performance on the test set
Compare model results side-by-side
Create ROC curves and variable importance charts
Export artifacts
Score new unseen customer data

Installation
Required R packages:
tidyverse
caret
pROC
randomForest

Install missing packages with:
install.packages(c("tidyverse","caret","pROC","randomForest"))

How to Run
Clone or download this repository.
Open the script in RStudio or another R environment.
Run the script from top to bottom.
Output files will be created automatically in your working directory.

Output Files
test_predictions.csv – Predicted probabilities and classes for test customers
model_comparison.csv – Comparison of model performance metrics
model_rf_fit.rds – Saved random forest model

Repository Structure
├─ README.md
├─ churn_model_script.R
├─ test_predictions.csv
├─ model_comparison.csv
└─ model_rf_fit.rds

Inference Example
The script includes an example showing how to preprocess and score new customers using the saved model. This can be adapted for batch scoring or production workflows.

Use Cases
Learning R-based machine learning workflows
Portfolio demonstrations
Classroom or instructional examples
Rapid prototypes for churn prediction
Benchmarking models

License
This project is provided for educational and demonstration purposes. You may reuse or modify it with attribution.
