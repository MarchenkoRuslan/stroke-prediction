# Stroke Prediction Model

This repository contains a machine-learning project aimed at predicting stroke events. The project utilizes the XGBoost algorithm, which is particularly well-suited for imbalanced classification tasks like this one.

## Context
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

## Project Overview

Strokes are a leading cause of mortality and severe long-term disability. This project is part of an effort to apply machine learning to help in early detection and potentially provide life-saving insights.

## Dataset

The dataset used in this project includes medical records of patients with various features such as age, hypertension presence, heart disease presence, average glucose level, BMI, and smoking status. Each record is labeled with whether a stroke occurred for that patient.

## Features

The following features are included in the dataset:

- `age`: Age of the patient
- `hypertension`: Presence of hypertension
- `heart_disease`: Presence of heart disease
- `avg_glucose_level`: Average blood glucose level
- `bmi`: Body mass index
- `gender`: Gender of the patient
- `smoking_status`: Smoking status of the patient

## Preprocessing

Data preprocessing steps included scaling of numerical features and one-hot encoding of categorical variables to prepare the dataset for model training.

## Model Training

The XGBoost classifier was trained with a special focus on handling the class imbalance inherent in the dataset. A grid search cross-validation approach was used to tune the hyperparameters of the model.

## Model Evaluation

The model was evaluated based on several metrics including precision, recall, F1-score, and the Area Under the ROC Curve (AUROC). A high recall for the 'Stroke' class is desirable for medical diagnostic tools to capture as many true stroke events as possible.

## Results

- High recall for 'Stroke' class (0.82) with a trade-off in precision (0.13).
- AUROC of 0.84 indicates good discriminative ability of the model.
- Grid search identified the best hyperparameters for the current dataset.

## Conclusions

The model shows promise in detecting stroke events, which is critical for early intervention. Further work is needed to refine the model and manage the precision-recall trade-off appropriately.

## Usage

To run the prediction model:

2. Load the model using `joblib.load('stroke_prediction_model.xgb')`.
3. Make predictions using the `predict` method on the loaded model.