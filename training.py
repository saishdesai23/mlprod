import os 
import json
import numpy as np
import pandas as pd
import dill as pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from preprocess import PreProcessing
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


def build_and_train():

	data = pd.read_csv('data/training.csv')
	data = data.dropna(subset=['Gender', 'Married', 'Credit_History', 'LoanAmount'])

	pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',\
				'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

	X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], \
														test_size=0.25, random_state=42)
	y_train = y_train.replace({'Y':1, 'N':0}).to_numpy()
	y_test = y_test.replace({'Y':1, 'N':0}).to_numpy()

	pipe = make_pipeline(PreProcessing(),
						RandomForestClassifier())

	param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
             "randomforestclassifier__max_depth" : [None, 6, 8, 10],
             "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
             "randomforestclassifier__min_impurity_decrease": [0.1, 0.2, 0.3]}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

	grid.fit(X_train, y_train)
    
	return(grid)