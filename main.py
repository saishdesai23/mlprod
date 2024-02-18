"""
File: main.py
Author: Saish Desai
Date: 2024-02-18

Description: RAG pipeline for developing a chatbot
"""

import os 
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from training import build_and_train
import dill as pickle
from preprocess import PreProcessing
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('models/'+filename, 'wb') as file:
        pickle.dump(model, file)
