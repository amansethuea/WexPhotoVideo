# Dataset Libraries
import pandas as pd
import numpy as np
import pickle

# Preprocessing Libraries
import contractions
import string
import spacy
import demoji
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
pd.options.display.max_rows = 10
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.svm import SVC
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed

# Other Libraries
import os
import sys
import time
import csv
import json
import re


class ModelTraining(object):
  def __init__(self):
    if sys.platform.startswith("win"):
      self.path = os.getcwd() + "/../Model_Train_Data_Files/"
      self.df = pd.read_csv(self.path + 'cleaned_reviews_approach3.csv')
      self.saved_models_path = os.getcwd() + "/../Saved_Models/"
    elif sys.platform.startswith("darwin"):
      self.path = os.getcwd() + "/Model_Train_Data_Files/"
      self.df = pd.read_csv(self.path + 'cleaned_reviews_approach3.csv')
      self.saved_models_path = os.getcwd() + "/Saved_Models/"
    
    self.svc_model_name = "model_svc_test.sav"
    self.lr_model_name = "model_lr_test.sav"

    # Splitting Data into Train / Test / Validate sets in constructor only to use global (self) variables in all functions
    print("Current Rating Distribution after data cleaning and pre-processing")
    print(self.df['Rating'].value_counts())

    X = self.df.Content.values.astype('U')
    y = self.df.Sentiment

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True)
    print("Input Variables train / test total records")
    print(self.X_train.shape)
    print(self.X_test.shape)
    print("Target Variable -> Sentiment train / test count distribution")
    self.y_train.value_counts(), self.y_test.value_counts()

    print("Data splitted successfully")
    print()

  def model_building(self):
      models = [LogisticRegression(random_state=1),
          RandomForestClassifier(random_state=1),
          KNeighborsClassifier(n_neighbors = 5),
          SVC(random_state=1),
          MultinomialNB()]
      results = []
      f1scores = []
      print("Calculating Accuracies and F1 Scores")
      for model in models:
        pipeline = Pipeline([
          ('vectorizer_tri_grams', TfidfVectorizer()),
          ('model', model)])
        pipeline.fit(self.X_train, self.y_train)
        scores = pipeline.score(self.X_test, self.y_test)
        predicted_val = pipeline.predict(self.X_test)
        f1_scores = f1_score(self.y_test, predicted_val, average='macro')
        results.append(scores)
        f1scores.append(f1_scores)
      print()
      for model, accuracy, f1 in zip(models, results, f1scores):
        print(f"Model: {model} >> Accuracy: {round(np.mean(accuracy), 3)}   |  F1 Score: {round(np.mean(f1), 3)}")

      print()
      return results, f1scores

  def hyper_param_tuning(self):
    print("Started Hyper Parameter Tuning using GridSearchCV")
    param_grid_lr = {
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.1, 1, 10],
        'model__solver': ['liblinear'],
        'model__max_iter': [100]
        }

    param_grid_rf = {
        'model__criterion': ['gini'],
        'model__min_samples_split': [4]
        }

    param_grid_svc = {
        'model__C': [1, 10],
        'model__gamma': [0.01, 0.001],
        'model__kernel': ['rbf']
        }

    param_grid = [param_grid_lr, param_grid_rf, param_grid_svc]
    models_final = [LogisticRegression(random_state=1),
                    RandomForestClassifier(random_state=1),
                    SVC(random_state=1)]

    best_params = []
    best_estimator = []

    for model, grid_params in zip(models_final, param_grid):
      print(f"Model: {model}")
      clf_pipeline = Pipeline(steps =[('vectorizer_tri_grams', TfidfVectorizer()),('model', model)])
      gs= GridSearchCV(clf_pipeline, grid_params, cv=5, scoring='accuracy', refit=True,verbose=3)
      gs.fit(self.X_train, self.y_train)
      best_params.append(gs.best_params_)
      best_estimator.append(gs.best_estimator_)

    print()
    return best_params, best_estimator

  def best_hyper_params(self):
    best_params, best_estimator = self.hyper_param_tuning()
    abrv = ['lr','rf','svc']
    for a, p, e in zip(abrv, best_params, best_estimator):
      print(f"""Model : {a.upper()}
      Best Parameter : {p}
      Best Estimator : {e} \n""")
    print()

  def calculate_display_results(self):
    print("Time for results")
    self.svc_result()
    self.logistic_regression_result()

  def svc_result(self):
    print("SVC Result")
    svc = SVC(C=10, gamma=0.01, kernel= 'rbf', random_state=1)
    model_svc = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer()),
        ('model', svc)])

    model_svc.fit(self.X_train, self.y_train)
    predict_svc = model_svc.predict(self.X_test)
    score = model_svc.score(self.X_test, self.y_test)
    print(score)

    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(self.y_test, predict_svc)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()

    # Classification Report
    print(classification_report(self.y_test, predict_svc, labels=[0, 1, 2]))
    print()

    # Saving SVC model
    filename = self.saved_models_path + self.svc_model_name
    pickle.dump(model_svc, open(filename, 'wb'))

  def logistic_regression_result(self):
    print("Logistic Regression Result")
    logistic_regression = LogisticRegression(C=1, penalty='l2', max_iter=100, solver='liblinear')

    model_lr = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer()),
        ('model', logistic_regression)])

    model_lr.fit(self.X_train, self.y_train)
    predict_lr = model_lr.predict(self.X_test)
    score = model_lr.score(self.X_test, self.y_test)
    print(score)

    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(self.y_test, predict_lr)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()

    # Classification Report
    print(classification_report(self.y_test, predict_lr, labels=[0, 1, 2]))
    print()

    # Saving LR model
    filename = self.saved_models_path + self.lr_model_name
    pickle.dump(model_lr, open(filename, 'wb'))

  def model_training_with_best_hyper_param(self):
    results, f1scores = self.model_building()
    self.best_hyper_params()
    models_hyp = [LogisticRegression(C=1, penalty='l2', max_iter=100, solver='liblinear'),
          RandomForestClassifier(criterion='gini', min_samples_split=4,random_state=1),
          SVC(C=10, gamma=0.01, kernel= 'rbf', random_state=1)]

    models_hyp_names = ['Logistic Regression', 'RandomForest Classifier', 'SVC']
    results_hyper = []
    f1scores_hyper = []
    model_stored = []

    for model in models_hyp:
      pipeline = Pipeline([
        ('vectorizer_tri_grams', TfidfVectorizer()),
        ('model', model)])
      pipeline.fit(self.X_train, self.y_train)
      scores = pipeline.score(self.X_test, self.y_test)
      predicted_val = pipeline.predict(self.X_test)
      f1_scores = f1_score(self.y_test, predicted_val, average='macro')
      results_hyper.append(scores)
      f1scores_hyper.append(f1_scores)

    for model, accuracy, f1 in zip(models_hyp_names, results_hyper, f1scores_hyper):
      print(f"""Model: {model}
                                  Accuracy: {round(np.mean(accuracy), 4)}
                                  F1 Score: {round(np.mean(f1), 4)}\n""")

    print()
    f1scores = [f1scores[0], f1scores[1], f1scores[4]]
    results = [results[0], results[1], results[4]]

    #Creating list to store F1 scores with hyperparameters
    df_plot = pd.DataFrame( {"Models": ["Logistic Regression",
                      "Random Forest",
                      "SVC"]*2,
              "F1 Scores": f1scores + f1scores_hyper,
              "Accuracy": results + results_hyper,
              "Normal/Hyper":(["Normal"]*3+["Hypertuned"]*3)
            })
    print(df_plot)
    print()


if __name__ == "__main__":
  obj = ModelTraining()
  obj.model_training_with_best_hyper_param()
  obj.calculate_display_results()

