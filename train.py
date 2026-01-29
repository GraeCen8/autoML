import sklearn as sk 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#
#model imports
#

from sklearn.model_selection import train_test_split
#model imports (regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#model imports (classification)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#model imports (clustering)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#model imports (dimensionality reduction)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#model imports (outlier detection)
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
#XG boost import
from xgboost import XGBClassifier
from xgboost import XGBRegressor


class ranges: 
    randomForestRange = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
    }

    treeRange = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    XGboostRange = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
    }

    svRange = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
    }

    linearRegressRange = {}
    logisticRegressRange = {}
    gaussianNBRange = {}

    knnRange = {
        'n_neighbors': [2, 3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
    }


class Test: 
    def __init__(self, classification=True, scoreFunc=None):   
            self.models = []  
            self.scoreboard = {}
            
            self.classification = classification
            self.scoreFunc = scoreFunc
            
            self.update_scoreBoard = lambda model_name, score: self.scoreboard.update({model_name: score})
    
    def random_forest_models(self):
            #randomForest
            for n_estimators in ranges.randomForestRange['n_estimators']:
                for max_depth in ranges.randomForestRange['max_depth']:
                    if self.classification:
                        model = lambda n_estimators=n_estimators, max_depth=max_depth: RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    else:
                        model = lambda n_estimators=n_estimators, max_depth=max_depth: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    self.models.append(model)
        
    def tree_models(self):
            #treeRange
            for max_depth in ranges.treeRange['max_depth']:
                for min_samples_split in ranges.treeRange['min_samples_split']:
                    for min_samples_leaf in ranges.treeRange['min_samples_leaf']:
                        if self.classification:
                            model = lambda max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf: DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                        else:
                            model = lambda max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf: DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                        self.models.append(model)

    def XGboost_models(self):
            #XGboost
            for n_estimators in ranges.XGboostRange['n_estimators']:
                for max_depth in ranges.XGboostRange['max_depth']:
                    if self.classification:
                        model = lambda n_estimators=n_estimators, max_depth=max_depth: XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    else:
                        model = lambda n_estimators=n_estimators, max_depth=max_depth: XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    self.models.append(model)

    def svm_models(self):
            #svm
            for C in ranges.svRange['C']:
                for kernel in ranges.svRange['kernel']:
                    if kernel == 'poly':
                        # Only add degree parameter for polynomial kernel
                        for degree in ranges.svRange['degree']:
                            if self.classification:
                                model = lambda C=C, kernel=kernel, degree=degree: SVC(C=C, kernel=kernel, degree=degree)
                            else:
                                model = lambda C=C, kernel=kernel, degree=degree: SVR(C=C, kernel=kernel, degree=degree)
                            self.models.append(model)
                    else:
                        # For other kernels, don't use degree parameter
                        if self.classification:
                            model = lambda C=C, kernel=kernel: SVC(C=C, kernel=kernel)
                        else:
                            model = lambda C=C, kernel=kernel: SVR(C=C, kernel=kernel)
                        self.models.append(model)
                        
    def regression_models(self):
            #logistic regression
            model = lambda : LogisticRegression(max_iter=1000)
            self.models.append(model)
            #gaussianNB
            model = lambda : GaussianNB()
            self.models.append(model) 
                        
    def knn_models(self):
            #knn
            for n_neighbors in ranges.knnRange['n_neighbors']:
                for weights in ranges.knnRange['weights']:
                    model = lambda n_neighbors=n_neighbors, weights=weights: KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                    self.models.append(model)

    def add_models(self): #loops to every value in ranges and add to models list
            self.random_forest_models()
            self.tree_models()
            self.XGboost_models()
            self.svm_models()
            if self.classification:
                    self.regression_models()
            if not self.classification:
                    self.knn_models()
                    
    def experiment(self, modelFunc, X_train, y_train, X_val, y_val):
            try:
                model = modelFunc()
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                # Ensure predictions are integers for classification
                if self.classification:
                    predictions = predictions.astype(int)
                metric = self.scoreFunc(y_val, predictions)
                return metric
            except Exception as e:
                print(f"Error with model {modelFunc().__class__.__name__}: {str(e)}")
                return 0  # Return 0 score for failed models

    def run_tests(self, X_train, y_train, X_val, y_val):
            results = []
            for modelFunc in self.models:
                score = self.experiment(modelFunc, X_train, y_train, X_val, y_val)
                update_name = modelFunc().__class__.__name__     
                self.update_scoreBoard(update_name, score) 
                results.append((update_name, score))
           # return results

    def split(self, X, y, val_size=0.2, test_size=0.1, random_state=42):
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_best_model(self):
            best_model = max(self.scoreboard, key=self.scoreboard.get)
            best_score = self.scoreboard[best_model]
            return best_model, best_score
        
    def save_scoreboard(self, filename):
        pass
    
    def save_model(self, model, filename):
            with open (filename, 'wb') as file:
                pickle.dump(model, file)

    def full(self, X, y, model_filename="best_model.pkl"):
            X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y, val_size=0.2, test_size=0.1, random_state=42)
            self.add_models()
            self.run_tests(X_train, y_train, X_val, y_val)
            best_model,  best_score_val = self.get_best_model()
            test_score = self.scoreFunc(best_model, X_test, y_test)
             
            self.save_scoreboard("scoreboard.csv")
            self.save_model(best_model, "best_model.pkl")
             
            print(f"Best Model: {best_model.__class__.__name__}")
            print(f"Validation Score: {best_score_val}")
            print(f"Test Score: {test_score}")
            
    
            
            return best_model, best_model.__class__.__name__, best_score_val, test_score, self.scoreboard
    


if __name__ == "__main__": 
    test1 = Test(classification=True, scoreFunc=sk.metrics.accuracy_score)
    #example usage with dummy data
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target
    best_model, best_model_name, best_score_val, test_score, scoreboard = test1.full(X, y)
    

    test1.save_scoreboard("scoreboard.csv")
    test1.save_model(best_model, "best_model.pkl")
    