from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
from utils.scoring import score_fn_gmean, score_fn_hybrid

from utils.scoring import compute_metrics

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import BorderlineSMOTE
from enum import Enum

models = [
    'KNeighborsClassifier',
    'RandomForestClassifier',
    'DecisionTreeClassifier',
    'SVC',
    'MLPClassifier',
    'GradientBoostingClassifier'
]

class Sampling(Enum):
    nosampling = 0
    oversample = 2
    undersample = 3

class Model:
    def __init__(self, name) -> None:
        self.name = name
        self.mod = eval(name)()

    def get_model(self):
        return self.mod

    def custom_grid_search(self, params, dataX, dataY, num_features, sample=Sampling.nosampling):
        cv_inner_hp= StratifiedKFold(n_splits=9, shuffle=True, random_state=1)
        custom_scorer = make_scorer(score_fn_hybrid, greater_is_better=True)
        new_params = {f'{self.name.lower()}__' + key: params[key] for key in params}
        if sample==Sampling.oversample:
            # SMOTE(random_state=32)
            # RandomOverSampler(random_state=32)
            # BorderlineSMOTE(random_state=32)
            imba_pipeline = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                                          RandomOverSampler(random_state=32, n_jobs=-1),
                                          MinMaxScaler(),
                                          SelectKBest(f_classif, k=num_features),
                                          self.mod)
            # convert params to imbpipeline format
            grid = GridSearchCV(imba_pipeline, param_grid=new_params, cv=cv_inner_hp, scoring=custom_scorer,
                                verbose=10, return_train_score=True, refit=True, n_jobs=-1)
        elif sample==Sampling.undersample:
            # ClusterCentroids(random_state=32)
            # RandomUnderSampler(random_state=32)
            # OneSidedSelection(random_state=32)
            imba_pipeline = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                                          RandomUnderSampler(random_state=32)(random_state=32, n_jobs=-1),
                                          MinMaxScaler(),
                                          SelectKBest(f_classif, k=num_features),
                                          self.mod)
            # convert params to imbpipeline format
            grid = GridSearchCV(imba_pipeline, param_grid=new_params, cv=cv_inner_hp, scoring=custom_scorer,
                                verbose=10, return_train_score=True, refit=True, n_jobs=-1)
        else:
            pipeline = Pipeline([('imputer',SimpleImputer(missing_values=np.nan, strategy='mean')),
                                 ('scalar',MinMaxScaler()),
                                 ('selectkbest',SelectKBest(f_classif, k=num_features)),
                                 (self.name.lower(),self.mod)])

            grid = GridSearchCV(estimator=pipeline, param_grid=new_params, scoring=custom_scorer, 
                                cv=cv_inner_hp, verbose=10, return_train_score=True, refit=True, n_jobs=-1)
        grid.fit(dataX, dataY)

        # update mod
        self.mod = grid.best_estimator_
        return self.mod, pd.DataFrame(grid.cv_results_), grid.best_params_
    
    def test(self, X, Y):
        # get all rows of 2nd column corr. to +ve class probs
        pred_prob = self.mod.predict_proba(X)[:,1]
        # by taking max class or prob>=0.5 for +ve class
        pred_class = self.mod.predict(X)


        all_model_metrics = compute_metrics(Y, pred_class, pred_prob)
        all_model_metrics['model_name'] = self.name
        return all_model_metrics


