from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
from utils.scoring import score_fn_gmean, score_fn_hybrid

from utils.scoring import compute_metrics

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE 
import numpy as np
from sklearn.model_selection import StratifiedKFold


models = [
    'KNeighborsClassifier',
    'RandomForestClassifier',
    'DecisionTreeClassifier',
    'SVC'
]

class Model:
    def __init__(self, name) -> None:
        self.name = name
        self.mod = eval(name)()

    def get_model(self):
        return self.mod

    def custom_grid_search(self, params, dataX, dataY, oversample=True):
        cv_inner_hp= StratifiedKFold(n_splits=8, shuffle=True, random_state=1)
        custom_scorer = make_scorer(score_fn_hybrid, greater_is_better=True)
        if oversample:
            imba_pipeline = make_pipeline(SMOTE(random_state=32), 
                                          self.mod)
            # convert params to imbpipeline format
            new_params = {f'{self.name.lower()}__' + key: params[key] for key in params}
            grid = GridSearchCV(imba_pipeline, param_grid=new_params, cv=cv_inner_hp, scoring=custom_scorer,
                                verbose=10, return_train_score=True, refit=True)
            
        else:
            grid = GridSearchCV(estimator=self.mod, param_grid=params, scoring=custom_scorer, 
                                cv=cv_inner_hp, verbose=10, return_train_score=True, refit=True)
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


