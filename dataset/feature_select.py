from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class FeatureSelection:
    def __init__(self, algo) -> None:
        self.algo_name = algo
        self.algorithm = None
    def fit(self, X, Y, params):
        if self.algo_name == 'k_best':
            self.algorithm = SelectKBest(f_classif, k=params['k'])
            self.algorithm.fit(X,Y)
    def get_selected_features(self):
        return self.algorithm.get_feature_names_out()
    def transform_data(self, X):
        return self.algorithm.transform(X)
    def explore_data(self, X, Y, save_path):
        # utility to explore feature correlations
        # and separability w.r.t to target binary variable
        X_norm = (X-X.mean())/X.std()
        comb = pd.concat([Y,X_norm],axis=1)
        comb_data = pd.melt(comb,
                            id_vars = Y.columns[0],
                            var_name = "Feature",
                            value_name = "FeatureValue")
        plt.figure(figsize=(10,10))
        vplot = sns.violinplot(x="Feature", 
                               y="FeatureValue", 
                               hue=Y.columns[0], 
                               data=comb_data,
                               split=True,
                               inner="quart")
        vplot.figure.savefig(save_path)
