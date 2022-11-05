
tree_params = {'class_weight':['balanced',None],
              'criterion':['gini','entropy','log_loss'],
              'min_samples_leaf':list(range(15,30,2)),
              'max_features':['sqrt', None],
              'splitter':['best'],
              'random_state':[3]}


forest_params = {'class_weight':['balanced','None'],
              'criterion':['gini','entropy'],
              'min_samples_leaf':list(range(1,36,5)), # 10, 40 ,5
              'max_features':['sqrt'],
              'n_estimators':list(range(20,51,5)),
              'bootstrap':[True],
              'oob_score':[True, False],
              'max_samples':[0.2,0.25,0.30],
              'random_state':[3]} # % of dataset need to be sampled when bootstrop for tree prepared

knn_params = {'n_neighbors':list(range(1,36,2)), # 2 to 36
              'weights':['uniform','distance'],
              'p':list(range(1,5))}


svm_params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma':['scale','auto'], # changes scaling for kernel functions rbf, linear and sigmoid
                'degree':list(range(0,4)),
                'C':[0.2,0.5,1.0],
                'probability':[True],
                'random_state':[3]}

mlp_params = {
    'hidden_layer_sizes': [(20,),(14,),(8,),(4,),(2,)],
    'activation': ['identity','tanh', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive','invscaling'],
    'random_state': [3]
}

gb_params = {
    'loss':['exponential','log_loss'],
    'learning_rate':[0.1,0.25,0.5,0.75],
    'max_depth':[3,4,5,6],
    'min_samples_leaf':list(range(1,36,5)),
    'random_state':[3]
}
