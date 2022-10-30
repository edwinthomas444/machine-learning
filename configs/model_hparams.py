# param definition for DT model
tree_params_full = {'criterion':['gini','entropy','log_loss'],
              'splitter':['best','random'],
              'max_depth':list(range(2,20)),
              'min_samples_split':list(range(2,60)), # specifying end limit as twice the end limit of min_leaf, assuming l_n = r_n where split can happpen and both conditions satisfy
              'min_samples_leaf':list(range(3,30)),
              'max_features':['sqrt','log2',None],
              'class_weight':['balanced',None,{0:1,1:100},{0:1,1:10},{0:1,1:5}]}


tree_params_small = {'class_weight':['balanced'],
                    'criterion':['gini','entropy','log_loss'],
                    'min_samples_leaf':list(range(20,30,2)),
                    'max_features':['sqrt','log2',None],
                    'splitter':['best','random']}

tree_params = {'class_weight':['balanced',None,{0:1,1:10},{0:1,1:5}],
              'criterion':['gini','entropy','log_loss'],
              'min_samples_leaf':list(range(3,30)),
              'max_features':['sqrt','log2',None],
              'splitter':['best','random'],
              'random_state':[3]}


# param definition for Random Forest Model
forest_params_small = {'class_weight':['balanced'],
              'criterion':['gini','entropy','log_loss'],
              'min_samples_leaf':list(range(20,40,5)),
              'max_features':['sqrt','log2'],
              'n_estimators':[25,30],
              'bootstrap':[True],
              'oob_score':[True, False],
              'max_samples':[0.2,0.25,0.30]} # % of dataset need to be sampled when bootstrop for tree prepared

forest_params = {'class_weight':['balanced','None'],
              'criterion':['gini','entropy','log_loss'],
              'min_samples_leaf':list(range(1,36,5)), # 10, 40 ,5
              'max_features':['sqrt','log2'],
              'n_estimators':list(range(20,51,5)),
              'bootstrap':[True],
              'oob_score':[True, False],
              'max_samples':[0.2,0.25,0.30],
              'random_state':[3]} # % of dataset need to be sampled when bootstrop for tree prepared


knn_params_small = {'n_neighbors':list(range(2,5)), 
              'weights':['uniform','distance'],
              'p':list(range(1,5))} # indicates power of distance, 2->euc, 1 manhattan, k minkowski distance

knn_params = {'n_neighbors':list(range(1,36,2)), # 2 to 36
              'weights':['uniform','distance'],
              'p':list(range(1,5))}


# param definition for SVM model
svm_params_small = {'kernel':['sigmoid'],
                    'gamma':['scale','auto'],
                    'degree':[1],
                    'C':[0.2,0.5,1.0,10.0],
                    'probability':[True]}

svm_params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma':['scale','auto'], # changes scaling for kernel functions rbf, linear and sigmoid
                'degree':list(range(0,4)),
                'C':[0.2,0.5,1.0,10.0],
                'probability':[True],
                'random_state':[3]}

# default params for testing (comment to use full grid space)
svm_params = {'probability':[True], 'random_state':[3]}
knn_params = {}
forest_params = {'random_state':[3]}
tree_params = {'random_state':[3]}