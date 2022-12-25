from skmultiflow.rules import VeryFastDecisionRulesClassifier
from skmultiflow.lazy import KNNClassifier

nc_params = {
    'def':{}
}
maj_params = {
    'def':{}
}
ht_params = {
    'def':{}
}
hta_params = {
    'def':{}
}
oboost_params = {
    'def':{}
}
hteddmc_params = {
    'def':{}
}
htadwin_params = {
    'def':{}
}
arf_params = {
    'def':{}
}

# base estimator configs
obag_params = {'vfdt':{
                    'base_estimator':VeryFastDecisionRulesClassifier(),
                    'n_estimators':10},
               'knn':{
                    'base_estimator':KNNClassifier(n_neighbors=8, max_window_size=1000, leaf_size=30),
                    'n_estimators':10}
}

levbag_params = {'vfdt':{
                    'base_estimator':VeryFastDecisionRulesClassifier(),
                    'n_estimators':10},
               'knn':{
                    'base_estimator':KNNClassifier(n_neighbors=8, max_window_size=1000, leaf_size=30),
                    'n_estimators':10}
}

sam_params = {
    'ltm':{
        'n_neighbors':5, 
        'weighting':'distance',
        'max_window_size':1000,
        'stm_size_option':'maxACCApprox', 
        'use_ltm':True},
    'no_ltm':{
        'n_neighbors':5, 
        'weighting':'distance',
        'max_window_size':1000,
        'stm_size_option':'maxACCApprox', 
        'use_ltm':False}
}



