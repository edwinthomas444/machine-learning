
import sys

from numpy import ndarray
# sys.path.append(os.path.join(os.getcwd(),os.path.dirname(__file__)))
sys.path.append('./')
# os.environ['PYTHONPATH'] = f"{os.path.join(os.getcwd(),os.path.dirname(__file__))}:$PYTHONPATH"
# print(sys.path[-1])
import os
from ast import Raise
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from dataset.dataset_base import DrugDataset, LabourDataset, HeartDiseaseDataset
from dataset.feature_select import FeatureSelection
import pandas as pd
from models.models import Model
from configs.model_hparams import knn_params, tree_params, forest_params, svm_params, mlp_params, gb_params
from utils.plot_results import Plot
import time


param_dict = {
    'KNeighborsClassifier':knn_params,
    'RandomForestClassifier':forest_params,
    'DecisionTreeClassifier':tree_params,
    'SVC':svm_params,
    'MLPClassifier':mlp_params,
    'GradientBoostingClassifier':gb_params
}

def run_out_dir():
    unique_stamp = time.strftime("%Y%m%d-%H%M%S")
    return f'run_{unique_stamp}'

        
def driver():
    output_dir = os.path.join('output',run_out_dir())

    # for each label create a dataset
    
    # filt_labels = ["Amphet","Cannabis","Ecstacy","LSD","Mushrooms","VSA"]
    # filt_labels = ["Cannabis"]
    # filt_labels = ["Labour"]
    filt_labels = ["HeartDisease"]
    for label in filt_labels:
        # inside output dir, there will be one folder per dataset
        dataset_dir = os.path.join(output_dir, f'{label}_Dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        # init metrics and grah dict
        # create plot object
        best_metrics_models = []

        #### Dataset creation ####
        # ds = DrugDataset(file_path='data/drug_consumption.data',
        #             col_name=label)
        # ds = LabourDataset(attr_file='data/labour_negotiations_attributes.xlsx',
        #                     f1='data/labour_negotiations_train.txt',
        #                     f2='data/labour_negotiations_test.txt')
        ds = HeartDiseaseDataset(attr_file='data/uci_heart_disease_attributes.xlsx',
                                f1='data/uci_heart_disease.csv')

        ##### Feature Selection based on hold-out train set ####################
        train_x, test_x, train_y, test_y = next(ds.create_splits(method='hold-out', params={'splits':0.33}))
        train_y_df = train_y

        # select features
        fs_algo = 'k_best'
        fs = FeatureSelection(algo='k_best')

        # explore data
        if isinstance(train_y, list) or isinstance(train_y, ndarray):
            train_y_df = pd.DataFrame(train_y, columns=[label])

        fs.explore_data(train_x, train_y_df, save_path=os.path.join(dataset_dir,"feature_analysis.png"))
        # change value of k here
        fs.fit(train_x, train_y, params = {'k':4})
        print(f'Selected set of features for label: {label} and algo: {fs_algo}: {fs.get_selected_features()}')

        # dump the selected features
        with open(os.path.join(dataset_dir, 'Selected_Features.txt'),'w') as f:
            [f.write(s_feat+"\n") for s_feat in fs.get_selected_features()]
        ###########################################################################

        meth = 'k-fold'
        par = {'folds':10}

        # meth = 'hold-out'
        # par = {'splits':0.33}

        for f_ind, data in enumerate(ds.create_splits(method=meth, params=par)):
            train_x, test_x, train_y, test_y = data

            # get reduced feature sets
            train_x_trans = fs.transform_data(train_x)
            test_x_trans = fs.transform_data(test_x)

            # train using grid search and params
            model_list = ['KNeighborsClassifier', 'RandomForestClassifier', 
                        'DecisionTreeClassifier', 'SVC', 'MLPClassifier', 'GradientBoostingClassifier']

            plot_obj = Plot(out_file=os.path.join(dataset_dir, f'ROC_plot_fold_{f_ind}.png'))
            for model_name in model_list:
                mod = Model(model_name)
                _, train_stats, best_param_dict = mod.custom_grid_search(params=param_dict[model_name],
                                                        dataX=train_x_trans, 
                                                        dataY=train_y,
                                                        oversample=False)
                
                # get roc curves for each of the models and store them
                # model name, dclass_label_info, fpr, tpr
                all_metrics = mod.test(test_x_trans, test_y)
                auc_sc = all_metrics['auc']
                plot_obj.add_results(X = all_metrics['fpr'],
                                    Y = all_metrics['tpr'],
                                    label = f'ROC_{model_name}_fold_{f_ind} (AUC: {auc_sc:.2f})')
                # save confusion matrix
                all_metrics['confusion_matrix'].figure_.savefig(os.path.join(dataset_dir,f'CM_{model_name}_fold{f_ind}.png'))
                all_metrics['best_params'] = best_param_dict

                # adding fold index
                all_metrics['fold'] = f_ind
                best_metrics_models.append(all_metrics)

            # plot the results.
            plot_obj.plot(title=f'ROC curves for 4 models on {label} Classification',
                        xlabel='False Positive Rate',
                        ylabel='True Positive Rate')

        # save the best model hparams for each dataset in .csv
        df_model_meta = pd.DataFrame(best_metrics_models)
        df_model_meta.to_csv(os.path.join(dataset_dir, 'BestModel_MetaData.csv'))

            
                        
def main():
    # call the main driver
    driver()

if __name__ == "__main__":
    main()