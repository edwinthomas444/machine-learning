import sys
sys.path.append('./')

from dataset.dataset_online import InsectDataset
from models.online_models import NoChangeClassifier, MajorityClassClassifier, HoefdingTreesEDDMComplete, HoefdingTreesADWINComplete
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import OnlineBoostingClassifier, OzaBaggingClassifier, LeveragingBaggingClassifier


from evaluate.online_evaluate import EvaluatePrequentialSlidingWindow
from utils.plot_results import Plot
from configs.online_model_hparams import *
import time
import os
from tqdm import tqdm
import pandas as pd

param_dict = {
    'NoChangeClassifier':nc_params,
    'MajorityClassClassifier':maj_params,
    'HoeffdingTreeClassifier':ht_params,
    'HoeffdingAdaptiveTreeClassifier':hta_params,
    'SAMKNNClassifier':sam_params,
    'AdaptiveRandomForestClassifier':arf_params,
    'OnlineBoostingClassifier':oboost_params,
    'OzaBaggingClassifier':obag_params,
    'LeveragingBaggingClassifier':levbag_params,
    'HoefdingTreesEDDMComplete':hteddmc_params,
    'HoefdingTreesADWINComplete':htadwin_params
}


def run_out_dir():
    unique_stamp = time.strftime("%Y%m%d-%H%M%S")
    return f'run_{unique_stamp}'

def driver():
    output_dir = os.path.join('output',run_out_dir())

    labels = [
              'INSECTS-abrupt_balanced_norm',
              'INSECTS-incremental_balanced_norm',
              'INSECTS-gradual_balanced_norm']
    

    for label in labels:
        # Online Models
        models = ['NoChangeClassifier',
                  'MajorityClassClassifier',
                  'SAMKNNClassifier',
                  'HoeffdingTreeClassifier',
                  'HoeffdingAdaptiveTreeClassifier',
                  'AdaptiveRandomForestClassifier',
                  'OzaBaggingClassifier',
                  'LeveragingBaggingClassifier']

        # Drift Detection Models
        # models = ['HoefdingTreesADWINComplete','HoeffdingTreeClassifier']

        dataset_dir = os.path.join(output_dir, f'{label}_Dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        # plot evaluation result
        plt_obj = Plot(out_file=os.path.join(dataset_dir, f'{label}_accuracy.png'))

        # evaluate object
        preq_eval = EvaluatePrequentialSlidingWindow(pretrain_size=1000,
                                                     window_size=1000,
                                                     slide_factor=1000, # 1 1000  or 1000 1
                                                     window_wait=1)

        for model_name in models:
            # get data
            ds = InsectDataset(raw_dp=f'data/usp-stream-data/repository/{label}.arff',
                               proc_dp=f'data/usp-stream-data/repository/{label}.csv')
            
            # get model
            for k, v in param_dict[model_name].items():
                ds_stream = ds.get_data_stream()
                model_label = f'{model_name}_{k}'
                model = eval(model_name)(**v)
                acc_stream, acc = preq_eval.evaluate(model=model,
                                                    stream=ds_stream)
                

                print(f'\nAvg Prequential Accuracy of {model_label} on {label}: {acc}')
                
                # add acc_stream results in a csv
                df = pd.DataFrame(data=acc_stream,columns=[model_label])
                df.to_csv(os.path.join(dataset_dir, f'{label}_{model_label}_preq_sliding_accuracy.csv'))

                if hasattr(model, 'change_points'):
                    print('Change points: ',model.change_points)
                    df_c = pd.DataFrame(data=model.change_points, columns=['Change_Points'])
                    df_c.to_csv(os.path.join(dataset_dir, f'{label}_{model_label}_change_points.csv'))

                plt_obj.add_results(X = list(range(len(acc_stream))),
                                    Y = acc_stream,
                                    label = f'{model_label}')
            
        plt_obj.plot(title=f'Prequential Accuracies for {label} Classification',
                     xlabel='Time, instances',
                     ylabel='Prequential accuracy %')
        

def main():
    driver()

if __name__ == '__main__':
    main()