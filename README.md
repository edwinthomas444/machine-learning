# Machine Learning

An automated tool for binary and multi-class classification and hyper-parameter optimization on stationary and streaming type datasets. 
Trains different architectures for traditional batch-type datasets (KNN, DT, Random Forests, SVM, Bagging, Boosting etc.) and streaming datasets (Hoeffding Tree classifier, SAM-KNN, Adaptive Hoeffding Trees, Adaptive Random Forests, OzaBag, OzaBoost etc.) and generates metric dumps and performance evaluation graphs (ROCs) comparing the best models.


### File Structure
```
├───configs
│   │───model_hparams.py
│
├───data
│   |───drug_consumption.data
│
├───dataset
│   │───dataset_base.py
│   │───feature_select.py
│
├───driver
│   |───driver.py
│
├───models
│   │───models.py
│
├───output
│   ├───run_20221001-124242
│       ...
|       ...
|       ...
└───utils
    │───plot_results.py
    │───scoring.py
```

### File descriptions
1. `model_hparams.py`: Hyperparameter combinations for each model can be specified here.
2. `drug_consumption.data`: Stores the dataset (all datasets are stored under data folder.)
3. `driver.py`: Starting point for execution of the program (default).
4. `driver_online.py`: Starting point for execution of the program for online models.
5. `models.py`: Model classes and definitions.
6. `plot_results.py`: Utility to plot ROC curves
7. `scoring.py`: Utility to compute different metrics such as GMean, F-score, AUC etc.
8. `dataset.py`: Dataset class, used for preparing train test splits and pre-processing data.
9. `feature_select.py`: Feature Selection algorithms used for feature reduction based on statistical tests.
10. `output`: Directory where run dumps are generated with evaluation of models and vizualisation of performance through ROC plots and confusion metrics.

### Run cmd

1. Batch based Models
```
# Navigate to the root directory
>> python ./driver/driver.py
```

2. Online Models

```
# Navigate to the root directory
>> python ./driver/driver_online.py
```
