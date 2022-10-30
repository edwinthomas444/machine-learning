
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

class Dataset:
    def __init__(self, file_path) -> None:
        self.fp = file_path
        self.df = self.read_df(file_path)
        # create a preprocessing dataframe
        self.pre_df = self.pre_process()
    def read_df(self, file_name):
        df = pd.read_csv(file_name)
        return df

    def pre_process(self):
        self.feature_cnames = ["ID","Age","Gender",
                "Education","Country","Ethnicity",
                "Nscore","Escore","Oscore",
                "Ascore","Cscore","Impulsive",
                "SS"]
        self.label_cnames = ["Alcohol","Amphet",
                "Amyl","Benzos","Caff",
                "Cannabis","Choc","Coke",
                "Crack","Ecstacy","Heroin",
                "Ketamine","Legath","LSD",
                "Meth","Mushrooms","Nicotine",
                "Semer","VSA"]
        
        cp_df = self.df.copy()
        cp_df.columns = self.feature_cnames + self.label_cnames

        # turn label columns into binary
        for lc_name in self.label_cnames:
            cp_df.loc[cp_df[lc_name].isin(["CL0","CL1"]),lc_name]=0
            cp_df.loc[cp_df[lc_name].isin(["CL2","CL3","CL4","CL5","CL6"]),lc_name]=1
        
        return cp_df

    def get_Xy(self, label_col_name):
        if label_col_name not in self.label_cnames:
            raise Exception(f"Column name {label_col_name} not present in dataset")
        X = self.pre_df[self.feature_cnames].drop(["ID"],axis=1)
        y = self.pre_df[[label_col_name]].astype(int).values.ravel()
        return X,y

    def get_hold_out_splits(self, X, y, params):
        train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=params['splits'], random_state=3)
        return train_x, test_x, train_y, test_y

    def get_k_fold_splits(self, X, y, params):
        stf_fold = StratifiedKFold(n_splits=params['folds'], random_state=3, shuffle=True)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    # return train and test splits
    def create_dataset(self, label_col_name, method, params):
        X, y = self.get_Xy(label_col_name)
        if method == 'hold-out':
            splits = self.get_hold_out_splits(X,y,params)
        elif method == 'k-fold':
            splits = self.get_k_fold_splits(X,y,params)
        else:
            raise Exception(f'Data split {method} not defined')

        return splits