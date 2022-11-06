
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class HeartDiseaseDataset:
    def __init__(self, attr_file, f1):
        self.f1 = f1
        self.attr_file = attr_file
        self.attr_df = self.get_attr()
        self.pre_df = self.pre_process()
        self.X, self.y = self.feature_engineer()

    def get_attr(self):
        df = pd.read_excel(self.attr_file).drop(columns = ["index"])
        df["feature_name"] = df["feature_name"].apply(lambda x: x.split(".")[0])
        self.col_name = df["feature_name"].tolist()
        return df
    
    def feature_engineer(self):
        # replace all missing values with np.nan
        cat_features = self.attr_df.loc[self.attr_df['type']=='categorical']['feature_name'].to_list()
        bin_features = self.attr_df.loc[self.attr_df['type']=='boolean']['feature_name'].to_list()
        cont_features = self.attr_df.loc[self.attr_df['type']=='continuous']['feature_name'].to_list()
        # binarize categorical features
        df_cat = pd.get_dummies(data=self.pre_df[cat_features], columns=cat_features)
        df_bin = self.pre_df[bin_features]
        df_num = self.pre_df[cont_features].astype(float)
        df_label =  self.pre_df['label']
        X = pd.concat([df_cat, df_bin, df_num], axis='columns')
        y = df_label.astype(int).values.ravel()
        return X,y

    def pre_process(self):
        dfd = pd.read_csv(self.f1)
        dfd.rename(columns = {'condition':'label'}, inplace = True)
        return dfd
    
    def normalize(self, train_x, test_x, norm = 'min-max'):
        if norm == 'min-max':
            scalar = MinMaxScaler()
        else:
            scalar = StandardScaler()
        scalar.fit(train_x)
        train_x, test_x = scalar.transform(train_x), scalar.transform(test_x)
        return train_x, test_x

    def get_hold_out_splits(self, X, y, params):
        train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=params['splits'], random_state=3)
        yield train_x, test_x, train_y, test_y

    def get_k_fold_splits(self, X, y, params):
        stf_fold = StratifiedKFold(n_splits=params['folds'], random_state=3, shuffle=True)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    def get_rep_k_fold_splits(self, X, y, params):
        stf_fold = RepeatedStratifiedKFold(n_splits=params['folds'], n_repeats=params['repeats'], random_state=3)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    # return train and test splits
    def create_splits(self, method, params):
        if method == 'hold-out':
            splits = self.get_hold_out_splits(self.X,self.y,params)
        elif method == 'k-fold':
            splits = self.get_k_fold_splits(self.X,self.y,params)
        elif method == 'rep-k-fold':
            splits = self.get_rep_k_fold_splits(self.X,self.y,params)
        else:
            raise Exception(f'Data split {method} not defined')

        return splits


class LabourDataset:
    def __init__(self, attr_file, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.attr_file = attr_file
        self.attr_df = self.get_attr()
        self.pre_df = self.pre_process()
        self.X, self.y = self.feature_engineer()

    def get_attr(self):
        df = pd.read_excel(self.attr_file).drop(columns = ["index"])
        df["feature_name"] = df["feature_name"].apply(lambda x: x.split(".")[0])
        self.col_name = df["feature_name"].tolist()
        return df
    def read_examples(self, file):
        def parse(lines, label_txt, label):
            elemd = {}
            c = True
            for line in lines:
                if line.startswith(label_txt):
                    c = False
                if c: continue
                if line.startswith("#") or line.startswith(label_txt):
                    continue
                elif line.strip(" ")=="\n":
                    c = True
                    continue
                else:
                    splitted = line.split()
                    elemd.setdefault(splitted[0],[label]).extend(splitted[1:])
            return elemd
        # processing uci dataset
        with open(file, "r") as f:
            lines = f.readlines()
            neg_arr = {}
            pos_arr = {}

            pos_arr = parse(lines, 'good-events', 1)
            neg_arr = parse(lines, 'bad-events', 0)
        return pos_arr, neg_arr
    
    def feature_engineer(self):
        # replace all missing values with np.nan
        self.pre_df.replace('*',np.nan, inplace=True)
        cat_features = self.attr_df.loc[self.attr_df['type']=='categorical']['feature_name'].to_list()
        bin_features = self.attr_df.loc[self.attr_df['type']=='boolean']['feature_name'].to_list()
        cont_features = self.attr_df.loc[self.attr_df['type']=='continuous']['feature_name'].to_list()
        nom_features = cat_features+bin_features
        df_cvt = pd.get_dummies(data=self.pre_df[nom_features], columns=nom_features)
        # replace 1 with 0.2
        df_cvt.replace(1, 0.2, inplace=True)
        df_num = self.pre_df[cont_features].astype(float)
        df_label =  self.pre_df['label']
        # replace nan with avg
        # df_num.fillna(df_num.mean(), inplace=True)
        # concat them
        X = pd.concat([df_cvt, df_num], axis='columns')
        y = df_label.astype(int).values.ravel()
        return X,y

    def pre_process(self):
        df1 = pd.read_excel(self.attr_file).drop(columns = ["index"])
        df1["feature_name"] = df1["feature_name"].apply(lambda x: x.split(".")[0])
        p1, n1 = self.read_examples(file=self.f1)
        p2, n2 = self.read_examples(file=self.f2)
        combined = list(p1.values()) + list(n1.values()) + list(p2.values()) + list(n2.values())
        self.col_name.insert(0,'label')
        cols = self.col_name
        df2 = pd.DataFrame(data=combined, columns = cols)
        return df2
    
    def normalize(self, train_x, test_x, norm = 'min-max'):
        if norm == 'min-max':
            scalar = MinMaxScaler()
        else:
            scalar = StandardScaler()
        scalar.fit(train_x)
        train_x, test_x = scalar.transform(train_x), scalar.transform(test_x)
        return train_x, test_x

    def get_hold_out_splits(self, X, y, params):
        train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=params['splits'], random_state=3)
        yield train_x, test_x, train_y, test_y

    def get_k_fold_splits(self, X, y, params):
        stf_fold = StratifiedKFold(n_splits=params['folds'], random_state=3, shuffle=True)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    def get_rep_k_fold_splits(self, X, y, params):
        stf_fold = RepeatedStratifiedKFold(n_splits=params['folds'], n_repeats=params['repeats'], random_state=3)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    # return train and test splits
    def create_splits(self, method, params):
        if method == 'hold-out':
            splits = self.get_hold_out_splits(self.X,self.y,params)
        elif method == 'k-fold':
            splits = self.get_k_fold_splits(self.X,self.y,params)
        elif method == 'rep-k-fold':
            splits = self.get_rep_k_fold_splits(self.X,self.y,params)
        else:
            raise Exception(f'Data split {method} not defined')

        return splits


class DrugDataset:
    def __init__(self, file_path, col_name) -> None:
        self.fp = file_path
        self.df = self.read_df(file_path)
        # create a preprocessing dataframe
        self.pre_df = self.pre_process()
        self.col_name = col_name
        self.X, self.y = self.get_Xy(col_name)

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
    
    def normalize(self, train_x, test_x, norm = 'min-max'):
        if norm == 'min-max':
            scalar = MinMaxScaler()
        else:
            scalar = StandardScaler()
        scalar.fit(train_x)
        train_x, test_x = scalar.transform(train_x), scalar.transform(test_x)
        return train_x, test_x

    def get_hold_out_splits(self, X, y, params):
        train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=params['splits'], random_state=3)
        yield train_x, test_x, train_y, test_y

    def get_k_fold_splits(self, X, y, params):
        stf_fold = StratifiedKFold(n_splits=params['folds'], random_state=3, shuffle=True)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    def get_rep_k_fold_splits(self, X, y, params):
        stf_fold = RepeatedStratifiedKFold(n_splits=params['folds'], n_repeats=params['repeats'], random_state=3)
        for train_ind, test_ind in stf_fold.split(X,y):
            train_x, test_x = X.iloc[train_ind], X.iloc[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]
            yield train_x, test_x, train_y, test_y

    # return train and test splits
    def create_splits(self, method, params):
        if method == 'hold-out':
            splits = self.get_hold_out_splits(self.X,self.y,params)
        elif method == 'k-fold':
            splits = self.get_k_fold_splits(self.X,self.y,params)
        elif method == 'rep-k-fold':
            splits = self.get_rep_k_fold_splits(self.X,self.y,params)
        else:
            raise Exception(f'Data split {method} not defined')

        return splits