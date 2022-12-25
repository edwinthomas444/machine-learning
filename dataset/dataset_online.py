import pandas as pd
from scipy.io import arff
from skmultiflow.data.file_stream import FileStream

class InsectDataset:
    def __init__(self, raw_dp, proc_dp):
        self.proc_path = proc_dp
        self.raw_path = raw_dp
        self.df = self.get_df(raw_dp)
        self.class_map = self.get_class_map()
        self.pre_process(proc_dp)

    def get_df(self, path):
        data = arff.loadarff(path)
        df=pd.DataFrame(data[0])
        df['class']=df['class'].apply(lambda x: x.decode('utf8'))
        return df
    
    def get_class_map(self):
        cm = {x:i for i,x in enumerate(list(self.df['class'].unique()))}
        return cm

    def pre_process(self, proc_dp):
        # replace class labels with their integer representation
        self.df['class']=self.df['class'].apply(lambda x: self.class_map[x])
        # for testing
        # self.df = self.df.head(6000)
        # save to csv
        self.df.to_csv(proc_dp, index=False)

    def get_data_stream(self):
        stream = FileStream(self.proc_path)
        return stream

    
    



