
import numpy as np
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from utils.drift_detection import EDDM
from skmultiflow.drift_detection import ADWIN
# from skmultiflow.drift_detection.eddm import EDDM

class NoChangeClassifier:
    def __init__(self, model_name='NoChangeClassifier'):
        self.model_name = model_name

    def partial_fit(self, X, y, classes=None):
        self.last_samp_label = y[-1]

    def predict(self, X):
        preds = [self.last_samp_label for _ in X]
        return preds


class MajorityClassClassifier:
    def __init__(self, model_name='MajorityClassClassifier'):
        self.model_name = model_name
        self.global_bin_count = []

    def partial_fit(self, X, y, classes=None):
        if isinstance(self.global_bin_count, list):
            self.global_bin_count = np.bincount(y)
        else:
            local_bin_count = np.bincount(y)
            max_len = max(len(self.global_bin_count), len(local_bin_count))
            self.global_bin_count = np.append(self.global_bin_count, [0] * (max_len - len(self.global_bin_count)))
            local_bin_count = np.append(local_bin_count, [0] * (max_len - len(local_bin_count)))
            self.global_bin_count = self.global_bin_count + local_bin_count
            
        self.maj_class = np.argmax(self.global_bin_count)

    def predict(self, X):
        preds = [self.maj_class for _ in X]
        return preds

# class MajorityClassClassifierNaive:
#     def __init__(self, model_name='MajorityClassClassifier'):
#         self.model_name = model_name
#         self.global_bin_count = []

#     def partial_fit(self, X, y, classes=None):
#         self.global_bin_count = np.bincount(y)
#         self.maj_class = np.argmax(self.global_bin_count)

#     def predict(self, X):
#         preds = [self.maj_class for _ in X]
#         return preds

class HoefdingTreesEDDMComplete:
    def __init__(self):
        self.model = HoeffdingTreeClassifier()
        self.eddm =  EDDM()
        self.stored_elements_X = []
        self.stored_elements_y = []
        self.warm_up = True
        self.element_count = 0
        self.change_points = []

    def partial_fit(self, X, y, classes=None):

        # add elements to the window
        rows, _ = X.shape

        if self.warm_up:
            self.model.partial_fit(X,y)
            self.warm_up = False
        else:
            for i in range(rows):
                self.element_count+=1
                pred, truth = self.model.predict(np.asarray([X[i]])), np.asarray([y[i]])
                self.eddm.add_element(1 if pred==truth else 0)
                if self.eddm.detected_warning_zone():

                    # start storing elements
                    if isinstance(self.stored_elements_X,list):
                        self.stored_elements_X = np.asarray([X[i]])
                        self.stored_elements_y = np.asarray([y[i]])
                    else:
                        self.stored_elements_X = np.concatenate([self.stored_elements_X, np.asarray([X[i]])],axis=0)
                        self.stored_elements_y = np.concatenate([self.stored_elements_y, np.asarray([y[i]])],axis=0)

                elif self.eddm.detected_change():

                    if isinstance(self.stored_elements_X,list):
                        self.stored_elements_X = np.asarray([X[i]])
                        self.stored_elements_y = np.asarray([y[i]])
                    else:
                        self.stored_elements_X = np.concatenate([self.stored_elements_X, np.asarray([X[i]])],axis=0)
                        self.stored_elements_y = np.concatenate([self.stored_elements_y, np.asarray([y[i]])],axis=0)
                        
                    rows_X, _ = self.stored_elements_X.shape

                    if rows_X == 1:
                        self.model.partial_fit(self.stored_elements_X, self.stored_elements_y)
                    else:
                        wind = min(rows_X, 1000)
                        self.model = HoeffdingTreeClassifier()
                        self.model.partial_fit(self.stored_elements_X[-wind:], self.stored_elements_y[-wind:])
                        self.stored_elements_X = []
                        self.stored_elements_y = []
                        self.change_points.append(self.element_count)

                else:
                    self.stored_elements_X = []
                    self.stored_elements_y = []
                    # fit the model with bs 1 incrementally
                    self.model.partial_fit(np.asarray([X[i]]), np.asarray([y[i]]))
            
    
    def predict(self, X):
        preds = self.model.predict(X)
        return preds


class HoefdingTreesADWINComplete:
    def __init__(self):
        self.model = HoeffdingTreeClassifier()
        self.adwin =  ADWIN()
        self.stored_elements_X = []
        self.stored_elements_y = []
        self.warm_up = True
        self.store_size = 2000 #20000
        self.element_count = 0
        self.change_points = []

    def partial_fit(self, X, y, classes=None):
        # add elements to the window
        rows, _ = X.shape

        if self.warm_up:
            self.model.partial_fit(X,y)
            self.warm_up = False
        else:
            for i in range(rows):
                self.element_count+=1
                if isinstance(self.stored_elements_X, list):
                    X_rows = 0
                else:
                    X_rows, _ = self.stored_elements_X.shape
                
                pred, truth = self.model.predict(np.asarray([X[i]])), np.asarray([y[i]])
                self.adwin.add_element(1 if pred==truth else 0)
                    
                
                # udpate the stored elements till capacity reached and after maintain a buffer of recent values
                if isinstance(self.stored_elements_X, list):
                    self.stored_elements_X = np.asarray([X[i]])
                    self.stored_elements_y = np.asarray([y[i]])
                else:
                    if X_rows>=self.store_size:
                        # take a slice from 1st element
                        self.stored_elements_X = np.concatenate([self.stored_elements_X[1:], np.asarray([X[i]])],axis=0)
                        self.stored_elements_y = np.concatenate([self.stored_elements_y[1:], np.asarray([y[i]])],axis=0)
                    else:
                        self.stored_elements_X = np.concatenate([self.stored_elements_X, np.asarray([X[i]])],axis=0)
                        self.stored_elements_y = np.concatenate([self.stored_elements_y, np.asarray([y[i]])],axis=0)
                # fit model sample by sample
                self.model.partial_fit(np.asarray([X[i]]),np.asarray([y[i]]))

                if self.adwin.detected_change():
                    self.change_points.append(self.element_count)
                    X_rows, _ = self.stored_elements_X.shape
                    # remove oldest element
                    if self.adwin.width < self.store_size:
                        
                        # start removing oldest samples (samples from the front of the store)
                        del_samples = self.store_size - self.adwin.width
                        self.stored_elements_X = self.stored_elements_X[del_samples:]
                        self.stored_elements_y = self.stored_elements_y[del_samples:]
                        
                        # reset model
                        self.model = HoeffdingTreeClassifier()
                        rows_X, _ = self.stored_elements_X.shape
                        offset = min(rows_X, 1000)
                        self.model.fit(self.stored_elements_X[-offset:], self.stored_elements_y[-offset:])

    
    def predict(self, X):
        preds = self.model.predict(X)
        return preds
